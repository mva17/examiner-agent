"""Агент-Экзаменатор: граф LangGraph.

    START → guard_input → extract_topic ─┬─ не физика ──────────────→ refuse → END
                                         └─ физика → retrieve → check_relevance
           ┌──── не по теме, попытки есть ── rewrite_query ←──────────┤
           │                                                          ├─ по теме → plan_manual ─┐
           └→ retrieve                      попытки кончились → plan_general ──────────────────┤
                                                                                                 ▼
             ┌──────────────────────────────── pick ── вопросов нет / лимит ──→ summarize → END
             │                                  │
             │                                 ask (interrupt: ждём ответ студента)
             │                                  │
             │                                grade ─ «стоп» / лимит вызовов ─→ summarize
             │                                  │
             └──── верно / неверно ─────────────┤
                                                └─ частично → followup → ask

Ход сессии сохраняется checkpointer'ом по thread_id = "<user_id>:<chat_id>",
слабые темы — в долгой памяти (memory.py) по user_id.
"""

from __future__ import annotations

import operator
import re
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Annotated, Any, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel

from . import logic, prompts
from .config import Settings, get_settings
from .knowledge import CONTROL_SECTION, Hit, ManualSearch, build_search
from .llm import BudgetExceeded, CallBudget, LLMError, ask_json, build_llm
from .memory import WeakTopicMemory
from .schemas import (
    SCORE,
    AnswerRecord,
    Grade,
    Question,
    QuestionBatch,
    QueryRewrite,
    RelevanceVerdict,
    Source,
    TopicExtraction,
)
from .text import best_sentence, content_stems, looks_like_injection
from .verifier import LLMQuestionVerifier, QuestionVerifier

Status = Literal["running", "asking", "finished", "refused", "error"]


class ExamState(TypedDict, total=False):
    user_id: str
    chat_id: str
    request: str
    topic: str
    is_physics: bool
    search_query: str
    search_attempts: int
    hits: list[dict]
    relevant: bool
    work: str | None
    notice: str
    queue: list[dict]
    asked_ids: list[str]
    current: dict | None
    answer: str
    last_grade: dict | None
    history: list[dict]
    llm_calls: int
    status: Status
    final_message: str
    stopped_early: bool
    events: Annotated[list[dict], operator.add]  # журнал шагов: узел, время, вызовы модели, ошибки


@dataclass
class ExaminerDeps:
    llm: Any
    search: ManualSearch
    memory: WeakTopicMemory
    verifier: QuestionVerifier = field(default_factory=LLMQuestionVerifier)
    settings: Settings = field(default_factory=get_settings)

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "ExaminerDeps":
        settings = settings or get_settings()
        return cls(
            llm=build_llm(settings),
            search=build_search(settings),
            memory=WeakTopicMemory(settings.memory_db),
            settings=settings,
        )


@dataclass
class Run:
    """Контекст одного шага: модель, лимит вызовов, журнал."""

    llm: Any
    budget: CallBudget
    trace: list[dict] = field(default_factory=list)


REFUSAL_NOT_PHYSICS = (
    "Я Экзаменатор лабораторного практикума по физике и готовлю только к допуску к лабораторным. "
    "Напишите тему по физике, например: «Проверь меня по теме микрометр» или «маятник»."
)


def _dump(items: list[BaseModel]) -> list[dict]:
    return [i.model_dump(mode="json") for i in items]


def _hits_to_dicts(hits: list[Hit]) -> list[dict]:
    return [{**h.source.model_dump(mode="json"), "score": h.score, "coverage": round(h.coverage, 3)} for h in hits]


def _sources(hits: list[dict]) -> list[Source]:
    return [Source.model_validate({k: v for k, v in h.items() if k in Source.model_fields}) for h in hits]


def build_graph(deps: ExaminerDeps, checkpointer=None):
    s = deps.settings

    def node(name: str):
        """Обёртка узла: лимит вызовов модели, журнал шагов, обработка ошибок кодом."""

        def decorator(fn):
            @wraps(fn)
            def wrapper(state: ExamState) -> dict:
                run = Run(deps.llm, CallBudget(s.max_llm_calls, state.get("llm_calls", 0)))
                started = time.perf_counter()
                try:
                    update = fn(state, run) or {}
                    ok, error = True, ""
                except BudgetExceeded as exc:
                    # Лимит вызовов: не падаем, а подводим итог по уже отвеченным вопросам.
                    update = {"status": "running", "stopped_early": True, "notice": str(exc)}
                    ok, error = False, str(exc)
                except LLMError as exc:
                    update = {
                        "status": "error",
                        "final_message": f"Не получилось обратиться к модели ({exc}). Попробуйте ещё раз чуть позже.",
                    }
                    ok, error = False, str(exc)
                update["llm_calls"] = run.budget.used
                update["events"] = [
                    {
                        "node": name,
                        "ok": ok,
                        "error": error,
                        "elapsed_s": round(time.perf_counter() - started, 3),
                        "llm_calls": run.budget.used - state.get("llm_calls", 0),
                        "steps": run.trace,
                    }
                ]
                return update

            return wrapper

        return decorator

    # --- Узлы ----------------------------------------------------------

    @node("guard_input")
    def guard_input(state: ExamState, run: Run) -> dict:
        request = (state.get("request") or "").strip()
        if not request:
            return {"status": "refused", "final_message": "Напишите, по какой теме вас проверить."}
        if len(request) > s.max_answer_chars:
            return {"status": "refused", "final_message": "Запрос слишком длинный: опишите тему в одном-двух предложениях."}
        if looks_like_injection(request):
            return {
                "status": "refused",
                "final_message": "В запросе есть инструкции для системы — я их не выполняю. Просто назовите тему по физике.",
            }
        return {"status": "running"}

    @node("extract_topic")
    def extract_topic(state: ExamState, run: Run) -> dict:
        system, user = prompts.extract_topic(state["request"])
        t = ask_json(run.llm, system, user, TopicExtraction, run.budget, trace=run.trace, step="extract_topic")
        update: dict = {"topic": t.topic, "is_physics": t.is_physics, "search_query": t.search_query}
        if not t.is_physics:
            update.update(status="refused", final_message=REFUSAL_NOT_PHYSICS)
        return update

    @node("retrieve")
    def retrieve(state: ExamState, run: Run) -> dict:
        hits = deps.search.search(state["search_query"], k=5)
        return {"hits": _hits_to_dicts(hits), "search_attempts": state.get("search_attempts", 0) + 1}

    @node("check_relevance")
    def check_relevance(state: ExamState, run: Run) -> dict:
        hits = state.get("hits", [])
        topic_terms = f"{state['topic']} {state['search_query']}"
        # 1) Дешёвый фильтр: если ни одно слово темы не встречается во фрагментах — модель не зовём.
        candidates = [h for h in hits if h["coverage"] > 0]
        if not candidates:
            return {"relevant": False, "work": None}
        # 2) Модель решает, действительно ли это материал по теме, а не совпадение слов.
        system, user = prompts.check_relevance(topic_terms, _sources(candidates))
        v = ask_json(run.llm, system, user, RelevanceVerdict, run.budget, trace=run.trace, step="check_relevance")
        works = {h["work"] for h in candidates}
        work = v.work if v.work in works else candidates[0]["work"]
        return {"relevant": v.relevant, "work": work if v.relevant else None}

    @node("rewrite_query")
    def rewrite_query(state: ExamState, run: Run) -> dict:
        system, user = prompts.rewrite_query(state["topic"], state["search_query"])
        r = ask_json(run.llm, system, user, QueryRewrite, run.budget, trace=run.trace, step="rewrite_query")
        if set(content_stems(r.query)) == set(content_stems(state["search_query"])):
            # Модель повторила запрос: второй поиск ничего не даст, сразу к вопросам «не из методички».
            return {"search_attempts": s.max_search_attempts}
        return {"search_query": r.query}

    @node("plan_manual")
    def plan_manual(state: ExamState, run: Run) -> dict:
        work, topic = state["work"], state["topic"]
        control = logic.control_questions_to_q(deps.search.control_questions(work), topic)
        for q in control:
            # Где в методичке ответ на контрольный вопрос — ищем внутри той же работы.
            ev = [h.source for h in deps.search.search(q.text, k=3, work=work)]
            q.evidence = [e for e in ev if not e.section.lower().startswith(CONTROL_SECTION)][:2]
        fragments = [
            src
            for src in _sources(state.get("hits", []))
            if src.work == work and not src.section.lower().startswith(CONTROL_SECTION)
        ][:3]
        generated: list[Question] = []
        need = max(2, s.max_questions - len(control))
        if fragments:
            system, user = prompts.generate_from_manual(topic, fragments, need)
            batch = ask_json(run.llm, system, user, QuestionBatch, run.budget, trace=run.trace, step="generate_from_manual")
            pages = {f.page: f for f in fragments}
            for g in batch.questions:
                src = pages.get(g.page) or fragments[0]
                generated.append(
                    Question(
                        id=logic.question_id(g.text, work),
                        text=g.text,
                        reference_answer=g.reference_answer,
                        topic=topic,
                        origin="manual",
                        kind="generated",
                        source=src,
                        evidence=[src],
                    )
                )
        mistakes = [q for q in deps.memory.mistakes(state["user_id"]) if q.source and q.source.work == work]
        queue = logic.order_questions(control, generated, mistakes, s.max_questions)
        notice = f"Вопросы по методичке: работа {work}."
        if mistakes:
            notice += f" Сначала повторим {len(mistakes)} вопрос(а), где раньше были ошибки."
        return {"queue": _dump(queue), "notice": notice}

    @node("plan_general")
    def plan_general(state: ExamState, run: Run) -> dict:
        topic = state["topic"]
        system, user = prompts.generate_general(topic, s.max_questions)
        batch = ask_json(run.llm, system, user, QuestionBatch, run.budget, trace=run.trace, step="generate_general")
        accepted: list[Question] = []
        for g in batch.questions:
            q = Question(
                id=logic.question_id(g.text, "general"),
                text=g.text,
                reference_answer=g.reference_answer,
                topic=topic,
                origin="general",
                kind="generated",
            )
            verdict = deps.verifier.verify(q, run)  # точка подключения агента-Проверяющего
            run.trace.append({"step": "verifier", "ok": verdict.ok, "reason": verdict.reason, "question": q.text})
            if verdict.ok:
                q.verified = True
                accepted.append(q)
        if not accepted:
            return {
                "status": "refused",
                "final_message": f"Темы «{topic}» нет в методичках, а надёжных вопросов по ней составить не удалось. "
                "Уточните тему или назовите лабораторную работу.",
            }
        mistakes = [q for q in deps.memory.mistakes(state["user_id"]) if q.origin == "general" and q.topic == topic]
        queue = logic.order_questions([], accepted, mistakes, s.max_questions)
        return {
            "queue": _dump(queue),
            "notice": f"Темы «{topic}» нет в методичках. Вопросы составлены моделью и помечены «не из методички»; "
            f"проверку прошли {len(accepted)} из {len(batch.questions)}.",
        }

    @node("pick")
    def pick(state: ExamState, run: Run) -> dict:
        history = [AnswerRecord.model_validate(r) for r in state.get("history", [])]
        answered = sum(1 for r in history if r.kind != "followup")
        if state.get("stopped_early") or answered >= s.max_questions:
            return {"current": None}
        queue = [Question.model_validate(q) for q in state.get("queue", [])]
        nxt = logic.pick_next(queue, history, set(state.get("asked_ids", [])))
        if nxt is None:
            return {"current": None}
        return {"current": nxt.model_dump(mode="json"), "asked_ids": [*state.get("asked_ids", []), nxt.id]}

    def ask(state: ExamState) -> dict:
        # Human-in-the-loop: граф останавливается и ждёт ответ студента
        # (продолжение — graph.invoke(Command(resume=ответ), config)).
        q = Question.model_validate(state["current"])
        history = state.get("history", [])
        answer = interrupt(
            {
                "type": "question",
                "number": sum(1 for r in history if r.get("kind") != "followup") + (q.kind != "followup"),
                "max_questions": s.max_questions,
                "question": q.text,
                "kind": q.kind,
                "origin": q.origin,
                "label": q.label,
                "source": q.source.model_dump(mode="json") if q.source else None,
                "topic": state.get("topic", ""),
                "notice": state.get("notice", ""),
                "last_grade": state.get("last_grade"),
            }
        )
        return {"answer": str(answer)[: s.max_answer_chars], "status": "running"}

    @node("grade")
    def grade(state: ExamState, run: Run) -> dict:
        q = Question.model_validate(state["current"])
        answer = state.get("answer", "")
        if logic.is_stop(answer):
            return {"stopped_early": True, "last_grade": None}
        evidence = q.evidence or ([q.source] if q.source else [])
        page = evidence[0].page if evidence else None
        if logic.is_dont_know(answer):
            hint = q.reference_answer or (best_sentence(q.text, evidence[0].quote) if evidence else "")
            g = Grade(verdict="неверно", explanation=f"Ответа нет. Правильно: {hint}".strip(), page=page)
        elif looks_like_injection(answer):
            g = Grade(
                verdict="неверно",
                explanation="В ответе инструкция для системы вместо ответа по существу — так оценку не изменить.",
                page=page,
            )
        else:
            system, user = prompts.grade_answer(q, answer, evidence)
            g = ask_json(run.llm, system, user, Grade, run.budget, trace=run.trace, step="grade_answer")
            if g.page is None:
                g.page = page
            if q.origin == "manual" and evidence and g.page not in {e.page for e in evidence} | {q.source.page if q.source else None}:
                g.page = page  # страницу берём только из найденных фрагментов, а не «из головы» модели
        record = AnswerRecord(
            question_id=q.id,
            question=q.text,
            topic=q.topic,
            answer=answer,
            verdict=g.verdict,
            explanation=g.explanation,
            page=g.page,
            kind=q.kind,
        )
        deps.memory.record(state["user_id"], q, SCORE[g.verdict])
        grade_view = {**g.model_dump(mode="json"), "question": q.text, "label": q.label, "origin": q.origin}
        return {"history": [*state.get("history", []), record.model_dump(mode="json")], "last_grade": grade_view}

    @node("followup")
    def followup(state: ExamState, run: Run) -> dict:
        parent = Question.model_validate(state["current"])
        g = state["last_grade"]
        text = g["followup_question"]
        q = Question(
            id=logic.question_id(text, parent.id),
            text=text,
            reference_answer="; ".join(
                [parent.reference_answer or (parent.evidence[0].quote if parent.evidence else ""), *g.get("missing", [])]
            ).strip("; "),
            topic=parent.topic,
            origin=parent.origin,
            kind="followup",
            source=parent.source,
            evidence=parent.evidence,
            parent_id=parent.id,
        )
        return {"current": q.model_dump(mode="json"), "asked_ids": [*state.get("asked_ids", []), q.id]}

    @node("summarize")
    def summarize(state: ExamState, run: Run) -> dict:
        history = [AnswerRecord.model_validate(r) for r in state.get("history", [])]
        text = logic.summary_text(state.get("topic", "?"), history, bool(state.get("stopped_early")))
        weak = deps.memory.weak_topics(state["user_id"])
        if weak:
            text += "\nСлабые темы по всем сессиям: " + ", ".join(w["topic"] for w in weak) + "."
        return {"status": "finished", "final_message": text, "current": None}

    @node("refuse")
    def refuse(state: ExamState, run: Run) -> dict:
        return {"status": "refused"}

    # --- Условные переходы --------------------------------------------

    def halt(state: ExamState) -> str | None:
        """Общие остановки: ошибка модели → END, лимит вызовов или «стоп» → итог."""
        if state.get("status") == "error":
            return END
        if state.get("status") == "refused":
            return "refuse"
        if state.get("stopped_early"):
            return "summarize"
        return None

    def after_guard(state: ExamState) -> str:
        return halt(state) or "extract_topic"

    def after_topic(state: ExamState) -> str:
        return halt(state) or "retrieve"

    def after_relevance(state: ExamState) -> str:
        if stop := halt(state):
            return stop
        if state.get("relevant"):
            return "plan_manual"
        if state.get("search_attempts", 0) < s.max_search_attempts:
            return "rewrite_query"
        return "plan_general"

    def after_rewrite(state: ExamState) -> str:
        if stop := halt(state):
            return stop
        if state.get("search_attempts", 0) >= s.max_search_attempts:
            return "plan_general"
        return "retrieve"

    def after_plan(state: ExamState) -> str:
        return halt(state) or "pick"

    def after_pick(state: ExamState) -> str:
        return "ask" if state.get("current") else "summarize"

    def after_grade(state: ExamState) -> str:
        if stop := halt(state):
            return stop
        g = state.get("last_grade") or {}
        current = state.get("current") or {}
        if g.get("verdict") == "частично" and g.get("followup_question") and current.get("kind") != "followup":
            return "followup"
        return "pick"

    def after_followup(state: ExamState) -> str:
        return halt(state) or "ask"

    g = StateGraph(ExamState)
    for name, fn in [
        ("guard_input", guard_input),
        ("extract_topic", extract_topic),
        ("retrieve", retrieve),
        ("check_relevance", check_relevance),
        ("rewrite_query", rewrite_query),
        ("plan_manual", plan_manual),
        ("plan_general", plan_general),
        ("pick", pick),
        ("ask", ask),
        ("grade", grade),
        ("followup", followup),
        ("summarize", summarize),
        ("refuse", refuse),
    ]:
        g.add_node(name, fn)

    stops = ["refuse", "summarize", END]
    g.add_edge(START, "guard_input")
    g.add_conditional_edges("guard_input", after_guard, ["extract_topic", *stops])
    g.add_conditional_edges("extract_topic", after_topic, ["retrieve", *stops])
    g.add_edge("retrieve", "check_relevance")
    g.add_conditional_edges("check_relevance", after_relevance, ["plan_manual", "rewrite_query", "plan_general", *stops])
    g.add_conditional_edges("rewrite_query", after_rewrite, ["retrieve", "plan_general", *stops])
    g.add_conditional_edges("plan_manual", after_plan, ["pick", *stops])
    g.add_conditional_edges("plan_general", after_plan, ["pick", *stops])
    g.add_conditional_edges("pick", after_pick, ["ask", "summarize"])
    g.add_edge("ask", "grade")
    g.add_conditional_edges("grade", after_grade, ["followup", "pick", *stops])
    g.add_conditional_edges("followup", after_followup, ["ask", *stops])
    g.add_edge("summarize", END)
    g.add_edge("refuse", END)
    return g.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver())


# --- Удобная обёртка для API, CLI и тестов --------------------------------

_ID = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


class ExamTurn(BaseModel):
    status: Literal["question", "finished", "refused", "error"]
    message: str = ""
    question: dict | None = None
    grade: dict | None = None
    notice: str = ""
    topic: str = ""
    score: float | None = None
    total: int | None = None
    sources: list[dict] = []
    events: list[dict] = []
    llm_calls: int = 0


class SessionConflict(RuntimeError):
    """Попытка начать новую сессию в чате, где ещё ждут ответа на вопрос."""


class Examiner:
    RECURSION_LIMIT = 100  # гардрейл LangGraph: максимум шагов графа за один вызов

    def __init__(self, deps: ExaminerDeps, checkpointer=None):
        self.deps = deps
        self.graph = build_graph(deps, checkpointer)

    @staticmethod
    def thread_id(user_id: str, chat_id: str) -> str:
        if not (_ID.match(user_id) and _ID.match(chat_id)):
            raise ValueError("user_id и chat_id: латиница, цифры, - и _, до 64 символов")
        return f"{user_id}:{chat_id}"

    def _config(self, user_id: str, chat_id: str) -> dict:
        return {"configurable": {"thread_id": self.thread_id(user_id, chat_id)}, "recursion_limit": self.RECURSION_LIMIT}

    def is_waiting(self, user_id: str, chat_id: str) -> bool:
        snapshot = self.graph.get_state(self._config(user_id, chat_id))
        return any(t.interrupts for t in snapshot.tasks)

    def start(self, user_id: str, chat_id: str, request: str) -> ExamTurn:
        config = self._config(user_id, chat_id)
        if self.is_waiting(user_id, chat_id):
            raise SessionConflict("в этом чате уже идёт сессия: ответьте на вопрос или начните новый чат")
        initial: ExamState = {
            "user_id": user_id,
            "chat_id": chat_id,
            "request": request,
            "search_attempts": 0,
            "hits": [],
            "relevant": False,
            "work": None,
            "notice": "",
            "queue": [],
            "asked_ids": [],
            "current": None,
            "answer": "",
            "last_grade": None,
            "history": [],
            "llm_calls": 0,
            "status": "running",
            "final_message": "",
            "stopped_early": False,
            "events": [],
        }
        return self._turn(self.graph.invoke(initial, config), config)

    def answer(self, user_id: str, chat_id: str, answer: str) -> ExamTurn:
        config = self._config(user_id, chat_id)
        if not self.is_waiting(user_id, chat_id):
            raise SessionConflict("в этом чате нет вопроса, ожидающего ответа: начните сессию")
        return self._turn(self.graph.invoke(Command(resume=answer), config), config)

    def _turn(self, result: dict, config: dict) -> ExamTurn:
        state = self.graph.get_state(config).values
        history = [AnswerRecord.model_validate(r) for r in state.get("history", [])]
        score, total = logic.score_of(history)
        common = dict(
            topic=state.get("topic", ""),
            notice=state.get("notice", ""),
            grade=state.get("last_grade"),
            score=score if total else None,
            total=total or None,
            sources=[{k: h[k] for k in ("work", "title", "page", "section")} for h in state.get("hits", [])][:5],
            events=state.get("events", [])[-30:],
            llm_calls=state.get("llm_calls", 0),
        )
        interrupts = result.get("__interrupt__") or []
        if interrupts:
            payload = interrupts[0].value
            return ExamTurn(status="question", message=payload["question"], question=payload, **common)
        status = state.get("status", "error")
        status = status if status in ("finished", "refused", "error") else "error"
        return ExamTurn(status=status, message=state.get("final_message", ""), **common)
