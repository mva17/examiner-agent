"""Демо-модель без ключа и без интернета (LLM_PROVIDER=fake).

Это НЕ нейросеть, а набор простых правил, который отвечает в тех же
JSON-форматах, что и настоящая модель. Нужна, чтобы:
  * показать работу графа, API и чата, когда ключа ещё нет;
  * прогонять тесты бесплатно и воспроизводимо.
Для метрик качества её использовать нельзя — только настоящую модель.
"""

from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage

from .text import best_sentence, content_stems, looks_like_injection

PHYSICS_STEMS = {
    content_stems(w)[0]
    for w in """физика маятник микрометр штангенциркуль нониус погрешность измерение период колебания
    сила масса энергия импульс скорость ускорение ток напряжение сопротивление заряд конденсатор
    магнит индукция оптика линза свет волна давление газ температура теплоемкость плотность
    гравитация трение момент инерции вращение ньютон закон кулон ом джоуль резонанс дифракция
    интерференция поляризация спектр атом ядро излучение механика кинематика динамика
    термодинамика электричество электростатика гироскоп упругость деформация""".split()
}

_TAG = re.compile(r"\[\[TASK:(\w+)\]\]")
_PREFIX = re.compile(
    r"^(пожалуйста\s+)?(проверь(те)?|спроси(те)?|подготовь(те)?|погоняй|экзамен\w*|хочу\s+подготовиться)"
    r"(\s+меня)?(\s+к\s+допуску)?(\s+по)?(\s+теме|\s+работе|\s+лабораторной)?\s*",
    re.IGNORECASE,
)


def _between(text: str, tag: str) -> list[str]:
    return [m.strip() for m in re.findall(rf"<{tag}>\n?(.*?)\n?</{tag}>", text, re.DOTALL)]


def _overlap(a: str, b: str) -> float:
    ref = set(content_stems(b))
    if not ref:
        return 0.0
    return len(ref & set(content_stems(a))) / len(ref)


def _first_sentence(text: str) -> str:
    for sent in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")):
        if len(sent) > 40:
            return sent.strip()
    return text.strip()[:200]


class FakeExamLLM:
    """Совместима с вызовом llm.invoke(messages) у моделей LangChain."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def invoke(self, messages, *args, **kwargs) -> AIMessage:
        system = messages[0].content
        user = messages[1].content if len(messages) > 1 else ""
        task = (_TAG.search(system) or [None, "unknown"])[1]
        self.calls.append(task)
        handler = getattr(self, f"_{task}", None)
        payload = handler(user) if handler else {"error": f"unknown task {task}"}
        return AIMessage(content=payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False))

    # --- обработчики задач -------------------------------------------

    def _extract_topic(self, user: str) -> dict:
        request = (_between(user, "student_request") or [user])[0]
        topic = _PREFIX.sub("", request.strip().rstrip("?.!")).strip() or request.strip()
        stems = set(content_stems(request))
        is_physics = bool(stems & PHYSICS_STEMS)
        return {"topic": topic[:120], "is_physics": is_physics, "search_query": topic[:200], "reason": "демо-правило"}

    def _check_relevance(self, user: str) -> dict:
        topic = user.split("\n", 1)[0].removeprefix("Тема:").strip()
        best_work, best = None, 0.0
        for frag in _between(user, "manual_fragment"):
            header, _, body = frag.partition("\n")
            score = _overlap(f"{header} {body}", topic)
            work = re.search(r"работа (\S+),", header)
            if score > best and work:
                best, best_work = score, work.group(1)
        relevant = best >= 0.5
        return {"relevant": relevant, "work": best_work if relevant else None, "reason": f"совпадение {best:.2f}"}

    def _rewrite_query(self, user: str) -> dict:
        topic = user.removeprefix("Тема:").strip()
        return {"query": f"{topic} прибор формула закон"}

    def _generate_from_manual(self, user: str) -> dict:
        questions = []
        for frag in _between(user, "manual_fragment"):
            header, _, body = frag.partition("\n")
            page = re.search(r"с\. (\d+)", header)
            section = re.search(r"«(.+?)»", header)
            sentence = _first_sentence(body)
            name = section.group(1) if section else "этой части работы"
            questions.append(
                {
                    "text": f"Объясните своими словами, что в методичке сказано в разделе «{name}».",
                    "reference_answer": sentence,
                    "page": int(page.group(1)) if page else None,
                }
            )
        return {"questions": questions[:3] or [{"text": "Какова цель работы?", "reference_answer": "см. методичку", "page": 1}]}

    def _generate_general(self, user: str) -> dict:
        topic = user.removeprefix("Тема:").strip()
        return {
            "questions": [
                {
                    "text": f"Дайте определение: что такое «{topic}»?",
                    "reference_answer": f"Определение понятия «{topic}» из курса общей физики.",
                    "page": None,
                },
                {
                    "text": f"Какие физические величины и единицы СИ связаны с темой «{topic}»?",
                    "reference_answer": f"Основные величины темы «{topic}» и их единицы СИ.",
                    "page": None,
                },
            ]
        }

    def _verify_question(self, user: str) -> dict:
        return {"ok": True, "reason": "демо-проверка"}

    def _grade_answer(self, user: str) -> dict:
        answer = (_between(user, "student_answer") or [""])[-1]
        question = (_between(user, "question") or [""])[0]
        fragments = [f.partition("\n")[2] for f in _between(user, "manual_fragment")]
        reference = " ".join(_between(user, "reference"))
        if not reference and fragments:
            reference = best_sentence(question, " ".join(fragments))
        page = re.search(r"с\. (\d+)", user)
        page_num = int(page.group(1)) if page else None
        if looks_like_injection(answer):
            return {"verdict": "неверно", "explanation": "Это не ответ по существу.", "missing": [], "page": page_num, "followup_question": None}
        score = _overlap(answer, reference)
        if score >= 0.5:
            return {"verdict": "верно", "explanation": "Ключевые элементы совпадают с методичкой.", "missing": [], "page": page_num, "followup_question": None}
        if score >= 0.2:
            return {
                "verdict": "частично",
                "explanation": f"Есть верная часть, но не всё. В методичке: {reference[:200]}",
                "missing": ["недостающие элементы эталона"],
                "page": page_num,
                "followup_question": "Дополните ответ: что ещё важно сказать по этому вопросу?",
            }
        return {"verdict": "неверно", "explanation": f"Не совпадает с методичкой. Правильно: {reference[:200]}", "missing": [], "page": page_num, "followup_question": None}

    def _baseline_grade(self, user: str) -> str:
        return "верно"
