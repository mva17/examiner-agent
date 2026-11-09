"""Детерминированная логика Экзаменатора (без вызовов модели, легко тестировать).

Здесь: порядок вопросов, выбор следующего вопроса с учётом ошибок,
команды студента («стоп», «не знаю»), итог сессии.
"""

from __future__ import annotations

import hashlib
import re

from .schemas import SCORE, AnswerRecord, Question, Source

STOP_WORDS = {"стоп", "хватит", "закончить", "конец", "завершить", "stop", "/stop", "выход"}
DONT_KNOW = re.compile(r"^\s*(не\s+знаю|без\s+понятия|хз|не\s+помню|пропуст\w*|\?+|-+)\s*[.!]*\s*$", re.IGNORECASE)


def question_id(text: str, work: str = "") -> str:
    """Стабильный id: один и тот же вопрос в разных сессиях получает один id (нужно для памяти)."""
    norm = " ".join(text.lower().split())
    return hashlib.sha1(f"{work}|{norm}".encode()).hexdigest()[:12]


def is_stop(answer: str) -> bool:
    return answer.strip().lower().strip(".!") in STOP_WORDS


def is_dont_know(answer: str) -> bool:
    return not answer.strip() or bool(DONT_KNOW.match(answer))


def order_questions(
    control: list[Question],
    generated: list[Question],
    prior_mistakes: list[Question],
    limit: int,
) -> list[Question]:
    """Очередь вопросов: сначала прошлые ошибки, затем «Контрольные вопросы», затем сгенерированные."""
    queue: list[Question] = []
    seen: set[str] = set()
    for q in [*prior_mistakes, *control, *generated]:
        if q.id not in seen:
            seen.add(q.id)
            queue.append(q)
    # Запас больше лимита: из запаса выбираются вопросы по слабым местам.
    return queue[: max(limit * 2, limit)]


def _section(q: Question) -> str:
    if q.evidence:
        return f"{q.evidence[0].work}|{q.evidence[0].section}"
    return q.topic


def pick_next(queue: list[Question], history: list[AnswerRecord], asked_ids: set[str]) -> Question | None:
    """Следующий вопрос. Если в прошлом ответе ошибка — сначала вопрос по тому же разделу методички."""
    pending = [q for q in queue if q.id not in asked_ids]
    if not pending:
        return None
    by_id = {q.id: q for q in queue}
    weak_sections = [
        _section(by_id[r.question_id])
        for r in reversed(history)
        if r.verdict != "верно" and r.question_id in by_id
    ]
    for section in weak_sections:
        for q in pending:
            if _section(q) == section:
                return q
    return pending[0]


def control_questions_to_q(sources: list[Source], topic: str) -> list[Question]:
    return [
        Question(
            id=question_id(s.quote, s.work),
            text=s.quote,
            topic=topic,
            origin="manual",
            kind="control",
            source=s,
        )
        for s in sources
    ]


def score_of(records: list[AnswerRecord]) -> tuple[float, int]:
    main = [r for r in records if r.kind != "followup"]
    return sum(SCORE[r.verdict] for r in main), len(main)


def summary_text(topic: str, records: list[AnswerRecord], stopped_early: bool = False) -> str:
    score, total = score_of(records)
    if total == 0:
        return f"Сессия по теме «{topic}» завершена: ни на один вопрос не было ответа."
    lines = [f"Итог по теме «{topic}»: {score:g} из {total}."]
    if stopped_early:
        lines.append("Сессия остановлена досрочно.")
    weak = [r for r in records if r.verdict != "верно" and r.kind != "followup"]
    if weak:
        lines.append("Повторите:")
        for r in weak:
            page = f" (с. {r.page})" if r.page else ""
            lines.append(f"• {r.question}{page}")
    else:
        lines.append("Все ответы верные — к допуску по этой теме вы готовы.")
    return "\n".join(lines)
