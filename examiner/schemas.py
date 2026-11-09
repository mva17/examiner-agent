"""Pydantic-схемы: всё, что приходит от модели, проверяется по ним.

Если модель вернула невалидный JSON или лишние/неверные поля, llm.ask_json
показывает ей ошибку и просит исправить (не больше JSON_RETRIES раз).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Verdict = Literal["верно", "частично", "неверно"]
Origin = Literal["manual", "general"]  # из методички / сгенерировано моделью
QuestionKind = Literal["control", "generated", "followup"]

SCORE: dict[str, float] = {"верно": 1.0, "частично": 0.5, "неверно": 0.0}


class Source(BaseModel):
    """Ссылка на методичку. Поля совпадают с метаданными в базе Оскара."""

    work: str = Field(description="номер или код лабораторной работы")
    title: str = ""
    page: int | None = None
    section: str = ""
    quote: str = Field(default="", max_length=1200)


# --- Ответы модели --------------------------------------------------------


class TopicExtraction(BaseModel):
    topic: str = Field(min_length=1, max_length=120)
    is_physics: bool
    search_query: str = Field(min_length=1, max_length=200)
    reason: str = Field(default="", max_length=400)

    @field_validator("topic", "search_query")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("пустая строка")
        return value


class RelevanceVerdict(BaseModel):
    relevant: bool
    work: str | None = Field(default=None, description="код работы, если найдено по теме")
    reason: str = Field(default="", max_length=400)


class QueryRewrite(BaseModel):
    query: str = Field(min_length=1, max_length=200)


class GeneratedQuestion(BaseModel):
    text: str = Field(min_length=5, max_length=500)
    reference_answer: str = Field(min_length=1, max_length=1200)
    page: int | None = None


class QuestionBatch(BaseModel):
    questions: list[GeneratedQuestion] = Field(min_length=1, max_length=6)


class VerifierVerdict(BaseModel):
    ok: bool
    reason: str = Field(default="", max_length=400)


class Grade(BaseModel):
    verdict: Verdict
    explanation: str = Field(min_length=1, max_length=1200)
    missing: list[str] = Field(default_factory=list, max_length=5)
    page: int | None = None
    followup_question: str | None = Field(default=None, max_length=400)

    @field_validator("verdict", mode="before")
    @classmethod
    def _normalize(cls, value: object) -> object:
        # Qwen иногда пишет «Верно.» или «частично верно».
        if isinstance(value, str):
            low = value.strip().lower().rstrip(".!")
            for verdict in ("неверно", "частично", "верно"):
                if low.startswith(verdict):
                    return verdict
        return value


# --- Внутренние объекты агента -------------------------------------------


class Question(BaseModel):
    id: str
    text: str
    reference_answer: str = ""
    topic: str
    origin: Origin
    kind: QuestionKind
    source: Source | None = None  # откуда вопрос (для «Контрольных вопросов» — их страница)
    evidence: list[Source] = Field(default_factory=list)  # где в методичке ответ
    parent_id: str | None = None  # для уточняющего вопроса
    verified: bool | None = None  # результат проверки (для origin="general")

    @property
    def label(self) -> str:
        if self.origin == "general":
            return "не из методички"
        page = f", с. {self.source.page}" if self.source and self.source.page else ""
        work = self.source.work if self.source else "методичка"
        return f"{work}{page}"


class AnswerRecord(BaseModel):
    question_id: str
    question: str
    topic: str
    answer: str
    verdict: Verdict
    explanation: str
    page: int | None = None
    kind: QuestionKind = "control"
