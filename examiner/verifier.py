"""Точка подключения агента-Проверяющего (Юнусов Ибрагим).

Вопросы «не из методички» сгенерированы моделью и могут быть ошибочными,
поэтому до показа студенту каждый вопрос с эталоном проходит проверку.
Пока в общем графе нет Проверяющего, работает LLMQuestionVerifier —
самопроверка той же моделью отдельным вызовом с отдельным промптом.

Чтобы подключить Проверяющего Ибрагима, достаточно передать в
ExaminerDeps(verifier=...) любой объект с методом verify(question, run).
"""

from __future__ import annotations

from typing import Protocol

from . import prompts
from .llm import ask_json
from .schemas import Question, VerifierVerdict


class QuestionVerifier(Protocol):
    def verify(self, question: Question, run: "RunContext") -> VerifierVerdict: ...


class RunContext(Protocol):
    llm: object
    budget: object
    trace: list[dict]


class LLMQuestionVerifier:
    def verify(self, question: Question, run: RunContext) -> VerifierVerdict:
        system, user = prompts.verify_question(question)
        return ask_json(run.llm, system, user, VerifierVerdict, run.budget, trace=run.trace, step="verify_question")


class AcceptAllVerifier:
    """Для тестов и отладки: пропускает всё (в продакшене не использовать)."""

    def verify(self, question: Question, run: RunContext) -> VerifierVerdict:
        return VerifierVerdict(ok=True, reason="проверка отключена")
