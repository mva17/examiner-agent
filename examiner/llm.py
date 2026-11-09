"""Подключение к модели и надёжный вызов со структурированным ответом.

Гардрейлы этого модуля:
  * CallBudget — лимит вызовов модели на одну сессию (защита от зацикливания и трат);
  * ask_json — ответ модели проверяется Pydantic-схемой; при ошибке модели
    показывают ошибку и просят исправить (не больше JSON_RETRIES раз);
  * strip_think — срезает рассуждения <think>…</think>, которые пишет Qwen3;
  * with_fallbacks — запасная модель при ошибке сети/лимита основной;
  * любые ошибки провайдера превращаются в LLMError с понятным текстом.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, TypeVar

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from .config import Settings, get_settings

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Модель недоступна или не смогла выдать корректный ответ."""


class BudgetExceeded(LLMError):
    """Исчерпан лимит вызовов модели на сессию."""


@dataclass
class CallBudget:
    limit: int
    used: int = 0

    def spend(self) -> None:
        if self.used >= self.limit:
            raise BudgetExceeded(f"превышен лимит вызовов модели ({self.limit}) на сессию")
        self.used += 1


def build_llm(settings: Settings | None = None):
    """Возвращает Runnable с методом invoke(list[BaseMessage]) -> AIMessage."""
    settings = settings or get_settings()
    if settings.is_fake:
        from .fake_llm import FakeExamLLM

        return FakeExamLLM()

    if not settings.has_key:
        raise LLMError("не задан LLM_API_KEY в .env (или включите LLM_PROVIDER=fake для демо)")

    import openai
    from langchain_openai import ChatOpenAI

    def make(base_url: str, api_key: str, model: str) -> ChatOpenAI:
        return ChatOpenAI(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=0,  # оценка и разбор должны быть воспроизводимыми
            timeout=settings.timeout_s,
            max_retries=1,
        )

    primary = make(settings.base_url, settings.api_key, settings.model)
    if not settings.has_fallback:
        return primary
    fallback = make(
        settings.fallback_base_url,
        settings.fallback_api_key or settings.api_key,
        settings.fallback_model,
    )
    return primary.with_fallbacks(
        [fallback],
        exceptions_to_handle=(
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.RateLimitError,
            openai.InternalServerError,
        ),
    )


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _content(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, list):  # мультимодальный ответ: склеиваем текстовые части
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def strip_think(text: str) -> str:
    text = _THINK.sub("", text)
    # Незакрытый <think> (ответ обрезан по длине) — берём всё после него нельзя, отрезаем.
    if "<think>" in text.lower():
        text = text[: text.lower().index("<think>")]
    return text.strip()


def extract_json(text: str) -> Any:
    """Достаёт первый JSON-объект из ответа модели (с ```json``` или без)."""
    text = strip_think(text)
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("в ответе нет JSON-объекта")
    depth, in_str, escape = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("JSON-объект не закрыт")


def _invoke(llm, messages: list[BaseMessage], budget: CallBudget) -> str:
    budget.spend()
    try:
        return _content(llm.invoke(messages))
    except BudgetExceeded:
        raise
    except Exception as exc:  # сеть, таймаут, 401, 429, 5xx ...
        raise LLMError(f"модель недоступна: {type(exc).__name__}") from exc


def ask_json(
    llm,
    system: str,
    user: str,
    schema: type[T],
    budget: CallBudget,
    *,
    retries: int | None = None,
    trace: list[dict] | None = None,
    step: str = "",
) -> T:
    """Вызов модели с ответом по Pydantic-схеме и повтором при невалидном JSON."""
    retries = get_settings().json_retries if retries is None else retries
    messages: list[BaseMessage] = [SystemMessage(content=system), HumanMessage(content=user)]
    started = time.perf_counter()
    last_error = ""
    for attempt in range(1, retries + 2):
        text = _invoke(llm, messages, budget)
        try:
            result = schema.model_validate(extract_json(text))
        except (ValueError, ValidationError) as exc:
            last_error = str(exc).splitlines()[0][:300]
            messages += [
                AIMessage(content=text[:2000]),
                HumanMessage(
                    content=f"Ответ не прошёл проверку схемы: {last_error}. "
                    "Верни исправленный JSON строго по формату, без пояснений."
                ),
            ]
            continue
        if trace is not None:
            trace.append(_event(step or schema.__name__, True, started, attempt))
        return result
    if trace is not None:
        trace.append(_event(step or schema.__name__, False, started, retries + 1, last_error))
    raise LLMError(f"модель не вернула корректный ответ ({schema.__name__}): {last_error}")


def ask_text(llm, system: str, user: str, budget: CallBudget) -> str:
    """Обычный текстовый вызов без схемы (нужен только baseline в бенчмарке)."""
    return strip_think(_invoke(llm, [SystemMessage(content=system), HumanMessage(content=user)], budget))


def _event(step: str, ok: bool, started: float, attempts: int, error: str = "") -> dict:
    return {
        "step": step,
        "ok": ok,
        "attempts": attempts,
        "elapsed_s": round(time.perf_counter() - started, 3),
        "error": error,
    }
