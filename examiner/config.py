"""Настройки Экзаменатора из переменных окружения и локального .env.

Один источник правды для всех модулей. По умолчанию LLM_PROVIDER=fake:
система поднимается без ключа и без интернета (демо-модель, см. fake_llm.py).
Ключ API хранится только в .env и никогда не печатается (см. __repr__).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


def _find_root() -> Path:
    """Корень проекта: ближайшая папка вверх с requirements.txt (работает и внутри общего репозитория)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "requirements.txt").exists():
            return parent
    return here.parents[1]


ROOT = _find_root()
load_dotenv(ROOT / ".env", override=False)


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _path(name: str, default: str) -> Path:
    value = _env(name, default)
    if value == ":memory:":  # SQLite в памяти (для тестов и демо)
        return Path(value)
    raw = Path(value)
    return raw if raw.is_absolute() else ROOT / raw


@dataclass(frozen=True)
class Settings:
    provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "fake").lower())
    base_url: str = field(
        default_factory=lambda: _env("LLM_BASE_URL", "https://foundation-models.api.cloud.ru/v1")
    )
    api_key: str = field(default_factory=lambda: _env("LLM_API_KEY"), repr=False)
    model: str = field(default_factory=lambda: _env("LLM_MODEL", "qwen/qwen3-vl-30b-a3b-instruct"))

    fallback_base_url: str = field(default_factory=lambda: _env("FALLBACK_BASE_URL"))
    fallback_api_key: str = field(default_factory=lambda: _env("FALLBACK_API_KEY"), repr=False)
    fallback_model: str = field(default_factory=lambda: _env("FALLBACK_MODEL"))

    timeout_s: int = field(default_factory=lambda: _env_int("LLM_TIMEOUT_S", 30))
    max_questions: int = field(default_factory=lambda: _env_int("MAX_QUESTIONS", 5))
    max_llm_calls: int = field(default_factory=lambda: _env_int("MAX_LLM_CALLS", 40))
    max_search_attempts: int = field(default_factory=lambda: _env_int("MAX_SEARCH_ATTEMPTS", 2))
    max_answer_chars: int = field(default_factory=lambda: _env_int("MAX_ANSWER_CHARS", 2000))
    json_retries: int = field(default_factory=lambda: _env_int("JSON_RETRIES", 2))

    search_backend: str = field(default_factory=lambda: _env("SEARCH_BACKEND", "bm25").lower())
    manuals_dir: Path = field(default_factory=lambda: _path("MANUALS_DIR", "data/manuals"))
    chroma_dir: Path = field(default_factory=lambda: _path("CHROMA_DIR", "data/index"))
    chroma_collection: str = field(default_factory=lambda: _env("CHROMA_COLLECTION", "manuals"))

    memory_db: Path = field(default_factory=lambda: _path("MEMORY_DB", "var/memory.sqlite3"))
    feedback_log: Path = field(default_factory=lambda: _path("FEEDBACK_LOG", "var/feedback.jsonl"))

    @property
    def is_fake(self) -> bool:
        return self.provider == "fake"

    @property
    def has_key(self) -> bool:
        return bool(self.api_key)

    @property
    def has_fallback(self) -> bool:
        return bool(self.fallback_base_url and self.fallback_model)


def get_settings() -> Settings:
    """Читает окружение заново: удобно в тестах (monkeypatch.setenv)."""
    return Settings()
