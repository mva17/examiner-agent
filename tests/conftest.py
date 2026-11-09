import pytest


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch, tmp_path):
    """Тесты всегда на демо-модели, с временными файлами памяти и логов."""
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("SEARCH_BACKEND", "bm25")
    monkeypatch.setenv("MEMORY_DB", str(tmp_path / "memory.sqlite3"))
    monkeypatch.setenv("FEEDBACK_LOG", str(tmp_path / "feedback.jsonl"))
    monkeypatch.setenv("MAX_QUESTIONS", "5")
    monkeypatch.setenv("MAX_LLM_CALLS", "40")
