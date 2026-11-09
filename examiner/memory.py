"""Долгая память Экзаменатора: слабые темы и вопросы с ошибками между сессиями.

Краткосрочная память (ход текущего диалога) хранится в checkpointer LangGraph
по thread_id = "<user_id>:<chat_id>". Долгая — здесь, по user_id, в SQLite,
поэтому переживает перезапуск сервера и новые чаты.
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import closing
from pathlib import Path

from .schemas import Question

SCHEMA = """
CREATE TABLE IF NOT EXISTS topic_stats (
    user_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    score_sum REAL NOT NULL DEFAULT 0,
    updated REAL NOT NULL,
    PRIMARY KEY (user_id, topic)
);
CREATE TABLE IF NOT EXISTS mistakes (
    user_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    question_json TEXT NOT NULL,
    updated REAL NOT NULL,
    PRIMARY KEY (user_id, question_id)
);
"""


class WeakTopicMemory:
    def __init__(self, path: Path | str):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._shared = sqlite3.connect(self.path, check_same_thread=False) if self.path == ":memory:" else None
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        if self._shared is not None:
            return self._shared
        return sqlite3.connect(self.path)

    def _run(self, sql: str, params: tuple = ()) -> list[tuple]:
        conn = self._conn()
        try:
            with conn:
                return conn.execute(sql, params).fetchall()
        finally:
            if self._shared is None:
                conn.close()

    @staticmethod
    def _key(topic: str) -> str:
        return " ".join(topic.lower().split())

    def record(self, user_id: str, question: Question, score: float) -> None:
        """Сохраняет результат ответа: статистика темы + вопрос в списке ошибок или удаление из него."""
        now = time.time()
        self._run(
            """INSERT INTO topic_stats(user_id, topic, attempts, score_sum, updated) VALUES (?, ?, 1, ?, ?)
               ON CONFLICT(user_id, topic) DO UPDATE SET attempts = attempts + 1,
               score_sum = score_sum + excluded.score_sum, updated = excluded.updated""",
            (user_id, self._key(question.topic), score, now),
        )
        if question.kind == "followup":
            return
        if score < 1.0:
            self._run(
                """INSERT OR REPLACE INTO mistakes(user_id, question_id, topic, question_json, updated)
                   VALUES (?, ?, ?, ?, ?)""",
                (user_id, question.id, self._key(question.topic), question.model_dump_json(), now),
            )
        else:
            self._run("DELETE FROM mistakes WHERE user_id = ? AND question_id = ?", (user_id, question.id))

    def weak_topics(self, user_id: str, threshold: float = 0.7, limit: int = 5) -> list[dict]:
        rows = self._run(
            """SELECT topic, attempts, score_sum / attempts AS avg FROM topic_stats
               WHERE user_id = ? AND score_sum / attempts < ? ORDER BY avg ASC, updated DESC LIMIT ?""",
            (user_id, threshold, limit),
        )
        return [{"topic": t, "attempts": a, "avg_score": round(s, 2)} for t, a, s in rows]

    def mistakes(self, user_id: str, topic: str | None = None, limit: int = 10) -> list[Question]:
        if topic:
            rows = self._run(
                "SELECT question_json FROM mistakes WHERE user_id = ? AND topic = ? ORDER BY updated DESC LIMIT ?",
                (user_id, self._key(topic), limit),
            )
        else:
            rows = self._run(
                "SELECT question_json FROM mistakes WHERE user_id = ? ORDER BY updated DESC LIMIT ?",
                (user_id, limit),
            )
        return [Question.model_validate_json(r[0]) for r in rows]

    def close(self) -> None:
        if self._shared is not None:
            with closing(self._shared):
                pass
