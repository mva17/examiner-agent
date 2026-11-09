"""HTTP API Экзаменатора (FastAPI).

Запуск:  uvicorn api.main:app --reload
Документация после запуска: http://127.0.0.1:8000/docs

Ключ модели живёт только на сервере (в .env), в ответы и в интерфейс не попадает.
Ошибки модели превращаются в понятные сообщения, а не в трассировку Python.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from examiner import Examiner, ExaminerDeps, ExamTurn, SessionConflict
from examiner.config import ROOT, get_settings
from examiner.llm import LLMError

log = logging.getLogger("examiner.api")

ID = r"^[A-Za-z0-9_\-]{1,64}$"
MAX_UPLOAD = 10 * 1024 * 1024
IMAGE_MAGIC = {b"\x89PNG\r\n\x1a\n": "png", b"\xff\xd8\xff": "jpg", b"RIFF": "webp"}


class StartRequest(BaseModel):
    user_id: str = Field(pattern=ID, examples=["student1"])
    chat_id: str = Field(pattern=ID, examples=["chat1"])
    request: str = Field(min_length=1, max_length=2000, examples=["Проверь меня по теме микрометр"])


class AnswerRequest(BaseModel):
    user_id: str = Field(pattern=ID)
    chat_id: str = Field(pattern=ID)
    answer: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    user_id: str = Field(pattern=ID)
    chat_id: str = Field(pattern=ID)
    message: str = Field(min_length=1, max_length=4000)


class ChatResponse(ExamTurn):
    agent: str = "examiner"


class FeedbackRequest(BaseModel):
    user_id: str = Field(pattern=ID)
    chat_id: str = Field(pattern=ID)
    useful: bool
    message: str = Field(default="", max_length=4000, description="на что дан отзыв")
    comment: str = Field(default="", max_length=1000)


def create_router(examiner: Examiner | None = None) -> APIRouter:
    """Все маршруты Экзаменатора. В общем приложении команды:
    app.include_router(create_router(), prefix="/examiner")."""
    app = APIRouter(tags=["examiner"])
    settings = get_settings()
    state: dict = {"examiner": examiner, "init_error": ""}

    def get_examiner() -> Examiner:
        if state["examiner"] is None:
            try:
                state["examiner"] = Examiner(ExaminerDeps.from_settings(settings))
            except LLMError as exc:
                state["init_error"] = str(exc)
                raise HTTPException(503, f"Модель не настроена: {exc}") from exc
        return state["examiner"]

    def run(fn, *args) -> ExamTurn:
        try:
            return fn(*args)
        except SessionConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except LLMError as exc:
            raise HTTPException(503, f"Модель недоступна: {exc}") from exc

    @app.get("/health")
    def health() -> dict:
        ok = True
        try:
            get_examiner()
        except HTTPException:
            ok = False
        return {
            "status": "ok" if ok else "degraded",
            "llm_provider": settings.provider,
            "model": "демо-модель (fake)" if settings.is_fake else settings.model,
            "key_configured": settings.has_key,
            "fallback_configured": settings.has_fallback,
            "search_backend": settings.search_backend,
            "error": state["init_error"],
        }

    @app.post("/exam/start", response_model=ExamTurn)
    def exam_start(req: StartRequest) -> ExamTurn:
        return run(get_examiner().start, req.user_id, req.chat_id, req.request)

    @app.post("/exam/answer", response_model=ExamTurn)
    def exam_answer(req: AnswerRequest) -> ExamTurn:
        return run(get_examiner().answer, req.user_id, req.chat_id, req.answer)

    # Синоним из общего плана команды: продолжение прерванного графа.
    app.post("/resume", response_model=ExamTurn)(exam_answer)

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest) -> ChatResponse:
        """Одна точка входа для чата и Диспетчера: ждём ответ — продолжаем, иначе начинаем сессию."""
        ex = get_examiner()
        if ex.is_waiting(req.user_id, req.chat_id):
            turn = run(ex.answer, req.user_id, req.chat_id, req.message)
        else:
            turn = run(ex.start, req.user_id, req.chat_id, req.message)
        return ChatResponse(**turn.model_dump())

    @app.get("/memory/{user_id}")
    def memory(user_id: str) -> dict:
        if not re.match(ID, user_id):
            raise HTTPException(422, "неверный user_id")
        mem = get_examiner().deps.memory
        return {
            "user_id": user_id,
            "weak_topics": mem.weak_topics(user_id),
            "mistakes": [q.text for q in mem.mistakes(user_id)],
        }

    @app.post("/feedback")
    def feedback(req: FeedbackRequest) -> dict:
        """Кнопки «полезно / нет» — источник продуктовой метрики (eval/feedback_stats.py)."""
        path: Path = settings.feedback_log
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"ts": time.time(), **req.model_dump()}
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {"saved": True}

    @app.post("/upload")
    async def upload(file: UploadFile = File(...)) -> dict:
        """Фото таблицы для Расчётчика: проверяем тип и размер, сохраняем. Сам Расчётчик — в общем репозитории."""
        data = await file.read(MAX_UPLOAD + 1)
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, "Файл больше 10 МБ")
        kind = next((ext for magic, ext in IMAGE_MAGIC.items() if data.startswith(magic)), None)
        if kind is None or (kind == "webp" and data[8:12] != b"WEBP"):
            raise HTTPException(415, "Нужна картинка PNG, JPEG или WEBP")
        file_id = uuid.uuid4().hex
        target = ROOT / "var" / "uploads" / f"{file_id}.{kind}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return {"file_id": file_id, "size": len(data), "type": kind, "status": "accepted"}

    return app


def create_app(examiner: Examiner | None = None) -> FastAPI:
    app = FastAPI(
        title="Экзаменатор — API",
        description="Тренажёр допуска к лабораторным по физике (команда 43, НИЯУ МИФИ). Агент на LangGraph.",
        version="0.2.0",
    )
    app.include_router(create_router(examiner))

    @app.middleware("http")
    async def timing(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Elapsed-S"] = f"{time.perf_counter() - started:.3f}"
        return response

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception):
        log.exception("необработанная ошибка")  # подробности — только в логе сервера
        return JSONResponse(status_code=500, content={"detail": "Внутренняя ошибка сервера. Попробуйте ещё раз."})

    return app


app = create_app()
