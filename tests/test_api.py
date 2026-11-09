import json

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from tests.fakes import ScriptedLLM, make_examiner

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


@pytest.fixture
def client():
    llm = ScriptedLLM({"grade_answer": [{"verdict": "верно", "explanation": "ok", "page": 1}] * 10})
    return TestClient(create_app(make_examiner(llm)))


def test_health_without_key(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["llm_provider"] == "fake"
    assert "key" not in json.dumps(body).lower().replace("key_configured", "")


def test_chat_flow_start_answer_finish(client):
    ids = {"user_id": "u1", "chat_id": "c1"}
    r = client.post("/chat", json={**ids, "message": "Проверь меня по теме маятник"})
    assert r.status_code == 200
    body = r.json()
    assert body["agent"] == "examiner" and body["status"] == "question"
    assert body["question"]["label"].startswith("ДЕМО-2")
    assert any(e["node"] == "check_relevance" for e in body["events"])
    assert "X-Elapsed-S" in r.headers

    body = client.post("/chat", json={**ids, "message": "T = 2π√(l/g)"}).json()
    assert body["grade"]["verdict"] == "верно"
    body = client.post("/chat", json={**ids, "message": "стоп"}).json()
    assert body["status"] == "finished" and "Итог" in body["message"]


def test_exam_endpoints_and_conflicts(client):
    ids = {"user_id": "u1", "chat_id": "c2"}
    assert client.post("/exam/answer", json={**ids, "answer": "x"}).status_code == 409
    assert client.post("/exam/start", json={**ids, "request": "микрометр"}).status_code == 200
    assert client.post("/exam/start", json={**ids, "request": "микрометр"}).status_code == 409
    assert client.post("/resume", json={**ids, "answer": "0,01 мм"}).json()["grade"]["verdict"] == "верно"


def test_validation_errors(client):
    assert client.post("/chat", json={"user_id": "bad id", "chat_id": "c", "message": "x"}).status_code == 422
    assert client.post("/chat", json={"user_id": "u", "chat_id": "c", "message": ""}).status_code == 422
    assert client.post("/chat", json={"user_id": "u", "chat_id": "c", "message": "x" * 5000}).status_code == 422


def test_refusal_is_200_with_message(client):
    body = client.post("/chat", json={"user_id": "u", "chat_id": "c3", "message": "Посоветуй сериал"}).json()
    assert body["status"] == "refused" and body["message"]


def test_feedback_is_logged(client, tmp_path):
    r = client.post("/feedback", json={"user_id": "u", "chat_id": "c", "useful": True, "message": "вопрос"})
    assert r.json() == {"saved": True}
    lines = (tmp_path / "feedback.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[-1])["useful"] is True


def test_memory_endpoint(client):
    ids = {"user_id": "u9", "chat_id": "c1"}
    client.post("/chat", json={**ids, "message": "микрометр"})
    client.post("/chat", json={**ids, "message": "не знаю"})
    body = client.get("/memory/u9").json()
    assert body["weak_topics"][0]["topic"] == "микрометр"
    assert len(body["mistakes"]) == 1


def test_upload_checks_type_and_size(client):
    assert client.post("/upload", files={"file": ("t.png", PNG, "image/png")}).json()["type"] == "png"
    assert client.post("/upload", files={"file": ("t.png", b"<script>", "image/png")}).status_code == 415
    big = PNG + b"\x00" * (10 * 1024 * 1024)
    assert client.post("/upload", files={"file": ("t.png", big, "image/png")}).status_code == 413


def test_model_not_configured_gives_503(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    c = TestClient(create_app())
    assert c.get("/health").json()["status"] == "degraded"
    r = c.post("/chat", json={"user_id": "u", "chat_id": "c", "message": "маятник"})
    assert r.status_code == 503 and "LLM_API_KEY" in r.json()["detail"]


def test_router_mounts_into_team_app_with_prefix():
    """Так Экзаменатор встраивается в общий FastAPI команды, не трогая чужие маршруты."""
    from fastapi import FastAPI

    from api.main import create_router

    team_app = FastAPI()

    @team_app.get("/health")
    def team_health():
        return {"owner": "calculator"}

    team_app.include_router(create_router(make_examiner()), prefix="/examiner")
    c = TestClient(team_app)
    assert c.get("/health").json() == {"owner": "calculator"}
    body = c.post("/examiner/chat", json={"user_id": "u", "chat_id": "c", "message": "маятник"}).json()
    assert body["status"] == "question"
