from examiner.memory import WeakTopicMemory
from examiner.schemas import Question


def make(id_="q1", kind="control", topic="Маятник"):
    return Question(id=id_, text="Почему период не зависит от массы?", topic=topic, origin="manual", kind=kind)


def test_memory_persists_between_instances(tmp_path):
    path = tmp_path / "m.sqlite3"
    mem = WeakTopicMemory(path)
    mem.record("u1", make(), 0.0)
    mem.record("u1", make("q2"), 0.5)
    mem2 = WeakTopicMemory(path)  # как будто сервер перезапустили
    weak = mem2.weak_topics("u1")
    assert weak == [{"topic": "маятник", "attempts": 2, "avg_score": 0.25}]
    assert {q.id for q in mem2.mistakes("u1")} == {"q1", "q2"}
    assert mem2.weak_topics("other_user") == []


def test_correct_answer_removes_mistake_and_followups_not_stored():
    mem = WeakTopicMemory(":memory:")
    mem.record("u", make(), 0.0)
    mem.record("u", make(), 1.0)
    mem.record("u", make("f", kind="followup"), 0.0)
    assert mem.mistakes("u") == []


def test_memory_db_in_memory_setting_creates_no_file(monkeypatch):
    from examiner.config import ROOT, get_settings

    monkeypatch.setenv("MEMORY_DB", ":memory:")
    settings = get_settings()
    WeakTopicMemory(settings.memory_db).record("u", make(), 0.0)
    assert not (ROOT / ":memory:").exists()
