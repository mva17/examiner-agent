import pytest

from examiner.llm import BudgetExceeded, CallBudget, LLMError, ask_json, extract_json, strip_think
from examiner.schemas import Grade, TopicExtraction
from tests.fakes import ScriptedLLM

SYSTEM = "[[TASK:extract_topic]]\nтест"


def test_strip_think_removes_qwen_reasoning():
    assert strip_think("<think>долго думаю {\"a\": 1}</think>\n{\"b\": 2}") == '{"b": 2}'
    assert strip_think('{"b": 2}<think>обрезано') == '{"b": 2}'


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"a": 1}', {"a": 1}),
        ('Вот ответ:\n```json\n{"a": {"b": "}"}}\n```', {"a": {"b": "}"}}),
        ('<think>x</think> текст {"a": "кавычка \\" внутри"} хвост', {"a": 'кавычка " внутри'}),
    ],
)
def test_extract_json(text, expected):
    assert extract_json(text) == expected


def test_extract_json_errors():
    with pytest.raises(ValueError):
        extract_json("нет json")
    with pytest.raises(ValueError):
        extract_json('{"a": 1')


def test_ask_json_retries_on_invalid_then_succeeds():
    llm = ScriptedLLM({"extract_topic": ["не json", '{"topic": "", "is_physics": true, "search_query": "x"}',
                                         '{"topic": "маятник", "is_physics": true, "search_query": "маятник"}']})
    budget = CallBudget(10)
    trace: list = []
    result = ask_json(llm, SYSTEM, "x", TopicExtraction, budget, retries=2, trace=trace)
    assert result.topic == "маятник"
    assert budget.used == 3
    assert trace[-1]["ok"] and trace[-1]["attempts"] == 3


def test_ask_json_gives_up_after_retries():
    llm = ScriptedLLM({"extract_topic": ["плохо"] * 5})
    with pytest.raises(LLMError, match="TopicExtraction"):
        ask_json(llm, SYSTEM, "x", TopicExtraction, CallBudget(10), retries=1)


def test_budget_limit():
    budget = CallBudget(1)
    llm = ScriptedLLM({"extract_topic": ['{"topic": "a", "is_physics": true, "search_query": "a"}'] * 2})
    ask_json(llm, SYSTEM, "x", TopicExtraction, budget)
    with pytest.raises(BudgetExceeded):
        ask_json(llm, SYSTEM, "x", TopicExtraction, budget)


def test_provider_error_becomes_llm_error():
    llm = ScriptedLLM(fail_on={"extract_topic"})
    with pytest.raises(LLMError, match="ConnectionError"):
        ask_json(llm, SYSTEM, "x", TopicExtraction, CallBudget(5))


@pytest.mark.parametrize("raw, verdict", [("Верно.", "верно"), ("частично верно", "частично"), ("НЕВЕРНО", "неверно")])
def test_grade_verdict_normalization(raw, verdict):
    assert Grade(verdict=raw, explanation="x").verdict == verdict


def test_build_llm_requires_key(monkeypatch):
    from examiner.config import get_settings
    from examiner.llm import build_llm

    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(LLMError, match="LLM_API_KEY"):
        build_llm(get_settings())


def test_settings_repr_hides_key(monkeypatch):
    from examiner.config import get_settings

    monkeypatch.setenv("LLM_API_KEY", "sk-secret-123")
    assert "sk-secret-123" not in repr(get_settings())
