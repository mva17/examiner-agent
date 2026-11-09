"""Проверка скриптов метрик (на демо-модели: сами числа здесь не важны)."""

import json
from pathlib import Path

import pytest

from eval import feedback_stats, run_grading_benchmark as bench
from eval.download_rummlu import parse_mera, parse_mmlu_ru

SAMPLE = Path(__file__).parent / "data" / "rummlu_sample.jsonl"


def test_cases_are_paired_and_deterministic():
    items = bench.load(SAMPLE, 10, seed=1)
    a, b = bench.make_cases(items, 1), bench.make_cases(items, 1)
    assert a == b and len(a) == 6
    assert [c["expected"] for c in a[:2]] == ["верно", "неверно"]
    assert a[0]["answer"] == a[0]["reference"] != a[1]["answer"]


@pytest.mark.parametrize("text, verdict", [("Верно.", "верно"), ("Ответ неверно, потому что", "неверно"),
                                           ("Частично верно", "частично"), ("Не могу сказать", None)])
def test_parse_baseline(text, verdict):
    assert bench.parse_baseline(text) == verdict


def test_metrics_math():
    rows = [
        {"expected": "верно", "verdict": "верно", "latency_s": 1, "llm_calls": 1},
        {"expected": "неверно", "verdict": "верно", "latency_s": 2, "llm_calls": 1},
        {"expected": "верно", "verdict": None, "latency_s": 3, "llm_calls": 3},
        {"expected": "неверно", "verdict": "частично", "latency_s": 4, "llm_calls": 1},
    ]
    m = bench.metrics(rows)
    assert m["accuracy"] == 0.5 and m["correct_accept"] == 0.5 and m["false_accept"] == 0.5
    assert m["wrong_partial"] == 0.5 and m["invalid"] == 0.25 and m["llm_calls_per_case"] == 1.5


def test_benchmark_refuses_fake_model_without_flag(tmp_path):
    with pytest.raises(SystemExit):
        bench.main(["--data", str(SAMPLE), "--out", str(tmp_path)])


@pytest.mark.parametrize("baseline", [False, True])
def test_benchmark_runs_end_to_end_and_marks_fake(tmp_path, baseline):
    args = ["--data", str(SAMPLE), "--out", str(tmp_path), "--limit", "3", "--allow-fake"]
    result = bench.main(args + (["--baseline"] if baseline else []))
    assert result["fake_model"] is True and result["metrics"]["cases"] == 6
    saved = list(tmp_path.glob("*_FAKE.json"))
    assert len(saved) == 1 and json.loads(saved[0].read_text(encoding="utf-8"))["mode"] == result["mode"]


def test_dataset_parsers():
    mera = {"inputs": {"text": "Q", "option_a": "a", "option_b": "b", "option_c": "c", "option_d": "d", "subject": "x"},
            "outputs": "C", "meta": {"domain": "college_physics", "id": 7}}
    assert parse_mera(mera)["answer_index"] == 2
    assert parse_mera({**mera, "outputs": ""}) is None  # скрытый ответ теста MERA
    assert parse_mera({**mera, "meta": {"domain": "world_religions", "id": 1}}) is None
    row = {"question_ru": "Q", "choices_ru": ["a", "b", "c", "d"], "answer": 3}
    assert parse_mmlu_ru(row, "college_physics")["answer_index"] == 3


def test_feedback_stats(tmp_path):
    log = tmp_path / "f.jsonl"
    log.write_text('{"user_id": "a", "useful": true}\n{"user_id": "b", "useful": false}\n', encoding="utf-8")
    assert feedback_stats.stats(log)["useful_share"] == 0.5
    assert feedback_stats.stats(tmp_path / "none.jsonl")["total"] == 0
