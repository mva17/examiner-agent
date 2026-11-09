"""Бенчмарк проверки ответов Экзаменатора на вопросах по физике из ruMMLU.

Идея: у каждого вопроса известен правильный вариант. Подаём агенту вопрос и
вариант как «ответ студента» — один раз правильный, один раз неправильный.
Правильный должен получить «верно», неправильный — нет.

Режимы (одни и те же данные, одна и та же модель, seed фиксирован):
  * agent     — как в узле grade: промпт из 6 частей + эталон + схема Grade (Pydantic) + повтор при ошибке JSON;
  * baseline  — «LLM без эталона»: та же модель, простой вопрос «верно/частично/неверно?»,
                без эталонного ответа, без схемы и без примеров.

    python -m eval.download_rummlu
    python -m eval.run_grading_benchmark --limit 100
    python -m eval.run_grading_benchmark --limit 100 --baseline

Результат: eval/results/grading_<mode>_<дата>.json (метрики + каждый ответ модели).
Метрики:
  accuracy        — доля случаев, где «верно» поставлено ровно тогда, когда ответ правильный;
  correct_accept  — правильный ответ получил «верно»;
  false_accept    — неправильный ответ получил «верно» (главная ошибка экзаменатора);
  wrong_partial   — неправильный ответ получил «частично»;
  invalid         — модель не дала разбираемой оценки (считается ошибкой в accuracy);
  latency p50/p95, вызовов модели на случай.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

from examiner import prompts
from examiner.config import get_settings
from examiner.llm import CallBudget, LLMError, ask_json, ask_text, build_llm
from examiner.schemas import Grade, Question

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "data" / "rummlu_physics.jsonl"
RESULTS = HERE / "results"
LETTERS = "ABCD"


def load(path: Path, limit: int, seed: int) -> list[dict]:
    if not path.exists():
        sys.exit(f"Нет файла {path}. Сначала: python -m eval.download_rummlu")
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rng = random.Random(seed)
    rng.shuffle(items)
    return items[:limit]


def make_cases(items: list[dict], seed: int) -> list[dict]:
    """Для каждого вопроса — правильный и один (детерминированно) неправильный вариант."""
    rng = random.Random(seed)
    cases = []
    for it in items:
        opts = it["options"]
        stem = it["question"] + "\nВарианты: " + "; ".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(opts))
        right = it["answer_index"]
        wrong = rng.choice([i for i in range(len(opts)) if i != right])
        for idx, expected in ((right, "верно"), (wrong, "неверно")):
            cases.append(
                {
                    "id": it["id"],
                    "subject": it.get("subject", ""),
                    "question": stem,
                    "reference": f"{LETTERS[right]}) {opts[right]}",
                    "answer": f"{LETTERS[idx]}) {opts[idx]}",
                    "expected": expected,
                }
            )
    return cases


def parse_baseline(text: str) -> str | None:
    low = text.lower()
    found = [(low.find(v), v) for v in ("неверно", "частично", "верно") if v in low]
    if not found:
        return None
    first = min(pos for pos, _ in found)
    # «неверно» содержит «верно»: при совпадении позиции побеждает более длинное слово
    return next(v for pos, v in found if pos == first)


def grade_case(llm, case: dict, baseline: bool) -> tuple[str | None, int, str]:
    budget = CallBudget(5)
    try:
        if baseline:
            system, user = prompts.baseline_grade(case["question"], case["answer"])
            return parse_baseline(ask_text(llm, system, user, budget)), budget.used, ""
        q = Question(id=case["id"], text=case["question"], reference_answer=case["reference"],
                     topic=case["subject"], origin="general", kind="generated")
        system, user = prompts.grade_answer(q, case["answer"], [])
        return ask_json(llm, system, user, Grade, budget).verdict, budget.used, ""
    except LLMError as exc:
        return None, budget.used, str(exc)


def metrics(rows: list[dict]) -> dict:
    right = [r for r in rows if r["expected"] == "верно"]
    wrong = [r for r in rows if r["expected"] == "неверно"]
    lat = sorted(r["latency_s"] for r in rows)

    def share(part, whole):
        return round(len(part) / len(whole), 4) if whole else None

    return {
        "cases": len(rows),
        "questions": len(right),
        "accuracy": share([r for r in rows if (r["verdict"] == "верно") == (r["expected"] == "верно") and r["verdict"]], rows),
        "correct_accept": share([r for r in right if r["verdict"] == "верно"], right),
        "false_accept": share([r for r in wrong if r["verdict"] == "верно"], wrong),
        "wrong_partial": share([r for r in wrong if r["verdict"] == "частично"], wrong),
        "invalid": share([r for r in rows if r["verdict"] is None], rows),
        "latency_p50_s": round(statistics.median(lat), 3) if lat else None,
        "latency_p95_s": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 3) if lat else None,
        "llm_calls_per_case": round(sum(r["llm_calls"] for r in rows) / len(rows), 3) if rows else None,
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Бенчмарк оценки ответов Экзаменатора (ruMMLU, физика)")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--limit", type=int, default=100, help="сколько вопросов (случаев будет вдвое больше)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline", action="store_true", help="режим «LLM без эталона»")
    parser.add_argument("--out", type=Path, default=RESULTS)
    parser.add_argument("--allow-fake", action="store_true", help="только для проверки скрипта: цифры не настоящие")
    args = parser.parse_args(argv)

    settings = get_settings()
    if settings.is_fake and not args.allow_fake:
        sys.exit("LLM_PROVIDER=fake: на демо-модели метрики бессмысленны. Впишите ключ в .env (LLM_PROVIDER=openai).")
    llm = build_llm(settings)
    mode = "baseline" if args.baseline else "agent"
    cases = make_cases(load(args.data, args.limit, args.seed), args.seed)
    print(f"Режим {mode}, модель {settings.model if not settings.is_fake else 'fake'}, случаев {len(cases)}")

    rows = []
    for i, case in enumerate(cases, 1):
        started = time.perf_counter()
        verdict, calls, error = grade_case(llm, case, args.baseline)
        rows.append({**case, "verdict": verdict, "llm_calls": calls, "error": error,
                     "latency_s": round(time.perf_counter() - started, 3)})
        if i % 20 == 0:
            print(f"  {i}/{len(cases)}")

    result = {
        "benchmark": "examiner_grading_rummlu_physics",
        "mode": mode,
        "fake_model": settings.is_fake,
        "model": "fake" if settings.is_fake else settings.model,
        "base_url": "" if settings.is_fake else settings.base_url,
        "data": args.data.name,
        "limit": args.limit,
        "seed": args.seed,
        "date": datetime.now().isoformat(timespec="seconds"),
        "metrics": metrics(rows),
        "rows": rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    suffix = "_FAKE" if settings.is_fake else ""
    path = args.out / f"grading_{mode}_{datetime.now():%Y%m%d_%H%M}{suffix}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    print(f"Сохранено: {path}")
    return result


if __name__ == "__main__":
    main()
