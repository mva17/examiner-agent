"""Скачивает вопросы по физике из ruMMLU (бенчмарк MERA) с известным правильным ответом.

    python -m eval.download_rummlu                 # MERA, при неудаче — запасной источник
    python -m eval.download_rummlu --source mmlu_ru

Данные берутся через HTTP API Hugging Face datasets-server (без библиотеки datasets).
Источники:
  * mera    — MERA-evaluation/MERA, конфиг rummlu. В MERA ответы тестовой части
              скрыты, поэтому берём только строки, где ответ есть (обычно сплит train/dev).
  * mmlu_ru — NLPCoreTeam/mmlu_ru (перевод MMLU), конфиги *_physics, ответы открыты.
Результат: eval/data/rummlu_physics.jsonl — {id, source, subject, question, options[4], answer_index}.
Скрипт печатает, сколько строк нашёл в каждом сплите и сколько из них с ответом.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import httpx

API = "https://datasets-server.huggingface.co"
OUT = Path(__file__).resolve().parent / "data" / "rummlu_physics.jsonl"
PHYSICS = ("high_school_physics", "college_physics", "conceptual_physics")
LETTERS = "ABCD"


def _get(client: httpx.Client, path: str, **params) -> dict:
    r = client.get(f"{API}{path}", params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def _rows(client: httpx.Client, dataset: str, config: str, split: str):
    offset = 0
    while True:
        page = _get(client, "/rows", dataset=dataset, config=config, split=split, offset=offset, length=100)
        rows = page.get("rows", [])
        for item in rows:
            yield item["row"]
        offset += len(rows)
        if not rows or offset >= page.get("num_rows_total", 0):
            break


def _answer_index(value) -> int | None:
    if isinstance(value, int) and 0 <= value < 4:
        return value
    if isinstance(value, str):
        v = value.strip().upper()
        if v in LETTERS and v:
            return LETTERS.index(v)
        if v.isdigit() and int(v) < 4:
            return int(v)
    return None


def parse_mera(row: dict) -> dict | None:
    inputs, meta = row.get("inputs") or {}, row.get("meta") or {}
    domain = str(meta.get("domain") or inputs.get("subject") or "")
    if not any(p in domain for p in PHYSICS) and "физик" not in domain.lower():
        return None
    options = [inputs.get(f"option_{c}") for c in "abcd"]
    answer = _answer_index(row.get("outputs"))
    if answer is None or not all(options) or not inputs.get("text"):
        return None
    return {"id": f"mera-{meta.get('id')}", "source": "MERA/rummlu", "subject": domain,
            "question": inputs["text"], "options": options, "answer_index": answer}


def parse_mmlu_ru(row: dict, config: str) -> dict | None:
    options = row.get("choices_ru") or row.get("choices")
    answer = _answer_index(row.get("answer"))
    question = row.get("question_ru") or row.get("question")
    if answer is None or not options or len(options) != 4 or not question:
        return None
    return {"id": f"mmlu_ru-{config}-{hashlib.sha1(question.encode()).hexdigest()[:10]}", "source": "NLPCoreTeam/mmlu_ru",
            "subject": config, "question": question, "options": list(options), "answer_index": answer}


def from_mera(client: httpx.Client) -> list[dict]:
    dataset = "MERA-evaluation/MERA"
    splits = [s["split"] for s in _get(client, "/splits", dataset=dataset)["splits"] if s["config"] == "rummlu"]
    items = []
    for split in splits:
        total = found = 0
        for row in _rows(client, dataset, "rummlu", split):
            total += 1
            item = parse_mera(row)
            if item:
                found += 1
                items.append(item | {"split": split})
        print(f"  MERA rummlu/{split}: строк {total}, физика с ответом {found}")
    return items


def from_mmlu_ru(client: httpx.Client) -> list[dict]:
    dataset = "NLPCoreTeam/mmlu_ru"
    available = _get(client, "/splits", dataset=dataset)["splits"]
    items = []
    for config in PHYSICS:
        for split in sorted({s["split"] for s in available if s["config"] == config}):
            rows = [parse_mmlu_ru(r, config) for r in _rows(client, dataset, config, split)]
            ok = [r | {"split": split} for r in rows if r]
            print(f"  mmlu_ru {config}/{split}: строк {len(rows)}, с ответом {len(ok)}")
            items.extend(ok)
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["auto", "mera", "mmlu_ru"], default="auto")
    parser.add_argument("--min", type=int, default=100, help="сколько вопросов нужно минимум (для auto)")
    args = parser.parse_args()

    items: list[dict] = []
    with httpx.Client() as client:
        if args.source in ("auto", "mera"):
            try:
                print("MERA-evaluation/MERA (rummlu):")
                items = from_mera(client)
            except httpx.HTTPError as exc:
                print(f"  не удалось: {exc}")
        if args.source == "mmlu_ru" or (args.source == "auto" and len(items) < args.min):
            print(f"Вопросов с ответом из MERA: {len(items)} (< {args.min}), берём NLPCoreTeam/mmlu_ru:")
            items += from_mmlu_ru(client)
    if not items:
        sys.exit("Ничего не скачалось: проверьте интернет/VPN и имя датасета.")

    seen, unique = set(), []
    for it in items:
        key = it["question"].strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(it)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for it in unique:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"Сохранено {len(unique)} вопросов → {OUT}")


if __name__ == "__main__":
    main()
