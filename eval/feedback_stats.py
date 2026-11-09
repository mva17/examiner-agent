"""Продуктовая метрика из кнопок «полезно / нет» в чате (цель команды: ≥ 80% полезных).

    python -m eval.feedback_stats
"""

from __future__ import annotations

import json
import sys
from collections import Counter

from examiner.config import get_settings


def stats(path) -> dict:
    if not path.exists():
        return {"total": 0, "useful": 0, "useful_share": None, "users": 0}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    useful = sum(1 for r in rows if r.get("useful"))
    return {
        "total": len(rows),
        "useful": useful,
        "useful_share": round(useful / len(rows), 4) if rows else None,
        "users": len({r.get("user_id") for r in rows}),
        "by_user": dict(Counter(r.get("user_id") for r in rows)),
    }


if __name__ == "__main__":
    result = stats(get_settings().feedback_log)
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    print()
