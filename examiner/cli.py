"""Экзаменатор в терминале, без сервера и интерфейса.

    python -m examiner.cli                 # спросит тему
    python -m examiner.cli "маятник"       # тема сразу
"""

from __future__ import annotations

import sys
import uuid

from .agent import Examiner, ExaminerDeps, ExamTurn
from .config import get_settings

COLORS = {"верно": "\033[32m", "частично": "\033[33m", "неверно": "\033[31m"}
RESET = "\033[0m"


def show(turn: ExamTurn) -> None:
    g = turn.grade
    if g and turn.status == "question" or (g and turn.status == "finished"):
        color = COLORS.get(g["verdict"], "")
        page = f" (с. {g['page']})" if g.get("page") else ""
        print(f"\n{color}{g['verdict'].upper()}{RESET}{page}: {g['explanation']}")
    if turn.status == "question":
        q = turn.question
        if q["number"] == 1 and q["kind"] != "followup" and turn.notice:
            print(f"\n[{turn.notice}]")
        prefix = "Уточнение" if q["kind"] == "followup" else f"Вопрос {q['number']}/{q['max_questions']}"
        print(f"\n{prefix} [{q['label']}]: {q['question']}")
    else:
        print(f"\n{turn.message}")


def main() -> None:
    settings = get_settings()
    mode = "демо-модель без ключа" if settings.is_fake else settings.model
    print(f"Экзаменатор (модель: {mode}). «стоп» — закончить.")
    examiner = Examiner(ExaminerDeps.from_settings(settings))
    user_id, chat_id = "cli", uuid.uuid4().hex[:8]
    request = " ".join(sys.argv[1:]) or input("По какой теме проверить? > ")
    turn = examiner.start(user_id, chat_id, request)
    show(turn)
    while turn.status == "question":
        try:
            answer = input("> ")
        except (EOFError, KeyboardInterrupt):
            answer = "стоп"
        turn = examiner.answer(user_id, chat_id, answer)
        show(turn)
    print(f"\nВызовов модели: {turn.llm_calls}")


if __name__ == "__main__":
    main()
