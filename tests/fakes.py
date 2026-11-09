"""Фейковые модели для тестов: ключ и интернет не нужны."""

from __future__ import annotations

import json
import re
from collections import defaultdict, deque

from langchain_core.messages import AIMessage

from examiner.agent import Examiner, ExaminerDeps
from examiner.config import ROOT, get_settings
from examiner.fake_llm import FakeExamLLM
from examiner.knowledge import BM25ManualSearch
from examiner.memory import WeakTopicMemory

_TAG = re.compile(r"\[\[TASK:(\w+)\]\]")


class ScriptedLLM(FakeExamLLM):
    """Для задачи с заготовленными ответами отдаёт их по очереди, иначе — правила FakeExamLLM."""

    def __init__(self, script: dict[str, list] | None = None, fail_on: set[str] | None = None):
        super().__init__()
        self.script = defaultdict(deque, {k: deque(v) for k, v in (script or {}).items()})
        self.fail_on = fail_on or set()

    def invoke(self, messages, *args, **kwargs):
        task = (_TAG.search(messages[0].content) or [None, "unknown"])[1]
        if task in self.fail_on:
            self.calls.append(task)
            raise ConnectionError("сеть недоступна")
        if self.script.get(task):
            self.calls.append(task)
            item = self.script[task].popleft()
            return AIMessage(content=item if isinstance(item, str) else json.dumps(item, ensure_ascii=False))
        return super().invoke(messages, *args, **kwargs)


def make_examiner(llm=None, verifier=None, memory=None, **env) -> Examiner:
    settings = get_settings()
    if env:
        from dataclasses import replace

        settings = replace(settings, **env)
    deps = ExaminerDeps(
        llm=llm or FakeExamLLM(),
        search=BM25ManualSearch(ROOT / "data" / "manuals"),
        memory=memory or WeakTopicMemory(":memory:"),
        settings=settings,
    )
    if verifier is not None:
        deps.verifier = verifier
    return Examiner(deps)
