"""Общие функции работы с русским текстом: токены, грубый стемминг, инъекции."""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)

STOPWORDS = frozenset(
    """
    а без более бы был была были было быть в вам вас весь во вот все всего всех вы
    где да даже для до его ее её если есть еще ещё же за и из или им их к как ко когда
    кто ли либо мне может мы на над надо не него нее неё нет ни них но ну о об однако
    он она они оно от по под при про с со так также такой там те тем то того тоже той
    только том ту ты у уже чем что чтобы эта эти это этого этой этот я
    меня мой мою моя теме тема тему темы проверь проверьте проверка спроси спросите
    подготовь подготовьте допуск допуску допуска хочу пожалуйста давай вопросы вопрос
    задай задайте лабораторной лабораторная лаба лабе работе работа работы
    """.split()
)


def tokens(text: str) -> list[str]:
    return [t.lower().replace("ё", "е") for t in _WORD.findall(text)]


def stem(word: str) -> str:
    """Очень грубый стемминг для русского: первые 5 букв (микрометр/микрометра → микро)."""
    return word[:5] if len(word) > 5 else word


def content_stems(text: str) -> list[str]:
    return [stem(t) for t in tokens(text) if t not in STOPWORDS and len(t) > 1]


_INJECTION = re.compile(
    r"(игнорир\w*|забудь|забыть|ignore|disregard)\s+(\w+\s+){0,3}(инструкц\w*|правил\w*|instruction\w*|prompt\w*)"
    r"|постав\w*\s+(мне\s+)?(верно|зач[её]т|5|отлично)"
    r"|(ты\s+теперь|you\s+are\s+now|system\s*prompt|системн\w+\s+промпт)",
    re.IGNORECASE,
)


def looks_like_injection(text: str) -> bool:
    """Дешёвый фильтр до вызова модели. Основная защита — теги и промпт, это доп. слой."""
    return bool(_INJECTION.search(text))


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    return [p.strip() for p in parts if len(p.strip()) > 3]


def best_sentence(question: str, text: str) -> str:
    """Предложение фрагмента, больше всего совпадающее по словам с вопросом (подсказка-эталон)."""
    q = set(content_stems(question))
    candidates = sentences(text)
    if not candidates:
        return text.strip()
    return max(candidates, key=lambda s: (len(q & set(content_stems(s))), -len(s)))
