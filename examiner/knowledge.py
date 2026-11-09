"""Поиск по методичкам.

Экзаменатор работает с поиском только через интерфейс ManualSearch. Сейчас
есть две реализации:

  * BM25ManualSearch — локальный поиск по data/manuals/*.md (ДЕМО-тексты,
    не методички МИФИ). Работает без ключа и интернета.
  * ChromaManualSearch — адаптер к базе методичек Оскара (ChromaDB).
    Поля метаданных: work / page / section (как в общем контракте Source).

Важно: любой поиск ВСЕГДА что-то возвращает (top-k ближайших), даже если
по теме ничего нет. Поэтому агент после поиска проверяет релевантность
(узел check_relevance в agent.py).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .schemas import Source
from .text import content_stems

CONTROL_SECTION = "контрольные вопросы"


@dataclass(frozen=True)
class Hit:
    source: Source
    score: float  # чем больше, тем ближе; шкала зависит от реализации
    coverage: float  # доля слов запроса, найденных во фрагменте (0..1)


class ManualSearch(Protocol):
    def search(self, query: str, k: int = 5, work: str | None = None) -> list[Hit]: ...

    def control_questions(self, work: str) -> list[Source]: ...


# --- Общие функции --------------------------------------------------------

_NUMBERED = re.compile(r"^\s*(\d{1,2})[.)]\s+(.+?)\s*$")


def split_control_questions(text: str) -> list[str]:
    """«1. Вопрос…\\n2. Вопрос…» → список вопросов (многострочные склеиваются)."""
    items: list[str] = []
    for line in text.splitlines():
        match = _NUMBERED.match(line)
        if match:
            items.append(match.group(2))
        elif items and line.strip() and not line.lstrip().startswith("#"):
            items[-1] += " " + line.strip()
    return [q for q in items if len(q) > 5]


def coverage(query: str, text: str) -> float:
    q = set(content_stems(query))
    if not q:
        return 0.0
    return len(q & set(content_stems(text))) / len(q)


# --- Локальный BM25 по Markdown ------------------------------------------

_PAGE = re.compile(r"<!--\s*page:\s*(\d+)\s*-->")
_HEADING = re.compile(r"^(#{1,3})\s+(.+)$")


@dataclass
class Chunk:
    work: str
    title: str
    page: int
    section: str
    text: str

    def to_source(self) -> Source:
        return Source(work=self.work, title=self.title, page=self.page, section=self.section, quote=self.text[:1200])


def parse_manual(path: Path) -> list[Chunk]:
    """Markdown с front matter (work, title) и метками <!-- page: N --> → куски по разделам."""
    raw = path.read_text(encoding="utf-8")
    meta: dict[str, str] = {}
    if raw.startswith("---"):
        _, header, raw = raw.split("---", 2)
        for line in header.strip().splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                meta[key.strip()] = value.strip()
    work = meta.get("work", path.stem)
    title = meta.get("title", path.stem)

    chunks: list[Chunk] = []
    page, section, buf = 1, "", []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            chunks.append(Chunk(work, title, page, section, body))
        buf.clear()

    for line in raw.splitlines():
        page_mark = _PAGE.search(line)
        if page_mark:
            flush()
            page = int(page_mark.group(1))
            continue
        heading = _HEADING.match(line)
        if heading:
            flush()
            section = heading.group(2).strip()
            continue
        buf.append(line)
    flush()
    return chunks


class BM25ManualSearch:
    """Okapi BM25 без внешних библиотек. k1, b — стандартные значения."""

    def __init__(self, manuals_dir: Path, k1: float = 1.5, b: float = 0.75):
        self.chunks: list[Chunk] = []
        for path in sorted(Path(manuals_dir).glob("*.md")):
            self.chunks.extend(parse_manual(path))
        self.k1, self.b = k1, b
        self._docs = [Counter(content_stems(f"{c.title} {c.section} {c.text}")) for c in self.chunks]
        self._len = [sum(d.values()) for d in self._docs]
        self._avg = (sum(self._len) / len(self._len)) if self._len else 1.0
        df: Counter[str] = Counter()
        for doc in self._docs:
            df.update(doc.keys())
        n = len(self._docs)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 5, work: str | None = None) -> list[Hit]:
        terms = content_stems(query)
        scored: list[tuple[float, int]] = []
        for i, doc in enumerate(self._docs):
            if work and self.chunks[i].work != work:
                continue
            score = 0.0
            for t in terms:
                tf = doc.get(t, 0)
                if tf:
                    norm = tf + self.k1 * (1 - self.b + self.b * self._len[i] / self._avg)
                    score += self._idf.get(t, 0.0) * tf * (self.k1 + 1) / norm
            scored.append((score, i))
        # Как и векторный поиск, возвращаем top-k ВСЕГДА — даже с нулевым счётом.
        scored.sort(key=lambda x: (-x[0], x[1]))
        hits = []
        for score, i in scored[:k]:
            c = self.chunks[i]
            hits.append(Hit(c.to_source(), round(score, 4), coverage(query, f"{c.title} {c.section} {c.text}")))
        return hits

    def control_questions(self, work: str) -> list[Source]:
        result = []
        for c in self.chunks:
            if c.work == work and c.section.lower().startswith(CONTROL_SECTION):
                for q in split_control_questions(c.text):
                    result.append(Source(work=c.work, title=c.title, page=c.page, section=c.section, quote=q))
        return result


# --- Адаптер к ChromaDB Оскара -------------------------------------------


class ChromaManualSearch:
    """Обёртка над коллекцией ChromaDB с методичками.

    Ожидаемые метаданные кусков: work (код/номер работы), page (int), section,
    по возможности title. Если у Оскара поля называются иначе, поправьте FIELD_MAP.
    collection — объект chromadb.Collection (или любой с методами query/get).
    """

    FIELD_MAP = {"work": ("work", "lab", "work_id"), "page": ("page", "page_number"), "section": ("section", "heading")}

    def __init__(self, collection: Any, embed_query: Any = None):
        self.collection = collection
        # Если коллекция построена эмбеддингами через API (Qwen3-Embedding), запрос надо
        # векторизовать той же моделью: передайте embed_query=embeddings.embed_query.
        self.embed_query = embed_query

    @classmethod
    def from_settings(cls, settings, embedding_function: Any = None, embed_query: Any = None) -> "ChromaManualSearch":
        import chromadb  # необязательная зависимость: pip install chromadb

        client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        kwargs = {"embedding_function": embedding_function} if embedding_function else {}
        return cls(client.get_collection(settings.chroma_collection, **kwargs), embed_query=embed_query)

    def _field(self, meta: dict, name: str, default: Any = "") -> Any:
        for key in self.FIELD_MAP[name]:
            if meta.get(key) not in (None, ""):
                return meta[key]
        return default

    def _source(self, doc: str, meta: dict | None) -> Source:
        meta = meta or {}
        page = self._field(meta, "page", None)
        return Source(
            work=str(self._field(meta, "work", "?")),
            title=str(meta.get("title", "")),
            page=int(page) if page not in (None, "") else None,
            section=str(self._field(meta, "section", "")),
            quote=(doc or "")[:1200],
        )

    def search(self, query: str, k: int = 5, work: str | None = None) -> list[Hit]:
        target = {"query_embeddings": [self.embed_query(query)]} if self.embed_query else {"query_texts": [query]}
        res = self.collection.query(
            **target,
            n_results=k,
            where={"work": work} if work else None,
            include=["documents", "metadatas", "distances"],
        )
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        hits = []
        for doc, meta, dist in zip(docs, metas, dists):
            src = self._source(doc, meta)
            hits.append(Hit(src, round(1.0 - float(dist), 4), coverage(query, f"{src.section} {doc}")))
        return hits

    def control_questions(self, work: str) -> list[Source]:
        res = self.collection.get(where={"work": work}, include=["documents", "metadatas"])
        result = []
        for doc, meta in zip(res.get("documents") or [], res.get("metadatas") or []):
            src = self._source(doc, meta)
            if src.section.lower().startswith(CONTROL_SECTION) or CONTROL_SECTION in (doc or "").lower()[:80]:
                for q in split_control_questions(doc):
                    result.append(src.model_copy(update={"quote": q}))
        return result


def build_search(settings) -> ManualSearch:
    if settings.search_backend == "chroma":
        return ChromaManualSearch.from_settings(settings)
    return BM25ManualSearch(settings.manuals_dir)
