from examiner.config import ROOT
from examiner.knowledge import BM25ManualSearch, ChromaManualSearch, parse_manual, split_control_questions

MANUALS = ROOT / "data" / "manuals"


def test_demo_manuals_are_marked_as_demo():
    for path in MANUALS.glob("*.md"):
        assert "НЕ методичка НИЯУ МИФИ" in path.read_text(encoding="utf-8")


def test_parse_manual_pages_and_sections():
    chunks = parse_manual(MANUALS / "demo_micrometer.md")
    assert {c.work for c in chunks} == {"ДЕМО-1"}
    sections = {(c.section, c.page) for c in chunks}
    assert ("Микрометр", 1) in sections
    assert ("Контрольные вопросы", 3) in sections


def test_bm25_finds_right_work():
    search = BM25ManualSearch(MANUALS)
    assert search.search("микрометр трещотка", k=3)[0].source.work == "ДЕМО-1"
    assert search.search("период маятника", k=3)[0].source.work == "ДЕМО-2"


def test_search_always_returns_something_even_off_topic():
    """Ровно та проблема, из-за которой агенту нужна проверка релевантности."""
    hits = BM25ManualSearch(MANUALS).search("фотоэффект работа выхода", k=5)
    assert len(hits) == 5
    assert max(h.coverage for h in hits) < 0.5


def test_work_filter_and_control_questions():
    search = BM25ManualSearch(MANUALS)
    assert all(h.source.work == "ДЕМО-2" for h in search.search("погрешность", k=5, work="ДЕМО-2"))
    control = search.control_questions("ДЕМО-1")
    assert len(control) == 5
    assert all(c.page == 3 for c in control)
    assert "трещотка" in control[1].quote


def test_split_control_questions_multiline():
    text = "1. Первый вопрос\nпродолжение.\n2) Второй вопрос?\n\n# заголовок"
    assert split_control_questions(text) == ["Первый вопрос продолжение.", "Второй вопрос?"]


class FakeCollection:
    def __init__(self):
        self.last = {}

    def query(self, n_results, where, include, query_texts=None, query_embeddings=None):
        self.last = {"texts": query_texts, "embeddings": query_embeddings, "where": where}
        return {
            "documents": [["Шаг винта 0,5 мм", "1. Чему равна цена деления?\n2. Зачем трещотка?"]],
            "metadatas": [[{"lab": "1.01", "page_number": 4, "heading": "Микрометр"}, {"work": "1.01", "page": 9, "section": "Контрольные вопросы"}]],
            "distances": [[0.2, 0.5]],
        }

    def get(self, where, include):
        return {"documents": ["1. Чему равна цена деления?\n2. Зачем трещотка?"], "metadatas": [{"work": "1.01", "page": 9, "section": "Контрольные вопросы"}]}


def test_chroma_adapter_maps_metadata():
    search = ChromaManualSearch(FakeCollection())
    hits = search.search("микрометр шаг винта", k=2)
    assert hits[0].source.work == "1.01" and hits[0].source.page == 4 and hits[0].source.section == "Микрометр"
    assert hits[0].score == 0.8
    control = search.control_questions("1.01")
    assert [c.quote for c in control] == ["Чему равна цена деления?", "Зачем трещотка?"]
    assert control[0].page == 9


def test_chroma_adapter_uses_query_embeddings_and_work_filter():
    col = FakeCollection()
    ChromaManualSearch(col, embed_query=lambda text: [0.1, 0.2]).search("микрометр", k=2, work="1.01")
    assert col.last == {"texts": None, "embeddings": [[0.1, 0.2]], "where": {"work": "1.01"}}
