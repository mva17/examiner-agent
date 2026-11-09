from examiner import logic
from examiner.schemas import AnswerRecord, Question, Source


def q(id_, section="Теория", kind="control"):
    return Question(id=id_, text=f"вопрос {id_}", topic="т", origin="manual", kind=kind,
                    evidence=[Source(work="W", page=1, section=section)])


def rec(id_, verdict, kind="control"):
    return AnswerRecord(question_id=id_, question=id_, topic="т", answer="", verdict=verdict, explanation="", kind=kind)


def test_question_id_is_stable():
    assert logic.question_id("Вопрос  про  микрометр", "W") == logic.question_id("вопрос про микрометр", "W")
    assert logic.question_id("a", "W1") != logic.question_id("a", "W2")


def test_commands():
    assert logic.is_stop("Стоп!") and not logic.is_stop("стоп-кран это")
    assert logic.is_dont_know("не знаю") and logic.is_dont_know("  ") and not logic.is_dont_know("не знаю точно, но 0,01 мм")


def test_order_puts_mistakes_first_and_dedups():
    order = logic.order_questions([q("a"), q("b")], [q("c")], [q("b")], limit=5)
    assert [x.id for x in order] == ["b", "a", "c"]


def test_pick_next_prefers_weak_section():
    queue = [q("a", "Теория"), q("b", "Порядок"), q("c", "Теория")]
    history = [rec("a", "неверно")]
    assert logic.pick_next(queue, history, {"a"}).id == "c"
    assert logic.pick_next(queue, [rec("a", "верно")], {"a"}).id == "b"
    assert logic.pick_next(queue, [], {"a", "b", "c"}) is None


def test_summary():
    text = logic.summary_text("маятник", [rec("a", "верно"), rec("b", "частично"), rec("f", "верно", "followup")])
    assert "1.5 из 2" in text and "Повторите" in text
    assert logic.score_of([rec("a", "верно")]) == (1.0, 1)
