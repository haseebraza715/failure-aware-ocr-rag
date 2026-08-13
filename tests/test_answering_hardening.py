from faar.answering import answer_from_hits
from faar.types import Chunk, RetrievalHit


def _hit(text: str, fused: float = 0.9) -> RetrievalHit:
    return RetrievalHit(
        chunk=Chunk(chunk_id="c1", example_id="ex1", doc_name="doc", page_id=0, text=text),
        bm25_score=0.5,
        dense_score=0.5,
        fused_score=fused,
    )


def test_answer_from_hits_without_hits_is_empty() -> None:
    result = answer_from_hits("What is the answer?", [])
    assert result["answer"] == ""
    assert result["answer_mode"] == "extractive_overlap"
    assert result["source_chunk_id"] == ""


def test_answer_from_hits_with_empty_chunk_text_does_not_crash() -> None:
    result = answer_from_hits("What is the answer?", [_hit("")])
    assert result["answer"] == ""


def test_answer_from_hits_stopword_only_question() -> None:
    result = answer_from_hits("What is the of in?", [_hit("nested deeply structured text here")])
    assert result["answer"] == "nested deeply structured text here"


def test_answer_from_hits_segments_long_lines() -> None:
    long_line = (
        "The committee reviewed the quarterly financial statements and approved "
        "the budget on Tuesday. Spending remains within the approved envelope. "
        "Unrelated trailing sentence here."
    )
    result = answer_from_hits("What day was the budget approved?", [_hit(long_line)])
    assert result["answer"]
    assert "Tuesday" in result["answer"] or "approved envelope" in result["answer"]


def test_answer_from_hits_extracts_currency() -> None:
    result = answer_from_hits(
        "How much is the copay for a specialist visit?",
        [_hit("The copay for a specialist visit is $1,200 per year.")],
    )
    assert result["answer"] == "$1,200"
    assert result["answer_mode"] == "numeric_span"


def test_answer_from_hits_extracts_width_unit() -> None:
    result = answer_from_hits(
        "What is the maximum width of the device?",
        [_hit("The device has a maximum width of 20mm.")],
    )
    assert result["answer"] == "20mm"


def test_answer_from_hits_extracts_duration() -> None:
    result = answer_from_hits(
        "How many days do we have to file the appeal?",
        [_hit("You have 5 days to file the appeal.")],
    )
    assert result["answer"] == "5 days"


def test_answer_from_hits_negative_number() -> None:
    result = answer_from_hits(
        "What is the difference between the two readings?",
        [_hit("The difference is -3.5 degrees.")],
    )
    assert result["answer"] == "-3.5"


def test_answer_from_hits_prefers_higher_fused_score_candidate() -> None:
    first = _hit("irrelevant filler text with nothing useful", fused=0.1)
    second = _hit("the percentage of change is 88% annually", fused=0.9)
    result = answer_from_hits("What percentage changed?", [first, second])
    assert result["answer"] == "88%"
    assert result["source_chunk_id"] == "c1"
