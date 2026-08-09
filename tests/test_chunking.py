
from pathlib import Path

from faar.chunking import build_chunks
from faar.settings import RetrievalSettings
from faar.types import Phase0Example


def _example(text: str, example_id: str = "ex1", page_texts: dict[int, str] | None = None) -> Phase0Example:
    return Phase0Example(
        example_id=example_id,
        doc_name="doc",
        question="Q?",
        correct_answer="A",
        page_ids=[0],
        ocr_text=text,
        ocr_text_path=Path("/tmp/ex.txt"),
        gt_text_path=None,
        image_paths=[],
        metadata={"page_texts": page_texts if page_texts is not None else {0: text}},
    )


def test_small_document_produces_single_chunk() -> None:
    chunks = build_chunks(_example("the quick brown fox"), RetrievalSettings())
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "ex1-p0-c0"
    assert chunks[0].page_id == 0
    assert chunks[0].text == "the quick brown fox"


def test_overlap_shares_words_across_boundaries() -> None:
    words = [f"w{i}" for i in range(12)]
    settings = RetrievalSettings(chunk_size_words=6, chunk_overlap_words=3)
    chunks = build_chunks(_example(" ".join(words)), settings)
    covered = set()
    for chunk in chunks:
        covered.update(chunk.text.split())
    assert covered == set(words)
    assert len(chunks) == 3
    assert "w3 w4 w5" in chunks[1].text
    assert "w6 w7 w8" in chunks[2].text
    assert chunks[0].chunk_id == "ex1-p0-c0"
    assert chunks[2].chunk_id == "ex1-p0-c2"


def test_exact_multiple_of_chunk_size_has_no_trailing_dup() -> None:
    words = [f"w{i}" for i in range(6)]
    settings = RetrievalSettings(chunk_size_words=6, chunk_overlap_words=0)
    chunks = build_chunks(_example(" ".join(words)), settings)
    assert len(chunks) == 1
    assert chunks[0].text == " ".join(words)


def test_single_word_chunk_size() -> None:
    settings = RetrievalSettings(chunk_size_words=1, chunk_overlap_words=0)
    chunks = build_chunks(_example("alpha beta gamma"), settings)
    assert [chunk.text for chunk in chunks] == ["alpha", "beta", "gamma"]


def test_unicode_words_are_preserved() -> None:
    text = "héllo wörld 中文测试 ünïcode naïve"
    chunks = build_chunks(_example(text), RetrievalSettings())
    assert chunks[0].text == text
    assert len(chunks[0].text.split()) == 5


def test_multiple_pages_use_page_ids_and_unique_chunk_ids() -> None:
    example = _example("x", page_texts={0: "page zero content", 7: "page seven content"})
    chunks = build_chunks(example, RetrievalSettings())
    assert [chunk.page_id for chunk in chunks] == [0, 7]
    ids = [chunk.chunk_id for chunk in chunks]
    assert ids == ["ex1-p0-c0", "ex1-p7-c0"]
    assert len(set(ids)) == len(ids)


def test_empty_and_whitespace_pages_produce_no_chunks() -> None:
    example = _example("x", page_texts={0: "", 1: "   \n\t ", 2: "real words here"})
    chunks = build_chunks(example, RetrievalSettings())
    assert len(chunks) == 1
    assert chunks[0].page_id == 2


def test_overlap_greater_than_zero_is_not_lossy() -> None:
    words = [f"tok{i}" for i in range(20)]
    settings = RetrievalSettings(chunk_size_words=8, chunk_overlap_words=4)
    chunks = build_chunks(_example(" ".join(words)), settings)
    flat = " ".join(chunk.text for chunk in chunks)
    for word in words:
        assert word in flat
    assert len(chunks) == 4
