import pytest

from src.ingest.chunk import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_short_text_is_a_single_chunk():
    chunks = chunk_text("hello world", size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"
    assert chunks[0].index == 0


def test_long_text_splits_with_overlap():
    text = "a" * 250
    chunks = chunk_text(text, size=100, overlap=20)
    assert [c.index for c in chunks] == [0, 1, 2]
    assert [len(c.text) for c in chunks] == [100, 100, 90]
    # хвост первого чанка повторяется в начале второго (перекрытие)
    assert chunks[0].text[-20:] == chunks[1].text[:20]


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("abcdef", size=10, overlap=10)
