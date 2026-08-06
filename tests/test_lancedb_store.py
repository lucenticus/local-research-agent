"""Юнит-тесты store/lancedb_store.py — реальный LanceDB на tmp_path (быстро,
не требует ML-моделей, только вставка/чтение)."""

from __future__ import annotations

from src.store.lancedb_store import Chunk, LanceDBStore, document_to_hit, hit_to_document


def test_url_and_citation_count_round_trip_through_rebuild(tmp_path):
    store = LanceDBStore(db_path=tmp_path / "lancedb")
    store.rebuild(
        [
            Chunk(
                id="1", text="text", source_id="a", source_title="A", section="",
                vector=[0.1, 0.2], url="https://example.com/a", citation_count=42,
            )
        ]
    )

    hits = store.search_dense([0.1, 0.2], k=1)
    assert hits[0]["url"] == "https://example.com/a"
    assert hits[0]["citation_count"] == 42


def test_url_and_citation_count_default_sentinels_when_unset(tmp_path):
    store = LanceDBStore(db_path=tmp_path / "lancedb")
    store.rebuild(
        [Chunk(id="1", text="text", source_id="a", source_title="A", section="", vector=[0.1])]
    )

    hits = store.search_dense([0.1], k=1)
    assert hits[0]["url"] == ""
    assert hits[0]["citation_count"] == -1


def test_mixed_sentinel_and_real_citation_count_across_add_chunks(tmp_path):
    """Ловушка, найденная вручную: смешение None/int в одной колонке между
    batch'ами (rebuild + add_chunks) в LanceDB даёт null-типизированную
    колонку, которая потом не принимает реальные int. Сентинел -1 вместо
    None — обязателен, этот тест закрепляет поведение."""
    store = LanceDBStore(db_path=tmp_path / "lancedb")
    store.rebuild(
        [Chunk(id="1", text="t1", source_id="a", source_title="A", section="", vector=[0.1])]
    )
    store.add_chunks(
        [
            Chunk(
                id="2", text="t2", source_id="b", source_title="B", section="",
                vector=[0.2], citation_count=99,
            )
        ]
    )

    hits = store.search_dense([0.2], k=2)
    citation_counts = {h["source_id"]: h["citation_count"] for h in hits}
    assert citation_counts == {"a": -1, "b": 99}


class _FakeEmbeddings:
    """`langchain_core.embeddings.Embeddings`-совместимая заглушка — тесты
    VectorStore-интерфейса не должны грузить реальный bge-m3."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 0.0] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 0.0]


def test_add_texts_and_similarity_search_round_trip(tmp_path):
    store = LanceDBStore(db_path=tmp_path / "lancedb", embedding=_FakeEmbeddings())
    ids = store.add_texts(
        ["about whales"],
        metadatas=[{"source_id": "a", "source_title": "A", "url": "https://x", "citation_count": 7}],
    )

    docs = store.similarity_search("whales", k=1)
    assert docs[0].page_content == "about whales"
    assert docs[0].metadata["source_id"] == "a"
    assert docs[0].metadata["citation_count"] == 7
    assert store.has_source("a")
    assert ids and store.has_source("a")


def test_from_texts_builds_a_searchable_store(tmp_path):
    store = LanceDBStore.from_texts(
        ["dogs are great"], _FakeEmbeddings(),
        metadatas=[{"source_id": "b", "source_title": "B"}],
        db_path=tmp_path / "lancedb",
    )
    docs = store.similarity_search("dogs", k=1)
    assert docs[0].page_content == "dogs are great"


def test_hit_to_document_and_back_round_trips():
    hit = {"text": "hello", "source_id": "a", "citation_count": 5}
    doc = hit_to_document(hit)
    assert doc.page_content == "hello"
    assert doc.metadata == {"source_id": "a", "citation_count": 5}
    assert document_to_hit(doc) == hit
