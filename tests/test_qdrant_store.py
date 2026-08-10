"""Юнит-тесты store/qdrant_store.py — реальный embedded Qdrant (`path=`) на
tmp_path (быстро, офлайн, не требует Docker/сети — см. `QdrantStore.__init__`
docstring: только для тестов, продовый код всегда ходит в Docker-сервер).
"""

from __future__ import annotations

from src.store.qdrant_store import Chunk, QdrantStore, document_to_hit, hit_to_document


def _store(tmp_path, **kwargs) -> QdrantStore:
    return QdrantStore(path=str(tmp_path / "qdrant"), **kwargs)


def test_url_and_citation_count_round_trip_through_rebuild(tmp_path):
    store = _store(tmp_path)
    store.rebuild(
        [
            Chunk(
                id="11111111-1111-1111-1111-111111111111", text="text", source_id="a",
                source_title="A", section="", vector=[0.1, 0.2],
                url="https://example.com/a", citation_count=42,
            )
        ]
    )

    hits = store.search_dense([0.1, 0.2], k=1)
    assert hits[0]["url"] == "https://example.com/a"
    assert hits[0]["citation_count"] == 42


def test_url_and_citation_count_default_sentinels_when_unset(tmp_path):
    store = _store(tmp_path)
    store.rebuild(
        [Chunk(id="11111111-1111-1111-1111-111111111111", text="text", source_id="a",
               source_title="A", section="", vector=[0.1])]
    )

    hits = store.search_dense([0.1], k=1)
    assert hits[0]["url"] == ""
    assert hits[0]["citation_count"] == -1


def test_rebuild_then_add_chunks_both_searchable(tmp_path):
    store = _store(tmp_path)
    store.rebuild(
        [Chunk(id="11111111-1111-1111-1111-111111111111", text="t1", source_id="a",
               source_title="A", section="", vector=[0.1])]
    )
    store.add_chunks(
        [
            Chunk(
                id="22222222-2222-2222-2222-222222222222", text="t2", source_id="b",
                source_title="B", section="", vector=[0.2], citation_count=99,
            )
        ]
    )

    hits = store.search_dense([0.2], k=2)
    citation_counts = {h["source_id"]: h["citation_count"] for h in hits}
    assert citation_counts == {"a": -1, "b": 99}


def test_has_source_true_for_existing_false_for_missing(tmp_path):
    """Реальный баг, найденный вручную: approximate count() по
    неиндексированному полю на маленькой коллекции давал неверный результат
    в обе стороны. Фикс — payload-индекс на source_id + exact=True."""
    store = _store(tmp_path)
    store.rebuild(
        [Chunk(id="11111111-1111-1111-1111-111111111111", text="t1", source_id="a",
               source_title="A", section="", vector=[0.1])]
    )

    assert store.has_source("a") is True
    assert store.has_source("does-not-exist") is False


def test_non_uuid_chunk_id_is_deterministically_mapped_and_deduped(tmp_path):
    """`Chunk.id` в реальном коде проекта — всегда `str(uuid.uuid4())`, но
    `_point_id()` обязан детерминированно обрабатывать и произвольные строки
    (id кандидата вроде "arxiv:1234"), не роняя upsert — тот же исходный id
    должен маппиться на ту же точку Qdrant (не плодить дубликаты)."""
    store = _store(tmp_path)
    store.rebuild([Chunk(id="arxiv:1234", text="v1", source_id="a", source_title="A", section="", vector=[0.1])])
    store.add_chunks([Chunk(id="arxiv:1234", text="v2", source_id="a", source_title="A", section="", vector=[0.2])])

    hits = store.search_dense([0.2], k=5)
    assert len(hits) == 1
    assert hits[0]["text"] == "v2"


class _FakeEmbeddings:
    """`langchain_core.embeddings.Embeddings`-совместимая заглушка — тесты
    VectorStore-интерфейса не должны грузить реальный bge-m3."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 0.0] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 0.0]


def test_add_texts_and_similarity_search_round_trip(tmp_path, monkeypatch):
    from src.providers import embed

    monkeypatch.setattr(embed, "embed_sparse", lambda texts: [{} for _ in texts])
    store = _store(tmp_path, embedding=_FakeEmbeddings())
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


def test_from_texts_builds_a_searchable_store(tmp_path, monkeypatch):
    from src.providers import embed

    monkeypatch.setattr(embed, "embed_sparse", lambda texts: [{} for _ in texts])
    store = QdrantStore.from_texts(
        ["dogs are great"], _FakeEmbeddings(),
        metadatas=[{"source_id": "b", "source_title": "B"}],
        path=str(tmp_path / "qdrant"),
    )
    docs = store.similarity_search("dogs", k=1)
    assert docs[0].page_content == "dogs are great"


def test_hit_to_document_and_back_round_trips():
    hit = {"text": "hello", "source_id": "a", "citation_count": 5}
    doc = hit_to_document(hit)
    assert doc.page_content == "hello"
    assert doc.metadata == {"source_id": "a", "citation_count": 5}
    assert document_to_hit(doc) == hit
