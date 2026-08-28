"""Юнит-тесты store/qdrant_store.py — реальный embedded Qdrant (`path=`) на
tmp_path (быстро, офлайн, не требует Docker/сети — см. `QdrantStore.__init__`
docstring: только для тестов, продовый код всегда ходит в Docker-сервер).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from qdrant_client import QdrantClient

from src.store import qdrant_store
from src.store.qdrant_store import (
    Chunk, QdrantStore, document_to_hit, hit_to_document, published_timestamp,
)


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


def _dated_chunk(chunk_id: str, source_id: str, published_date: str, **kwargs) -> Chunk:
    return Chunk(
        id=chunk_id, text=f"Title {source_id}\nAbstract {source_id}", source_id=source_id,
        source_title=f"Title {source_id}", section="abstract", vector=[0.1, 0.2],
        published_date=published_date, **kwargs,
    )


def test_published_timestamp_parses_iso_and_rejects_junk():
    assert published_timestamp("2026-08-10T12:00:00Z") == pytest.approx(
        datetime(2026, 8, 10, 12, 0, tzinfo=timezone.utc).timestamp()
    )
    assert published_timestamp("") is None
    assert published_timestamp("not-a-date") is None


def test_load_pool_returns_only_points_inside_the_window(tmp_path):
    """Digest поднимает по этому окно «за последние N дней» прямо из индекса,
    вместо повторной выгрузки тех же статей из arXiv."""
    store = _store(tmp_path)
    now = datetime.now(timezone.utc)
    store.rebuild(
        [
            _dated_chunk("11111111-1111-1111-1111-111111111111", "fresh",
                         (now - timedelta(days=1)).isoformat(), authors=["A. Uthor"]),
            _dated_chunk("22222222-2222-2222-2222-222222222222", "old",
                         (now - timedelta(days=40)).isoformat()),
        ]
    )

    cutoff = (now - timedelta(days=7)).timestamp()
    payloads = store.load_pool(cutoff)
    assert [p["source_id"] for p in payloads] == ["fresh"]
    assert payloads[0]["authors"] == ["A. Uthor"]
    assert payloads[0]["source_title"] == "Title fresh"


def test_load_pool_skips_points_without_a_publication_date(tmp_path):
    """`cli index`/funnel кладут чанки без даты — в digest-пул им нельзя:
    непонятно, попадают ли они в окно."""
    store = _store(tmp_path)
    store.rebuild(
        [
            Chunk(id="33333333-3333-3333-3333-333333333333", text="t", source_id="undated",
                  source_title="U", section="", vector=[0.1, 0.2]),
        ]
    )
    assert store.load_pool(0) == []
    assert store.oldest_published_ts() is None


def test_oldest_published_ts_reports_the_earliest_known_paper(tmp_path):
    store = _store(tmp_path)
    now = datetime.now(timezone.utc)
    oldest = now - timedelta(days=40)
    store.rebuild(
        [
            _dated_chunk("11111111-1111-1111-1111-111111111111", "a", (now - timedelta(days=1)).isoformat()),
            _dated_chunk("22222222-2222-2222-2222-222222222222", "b", oldest.isoformat()),
        ]
    )
    assert store.oldest_published_ts() == pytest.approx(oldest.timestamp())


def test_load_pool_on_missing_collection_is_empty_not_an_error(tmp_path):
    store = _store(tmp_path, collection_name="never-created")
    assert store.load_pool(0) == []
    assert store.oldest_published_ts() is None


def test_add_chunks_splits_large_upserts_into_batches(tmp_path, monkeypatch):
    """Один upsert на весь digest-пул упирался в лимит тела запроса Qdrant
    (2143 чанка = 53МБ при лимите 32МБ, поймано реальным прогоном)."""
    import src.store.qdrant_store as qs

    monkeypatch.setattr(qs, "_UPSERT_BATCH", 2)
    store = _store(tmp_path)
    chunks = [
        Chunk(id=str(uuid.uuid4()), text=f"t{i}", source_id=f"s{i}", source_title="T",
              section="", vector=[0.1, 0.2])
        for i in range(5)
    ]

    sizes = []
    real_upsert = QdrantClient.upsert
    monkeypatch.setattr(
        QdrantClient, "upsert",
        lambda self, collection_name, points, **kw: (
            sizes.append(len(points)), real_upsert(self, collection_name, points, **kw)
        )[1],
    )

    store.add_chunks(chunks)
    assert sizes == [2, 2, 1]
    assert all(store.has_source(f"s{i}") for i in range(5))


def test_a_stale_connection_is_retried_once(monkeypatch):
    """Клиент держит пул keep-alive-соединений. Сервер закрывает простоявшее,
    пул об этом не знает и отдаёт его следующему запросу — тот падает, ничего
    не выполнив. Поймано на замере: вопрос падал после того, как предыдущий
    считался 4.5 минуты."""
    store = QdrantStore(url="http://localhost:6333", collection_name="c")
    clients = []
    calls = []

    class _Client:
        def __init__(self):
            clients.append(self)

        def collection_exists(self, name):
            calls.append(name)
            if len(calls) == 1:
                raise httpx.RemoteProtocolError("Server disconnected without sending a response.")
            return True

    monkeypatch.setattr(qdrant_store, "QdrantClient", lambda **kw: _Client())

    assert store._ensure_collection() == "c"
    assert len(calls) == 2, "первый вызов оборвался, второй должен пройти"
    assert len(clients) == 2, "клиента надо пересоздать — старый пул держит мёртвое соединение"


def test_a_second_disconnect_is_not_swallowed(monkeypatch):
    """Ровно одна попытка. Второй обрыв подряд — это уже недоступный Qdrant,
    и притворяться, что это лечится повтором, значит прятать поломку."""
    store = QdrantStore(url="http://localhost:6333", collection_name="c")

    class _AlwaysDown:
        def collection_exists(self, name):
            raise httpx.RemoteProtocolError("Server disconnected without sending a response.")

    monkeypatch.setattr(qdrant_store, "QdrantClient", lambda **kw: _AlwaysDown())

    with pytest.raises(httpx.RemoteProtocolError):
        store._ensure_collection()
