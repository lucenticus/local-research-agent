"""Юнит-тесты store/lancedb_store.py — реальный LanceDB на tmp_path (быстро,
не требует ML-моделей, только вставка/чтение)."""

from __future__ import annotations

from src.store.lancedb_store import Chunk, LanceDBStore


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
