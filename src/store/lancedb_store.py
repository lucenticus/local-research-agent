"""LanceDB-хранилище чанков: on-disk индекс, dense vector search + hybrid.

Milestone 1: добавлен FTS-индекс по тексту и гибридный поиск (dense + FTS,
слияние через RRF — реранкер LanceDB по умолчанию, проверено на этой версии
lancedb 0.36.0). Полный rebuild таблицы на каждой индексации, без
инкрементального upsert (см. Milestone 3 — воронка дополняет индекс на лету).

Отдельный sparse-выход bge-m3 (lexical weights) не используется: LanceDB FTS
со стеммером `language="Russian"` уже даёт лексический сигнал на кириллице
(проверено вручную на этой машине) — использовать оба было бы дублирующим
сигналом без явной пользы на масштабе этого проекта.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lancedb
from lancedb.index import FTS

from .. import config


@dataclass
class Chunk:
    id: str
    text: str
    source_id: str
    source_title: str
    section: str
    vector: list[float]
    # "" / -1 — сентинелы "неизвестно" (локальный корпус, источник без URL/
    # цитируемости), не None: смешение None/int в одной LanceDB-колонке через
    # разные batch'и (rebuild + add_chunks) даёт null-типизированную колонку
    # в Arrow, которая потом не принимает реальные int — проверено вручную.
    url: str = ""
    citation_count: int = -1


class LanceDBStore:
    def __init__(self, db_path: Path | None = None, table_name: str | None = None):
        self._db_path = db_path or config.INDEX_DIR
        self._table_name = table_name or config.INDEX_TABLE
        self._db: Any = None
        self._table: Any = None

    def _connect(self) -> Any:
        if self._db is None:
            self._db_path.mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(str(self._db_path))
        return self._db

    def rebuild(self, chunks: list[Chunk]) -> None:
        """Полная пересборка таблицы из chunks."""
        db = self._connect()
        if not chunks:
            if self._table_name in db.list_tables().tables:
                db.drop_table(self._table_name)
            self._table = None
            return
        rows = [
            {
                "id": c.id,
                "text": c.text,
                "source_id": c.source_id,
                "source_title": c.source_title,
                "section": c.section,
                "vector": c.vector,
                "url": c.url,
                "citation_count": c.citation_count,
            }
            for c in chunks
        ]
        self._table = db.create_table(self._table_name, data=rows, mode="overwrite")
        self._table.create_index("text", config=FTS(language="Russian", stem=True))

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Инкрементально дописывает chunks (создаёт таблицу, если её ещё нет).

        Используется воронкой (agent/funnel.py) — прочитанное пополняет индекс
        на лету, без полной пересборки (в отличие от `rebuild`, который зовёт
        только `cli.py index` для локального корпуса).
        """
        if not chunks:
            return
        db = self._connect()
        rows = [
            {
                "id": c.id,
                "text": c.text,
                "source_id": c.source_id,
                "source_title": c.source_title,
                "section": c.section,
                "vector": c.vector,
                "url": c.url,
                "citation_count": c.citation_count,
            }
            for c in chunks
        ]
        if self._table_name in db.list_tables().tables:
            table = self._ensure_table()
            table.add(rows)
        else:
            self._table = db.create_table(self._table_name, data=rows, mode="overwrite")
        self._table.create_index("text", config=FTS(language="Russian", stem=True), replace=True)

    def has_source(self, source_id: str) -> bool:
        """Есть ли уже чанки с этим source_id — кэш-хит для funnel (не читать повторно)."""
        db = self._connect()
        if self._table_name not in db.list_tables().tables:
            return False
        table = self._ensure_table()
        escaped = source_id.replace("'", "''")
        return table.count_rows(filter=f"source_id = '{escaped}'") > 0

    def _ensure_table(self) -> Any:
        if self._table is None:
            db = self._connect()
            if self._table_name not in db.list_tables().tables:
                raise RuntimeError(
                    "Индекс пуст — сначала выполни `python -m src.cli index`"
                )
            self._table = db.open_table(self._table_name)
        return self._table

    def search_dense(self, query_vector: list[float], k: int) -> list[dict[str, Any]]:
        """Dense-only поиск — Milestone 0 baseline, используется для сравнения в eval."""
        table = self._ensure_table()
        return table.search(query_vector).limit(k).to_list()

    def search_hybrid(
        self, query_text: str, query_vector: list[float], k: int
    ) -> list[dict[str, Any]]:
        """Dense + FTS, слияние через RRF (реранкер LanceDB по умолчанию)."""
        table = self._ensure_table()
        return (
            table.search(query_type="hybrid", vector_column_name="vector")
            .vector(query_vector)
            .text(query_text)
            .limit(k)
            .to_list()
        )
