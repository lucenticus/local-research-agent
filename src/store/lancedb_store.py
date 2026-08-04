"""LanceDB-хранилище чанков: on-disk индекс + dense vector search.

Milestone 0: полный rebuild таблицы, без FTS/hybrid (см. Milestone 1) и без
инкрементального upsert (см. Milestone 3 — воронка дополняет индекс на лету).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lancedb

from .. import config


@dataclass
class Chunk:
    id: str
    text: str
    source_id: str
    source_title: str
    vector: list[float]


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
                "vector": c.vector,
            }
            for c in chunks
        ]
        self._table = db.create_table(self._table_name, data=rows, mode="overwrite")

    def _ensure_table(self) -> Any:
        if self._table is None:
            db = self._connect()
            if self._table_name not in db.list_tables().tables:
                raise RuntimeError(
                    "Индекс пуст — сначала выполни `python -m src.cli index`"
                )
            self._table = db.open_table(self._table_name)
        return self._table

    def search(self, query_vector: list[float], k: int) -> list[dict[str, Any]]:
        table = self._ensure_table()
        return table.search(query_vector).limit(k).to_list()
