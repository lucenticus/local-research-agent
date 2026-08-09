"""LanceDB-хранилище чанков: on-disk индекс, dense vector search + hybrid.

Milestone 1: добавлен FTS-индекс по тексту и гибридный поиск (dense + FTS,
слияние через RRF — реранкер LanceDB по умолчанию, проверено на этой версии
lancedb 0.36.0). Полный rebuild таблицы на каждой индексации, без
инкрементального upsert (см. Milestone 3 — воронка дополняет индекс на лету).

Отдельный sparse-выход bge-m3 (lexical weights) не используется: LanceDB FTS
со стеммером `language="Russian"` уже даёт лексический сигнал на кириллице
(проверено вручную на этой машине) — использовать оба было бы дублирующим
сигналом без явной пользы на масштабе этого проекта.

`LanceDBStore` реализует `langchain_core.vectorstores.VectorStore`
(`similarity_search`/`add_texts`/`from_texts`/`embeddings`) поверх тех же
`rebuild`/`add_chunks`/`search_hybrid` — это чисто аддитивный слой для
интеропа со стандартным LangChain-кодом (`.as_retriever()`, см.
`agent/research_runner.retrieve()`), исходные методы и их поведение (в т.ч.
url/citation_count-сентинелы, инкрементальный `add_chunks`) не меняются ни
на бит — на них по-прежнему завязаны `agent/funnel.py`/`agent/loop.py` и
существующие тесты."""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lancedb
from lancedb.index import FTS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

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


def _rows(chunks: list[Chunk]) -> list[dict[str, Any]]:
    return [
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


class LanceDBStore(VectorStore):
    def __init__(
        self,
        db_path: Path | None = None,
        table_name: str | None = None,
        embedding: Embeddings | None = None,
    ):
        self._db_path = db_path or config.INDEX_DIR
        self._table_name = table_name or config.INDEX_TABLE
        self._db: Any = None
        self._table: Any = None
        # Ленивый дефолт (не грузит модель здесь, см. providers/embed.py) —
        # большинство конструкторов LanceDBStore по кодовой базе не передают
        # embedding вовсе, им нужны только rebuild/add_chunks/search_hybrid.
        self._embedding = embedding

    @property
    def embeddings(self) -> Embeddings | None:
        return self._embedding

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
        self._table = db.create_table(self._table_name, data=_rows(chunks), mode="overwrite")
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
        rows = _rows(chunks)
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

    # --- langchain_core.vectorstores.VectorStore ---

    def _default_embedding(self) -> Embeddings:
        if self._embedding is None:
            from ..providers.langchain_embeddings import MLXBGEEmbeddings

            self._embedding = MLXBGEEmbeddings()
        return self._embedding

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        """`.as_retriever()`-совместимый поиск — под капотом тот же
        `search_hybrid`, эмбеддинг запроса берётся из `self.embeddings`
        (дефолт — `MLXBGEEmbeddings`, та же резидентная bge-m3, что и везде
        в проекте, не отдельная модель)."""
        query_vector = self._default_embedding().embed_query(query)
        hits = self.search_hybrid(query, query_vector, k=k)
        return [hit_to_document(hit) for hit in hits]

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        texts = list(texts)
        metadatas = metadatas or [{} for _ in texts]
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        vectors = self._default_embedding().embed_documents(texts)
        chunks = [
            Chunk(
                id=id_,
                text=text,
                source_id=meta.get("source_id", ""),
                source_title=meta.get("source_title", ""),
                section=meta.get("section", ""),
                vector=vector,
                url=meta.get("url") or "",
                citation_count=meta["citation_count"] if meta.get("citation_count") is not None else -1,
            )
            for id_, text, meta, vector in zip(ids, texts, metadatas, vectors, strict=True)
        ]
        self.add_chunks(chunks)
        return ids

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        db_path: Path | None = None,
        table_name: str | None = None,
        **kwargs: Any,
    ) -> "LanceDBStore":
        store = cls(db_path=db_path, table_name=table_name, embedding=embedding)
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store


def hit_to_document(hit: dict[str, Any]) -> Document:
    """`search_hybrid`/`search_dense`-хит -> LangChain `Document` (текст +
    остальные поля как metadata) — используется `similarity_search` и
    `agent/research_runner.retrieve()` (через `.as_retriever()`)."""
    metadata = {k: v for k, v in hit.items() if k != "text"}
    return Document(page_content=hit.get("text", ""), metadata=metadata)


def document_to_hit(document: Document) -> dict[str, Any]:
    """Обратное преобразование — восстанавливает исходную форму хита
    (`{"text": ..., "source_id": ..., ...}`), которую ожидают
    `synthesize.py`/`evaluate.py`/`cli.py`."""
    return {"text": document.page_content, **document.metadata}
