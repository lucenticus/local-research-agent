"""Qdrant-хранилище чанков: dense + sparse (learned lexical weights от
bge-m3) гибридный поиск через RRF-fusion в Qdrant Query API.

Docker-сервер (`docker-compose.yml`), не embedded — в отличие от прежнего
LanceDB-хранилища этого проекта. Осознанный выбор пользователя: полноценный
Qdrant (продовый API, отдельный процесс) вместо embedded-режима
(`QdrantClient(path=...)`), который был бы ближе к прежней "нигде ничего не
поднимать" философии, но урезан в части фич. Поднять перед использованием:

    docker compose up -d qdrant

Гибридный поиск здесь устроен иначе, чем в LanceDB: там лексический сигнал
шёл от FTS (tantivy BM25) прямо по сырому тексту, без эмбеддинга запроса.
Qdrant так не умеет — вместо BM25-по-корпусу используется sparse-выход
bge-m3 (learned lexical weights, тот же forward pass, что и dense, см.
`providers/embed.py`). Прогонов на реальном корпусе, показывающих, какой из
двух лексических сигналов лучше на этом масштабе, не делалось — принято как
следствие смены хранилища, не отдельное исследование.

`QdrantStore` реализует `langchain_core.vectorstores.VectorStore`
(`similarity_search`/`add_texts`/`from_texts`/`embeddings`) — тот же
аддитивный слой для интеропа с LangChain (`.as_retriever()`, см.
`agent/research_runner.retrieve()`), что был у `LanceDBStore`.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from qdrant_client import QdrantClient, models

from .. import config
from ..providers import embed

_DENSE_VECTOR_NAME = "dense"
_SPARSE_VECTOR_NAME = "sparse"
# Кандидатов на каждую сторону (dense/sparse) перед RRF-слиянием — шире, чем
# финальный k, иначе fusion нечего сливать (тот же принцип, что и
# config.RERANK_CANDIDATES_K для реранкера поверх retrieval).
_PREFETCH_MULTIPLIER = 5
# Размер страницы при scroll'е digest-пула (`load_pool`) — пул за неделю это
# тысячи точек, тянуть их одним запросом незачем.
_SCROLL_PAGE = 1000
# Точек на один upsert. 2143 чанка одним запросом — это 53МБ, а у Qdrant
# лимит тела 32МБ (см. `_upsert_batched`); 256 даёт запросы в единицы МБ.
_UPSERT_BATCH = 256


@dataclass
class Chunk:
    id: str
    text: str
    source_id: str
    source_title: str
    section: str
    vector: list[float]
    # "" / -1 — сентинелы "неизвестно" (тот же контракт, что был у
    # LanceDBStore — Qdrant-payload схемы не требует, но менять контракт
    # заодно с хранилищем не стали, чтобы не трогать вызывающий код).
    url: str = ""
    citation_count: int = -1
    # Заполняется `embed.embed_texts_hybrid()` при индексации (см.
    # cli.py/funnel.py) — пустой sparse-вектор означает "sparse не считался",
    # не "текст пустой".
    sparse: dict[int, float] = field(default_factory=dict)
    # Метаданные публикации. Нужны digest-пулу (agent/digest.py), чтобы
    # восстанавливать окно «за последние N дней» прямо из индекса и не
    # выкачивать те же статьи из arXiv на каждый запрос; остальные
    # потребители (cli index, funnel) их не заполняют — пустые значения
    # означают "неизвестно", и такие точки просто не попадают в digest-пул.
    published_date: str = ""  # ISO, как отдал источник — для отображения
    authors: list[str] = field(default_factory=list)


def chunk_id_for(source_id: str, section: str, text: str) -> str:
    """Детерминированный id чанка: одно и то же содержимое -> тот же id.

    Раньше здесь стоял `uuid.uuid4()`, и это делало прогон невоспроизводимым:
    точки Qdrant получали новые идентификаторы каждый раз, а с ними менялся
    порядок выдачи среди чанков с близкими скорами. Поймано на замере — два
    прогона с полностью замороженными источниками, временем и temperature=0
    всё равно расходились по отдельным вопросам.

    Побочно даёт идемпотентную индексацию: повторный upsert того же чанка
    перезаписывает точку, а не плодит дубликат.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_OID, f"{source_id}|{section}|{text}"))


def _point_id(chunk_id: str) -> str:
    """Qdrant принимает id точки только как UUID или unsigned int — `Chunk.id`
    уже везде в проекте генерируется как `str(uuid.uuid4())`, так что обычно
    это no-op; на случай другого формата приводим к валидному через uuid5,
    детерминированно от исходного id (повторный upsert того же chunk.id даёт
    ту же точку, не дубликат)."""
    try:
        return str(uuid.UUID(chunk_id))
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_OID, chunk_id))


def published_timestamp(published_date: str) -> float | None:
    """ISO-дата -> unix-время. `None`, если даты нет или она не парсится.

    Qdrant умеет range-фильтр и order_by только по числовому полю, а
    сортировать/резать ISO-строки лексикографически некорректно (разные
    смещения таймзон), поэтому рядом с `published_date` в payload кладётся
    числовой `published_ts` — по нему и идёт отбор окна в `load_pool`.
    """
    if not published_date:
        return None
    try:
        return datetime.fromisoformat(published_date.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _payload(chunk: Chunk) -> dict[str, Any]:
    payload = {
        "chunk_id": chunk.id,
        "text": chunk.text,
        "source_id": chunk.source_id,
        "source_title": chunk.source_title,
        "section": chunk.section,
        "url": chunk.url,
        "citation_count": chunk.citation_count,
        "published_date": chunk.published_date,
        "authors": list(chunk.authors),
    }
    published_ts = published_timestamp(chunk.published_date)
    if published_ts is not None:
        payload["published_ts"] = published_ts
    return payload


def _point(chunk: Chunk) -> models.PointStruct:
    vector: dict[str, Any] = {_DENSE_VECTOR_NAME: chunk.vector}
    if chunk.sparse:
        vector[_SPARSE_VECTOR_NAME] = models.SparseVector(
            indices=list(chunk.sparse.keys()), values=list(chunk.sparse.values())
        )
    return models.PointStruct(id=_point_id(chunk.id), vector=vector, payload=_payload(chunk))


class QdrantStore(VectorStore):
    def __init__(
        self,
        url: str | None = None,
        collection_name: str | None = None,
        embedding: Embeddings | None = None,
        *,
        path: str | None = None,
    ):
        """`url` — Docker-сервер (по умолчанию, `config.QDRANT_URL`), реальный
        путь этого проекта. `path` — embedded режим на локальный файл
        (persist между реконнектами, в отличие от `location=":memory:"`) —
        только для офлайн-тестов (tests/test_qdrant_store.py,
        tests/test_cli_smoke.py), не используется в продовом коде: не тот же
        движок под капотом, что Docker-сервер, поэтому не альтернативный
        режим работы агента, а именно тестовый двойник."""
        self._url = url or config.QDRANT_URL
        self._path = path
        self._collection_name = collection_name or config.QDRANT_COLLECTION
        self._client: QdrantClient | None = None
        # Ленивый дефолт (не грузит модель здесь, см. providers/embed.py) —
        # большинство конструкторов QdrantStore по кодовой базе не передают
        # embedding вовсе, им нужны только rebuild/add_chunks/search_hybrid.
        self._embedding = embedding

    @property
    def embeddings(self) -> Embeddings | None:
        return self._embedding

    def _connect(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(path=self._path) if self._path else QdrantClient(url=self._url)
        return self._client

    def _create_collection(self, dim: int) -> None:
        """`dim` — размерность dense-вектора, берётся из первого чанка,
        который реально индексируется (см. `rebuild`/`add_chunks`), а не из
        конфига: раньше (LanceDB) схема таблицы выводилась из первого batch
        данных, Qdrant же требует размер заранее при create_collection —
        так `QdrantStore` не привязан к конкретной модели эмбеддингов
        (bge-m3, 1024) намертво, и тесты на игрушечных векторах маленькой
        размерности работают без подмены конфига."""
        client = self._connect()
        client.create_collection(
            self._collection_name,
            vectors_config={_DENSE_VECTOR_NAME: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={_SPARSE_VECTOR_NAME: models.SparseVectorParams()},
        )
        # Без индекса по source_id filtered count() (has_source) реально
        # ловился на выдаче неверных результатов (найдено вручную: approximate
        # count по неиндексированному полю на маленькой коллекции считал в
        # обе стороны неправильно — и false positive, и false negative).
        # keyword-индекс делает filtered count точным и быстрым при любом
        # размере коллекции, не только "работает случайно на малых данных".
        client.create_payload_index(
            self._collection_name, field_name="source_id", field_schema=models.PayloadSchemaType.KEYWORD
        )
        # Индекс под `load_pool`/`oldest_published_ts` (digest-пул): без него
        # range-фильтр и order_by по published_ts либо не работают, либо
        # деградируют на растущей коллекции.
        client.create_payload_index(
            self._collection_name, field_name="published_ts", field_schema=models.PayloadSchemaType.FLOAT
        )

    def rebuild(self, chunks: list[Chunk]) -> None:
        """Полная пересборка коллекции из chunks."""
        client = self._connect()
        if client.collection_exists(self._collection_name):
            client.delete_collection(self._collection_name)
        if not chunks:
            return
        self._create_collection(dim=len(chunks[0].vector))
        self._upsert_batched(chunks)

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Инкрементально дописывает chunks (создаёт коллекцию, если её ещё нет).

        Используется воронкой (agent/funnel.py) — прочитанное пополняет индекс
        на лету, без полной пересборки (в отличие от `rebuild`, который зовёт
        только `cli.py index` для локального корпуса).
        """
        if not chunks:
            return
        client = self._connect()
        if not client.collection_exists(self._collection_name):
            self._create_collection(dim=len(chunks[0].vector))
        self._upsert_batched(chunks)

    def _upsert_batched(self, chunks: list[Chunk]) -> None:
        """Заливка пачками, а не одним запросом.

        У Qdrant есть лимит на размер тела запроса (32МБ по умолчанию), и
        digest-пул за неделю в него не влезает: 2143 чанка с 1024-мерным
        dense-вектором, sparse и текстом дали 53МБ и 400 Bad Request
        (поймано реальным прогоном, не гипотетически).
        """
        client = self._connect()
        for start in range(0, len(chunks), _UPSERT_BATCH):
            batch = chunks[start : start + _UPSERT_BATCH]
            client.upsert(self._collection_name, points=[_point(c) for c in batch])

    def has_source(self, source_id: str) -> bool:
        """Есть ли уже чанки с этим source_id — кэш-хит для funnel (не читать повторно)."""
        client = self._connect()
        if not client.collection_exists(self._collection_name):
            return False
        result = client.count(
            self._collection_name,
            count_filter=models.Filter(
                must=[models.FieldCondition(key="source_id", match=models.MatchValue(value=source_id))]
            ),
            exact=True,
        )
        return result.count > 0

    def load_pool(self, min_published_ts: float) -> list[dict[str, Any]]:
        """Payload'ы всех точек с `published_ts >= min_published_ts`.

        Digest-пул (agent/digest.py) восстанавливает по ним окно «за
        последние N дней» прямо из индекса, вместо повторной выгрузки тех же
        статей из arXiv на каждый запрос. Точки без `published_ts` (всё, что
        индексировали `cli index`/funnel) под фильтр не попадают — пул строго
        из того, у чего известна дата публикации.
        """
        client = self._connect()
        if not client.collection_exists(self._collection_name):
            return []
        payloads: list[dict[str, Any]] = []
        offset = None
        while True:
            points, offset = client.scroll(
                self._collection_name,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(key="published_ts", range=models.Range(gte=min_published_ts))]
                ),
                limit=_SCROLL_PAGE,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            payloads.extend(point.payload for point in points)
            if offset is None:
                break
        return payloads

    def oldest_published_ts(self) -> float | None:
        """Самая ранняя `published_ts` в коллекции (None — таких точек нет).

        Digest использует это, чтобы понять, покрывает ли кэш всё запрошенное
        окно: если самая старая известная статья старше начала окна, то
        нижняя граница окна уже в индексе и выгрузку из arXiv можно
        останавливать досрочно, как только пошли уже известные статьи.
        """
        client = self._connect()
        if not client.collection_exists(self._collection_name):
            return None
        points, _ = client.scroll(
            self._collection_name,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="published_ts", range=models.Range(gte=0))]
            ),
            limit=1,
            with_payload=["published_ts"],
            with_vectors=False,
            order_by=models.OrderBy(key="published_ts", direction=models.Direction.ASC),
        )
        return points[0].payload.get("published_ts") if points else None

    def _ensure_collection(self) -> str:
        client = self._connect()
        if not client.collection_exists(self._collection_name):
            raise RuntimeError("Индекс пуст — сначала выполни `python -m src.cli index`")
        return self._collection_name

    def search_dense(self, query_vector: list[float], k: int) -> list[dict[str, Any]]:
        """Dense-only поиск — используется для сравнения в eval."""
        client = self._connect()
        collection = self._ensure_collection()
        result = client.query_points(
            collection, query=query_vector, using=_DENSE_VECTOR_NAME, limit=k, with_payload=True
        )
        return [point.payload for point in result.points]

    def search_hybrid(self, query_text: str, query_vector: list[float], k: int) -> list[dict[str, Any]]:
        """Dense + sparse, слияние через RRF (Qdrant Query API prefetch+fusion).

        Sparse-вектор запроса считается здесь же, из `query_text` — единственное
        место в проекте, которое явно знает про sparse-эмбеддинги; вызывающий
        код (agent/loop.py, agent/funnel.py, cli.py) передаёт `query_text` +
        уже готовый dense `query_vector`, как и раньше для LanceDB, и понятия
        не имеет о sparse.
        """
        client = self._connect()
        collection = self._ensure_collection()
        sparse = embed.embed_sparse([query_text])[0]
        prefetch_limit = max(k * _PREFETCH_MULTIPLIER, config.TOP_K_RETRIEVE)
        prefetch = [models.Prefetch(query=query_vector, using=_DENSE_VECTOR_NAME, limit=prefetch_limit)]
        if sparse:
            prefetch.append(
                models.Prefetch(
                    query=models.SparseVector(indices=list(sparse.keys()), values=list(sparse.values())),
                    using=_SPARSE_VECTOR_NAME,
                    limit=prefetch_limit,
                )
            )
        result = client.query_points(
            collection,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=k,
            with_payload=True,
        )
        return [point.payload for point in result.points]

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
        sparse_vectors = embed.embed_sparse(texts)
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
                sparse=sparse,
            )
            for id_, text, meta, vector, sparse in zip(
                ids, texts, metadatas, vectors, sparse_vectors, strict=True
            )
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
        url: str | None = None,
        collection_name: str | None = None,
        path: str | None = None,
        **kwargs: Any,
    ) -> "QdrantStore":
        store = cls(url=url, collection_name=collection_name, embedding=embedding, path=path)
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
