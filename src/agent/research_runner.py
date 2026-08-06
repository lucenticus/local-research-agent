"""Склейка loop.run() + retrieve + synthesize в один вызов.

Используется и `cli.py` (`research`), и `web/app.py` (веб-интерфейс) — чтобы
не дублировать одну и ту же последовательность действий в двух местах.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .. import config
from ..providers import embed, rerank
from ..sources.arxiv import ArxivSource
from ..sources.base import Source
from ..sources.semantic_scholar import SemanticScholarSource
from ..sources.web import WebSource
from ..store.lancedb_store import LanceDBStore
from . import loop
from .progress import ProgressCallback
from .synthesize import synthesize


@dataclass
class SourceRef:
    title: str
    url: str = ""
    citation_count: int | None = None  # None — неизвестно (не -1 сентинел LanceDB, см. ниже)


@dataclass
class ResearchResult:
    answer: str
    sources: list[SourceRef]
    gaps: list[str]
    iterations: int
    read_count: int
    candidates_count: int


def default_sources() -> list[Source]:
    return [ArxivSource(), SemanticScholarSource(), WebSource()]


def retrieve(store: LanceDBStore, question: str) -> list[dict[str, Any]]:
    """Retrieval + опциональный реранк поверх уже построенного индекса.

    Общий шаг и для `ask` (индекс из corpus/), и для `research` (индекс,
    пополненный воронкой) — retrieval-механика одна и та же.
    """
    query_vector = embed.embed_texts([question])[0]
    candidate_k = config.RERANK_CANDIDATES_K if config.RERANK_ENABLED else config.TOP_K_RETRIEVE
    hits = store.search_hybrid(question, query_vector, k=candidate_k)
    if config.RERANK_ENABLED:
        hits = rerank.rerank(question, hits, top_n=config.TOP_K_RETRIEVE)
    return hits


def _unique_sources(hits: list[dict[str, Any]]) -> list[SourceRef]:
    """Дедуп по заголовку + ссылка/цитируемость для рендера (CLI-текст,
    кликабельная ссылка в веб-интерфейсе). `citation_count == -1` — сентинел
    LanceDB "неизвестно" (см. store/lancedb_store.py), превращаем в None —
    вызывающему коду (UI) не нужно знать про внутренний сентинел хранилища.
    """
    seen: set[str] = set()
    refs: list[SourceRef] = []
    for hit in hits:
        title = hit.get("source_title") or hit.get("source_id") or "?"
        if title in seen:
            continue
        seen.add(title)
        citation_count = hit.get("citation_count")
        refs.append(
            SourceRef(
                title=title,
                url=hit.get("url") or "",
                citation_count=(
                    citation_count
                    if citation_count is not None and citation_count >= 0
                    else None
                ),
            )
        )
    return refs


def run_research(
    question: str,
    store: LanceDBStore,
    on_progress: ProgressCallback | None = None,
) -> ResearchResult:
    state = loop.run(question, default_sources(), store, on_progress=on_progress)
    hits = retrieve(store, question)
    answer = synthesize(question, hits, gaps=state.gaps)

    return ResearchResult(
        answer=answer,
        sources=_unique_sources(hits),
        gaps=state.gaps,
        iterations=state.iterations,
        read_count=len(state.read_ids),
        candidates_count=len(state.candidates),
    )
