"""Склейка loop.run() + retrieve + synthesize в один вызов.

Используется и `cli.py` (`research`), и `web/app.py` (веб-интерфейс) — чтобы
не дублировать одну и ту же последовательность действий в двух местах.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .. import config
from ..providers import rerank
from ..sources.arxiv import ArxivSource
from ..sources.base import Source
from ..sources.crossref import CrossrefSource
from ..sources.github_mcp import GitHubMCPSource
from ..sources.semantic_scholar import SemanticScholarSource
from ..sources.tavily import TavilySource
from ..sources.web import WebSource
from ..sources.wikipedia import WikipediaSource
from ..store.qdrant_store import QdrantStore, document_to_hit
from . import funnel, loop
from .progress import ProgressCallback
from .state import ResearchState
from .synthesize import synthesize


@dataclass
class SourceRef:
    title: str
    url: str = ""
    citation_count: int | None = None  # None — неизвестно (не -1 сентинел хранилища, см. ниже)


@dataclass
class CandidateSummary:
    """Кандидат, найденный воронкой (не обязательно процитирован в ответе) —
    показывает, КАК агент сузил поиск: все найденные, score триажа, кто
    реально был прочитан. Отдельно от `SourceRef` (только процитированное в
    финальном ответе)."""

    title: str
    source: str
    id: str = ""
    url: str = ""
    citation_count: int | None = None
    triage_score: float | None = None
    read: bool = False


@dataclass
class ResearchResult:
    answer: str
    sources: list[SourceRef]
    candidates: list[CandidateSummary]
    gaps: list[str]
    iterations: int
    read_count: int
    candidates_count: int
    # Не JSON-сериализуем и не идёт в веб-ответ (см. web/app.py::_job_to_dict) —
    # нужен только серверной стороне, чтобы follow-up-вопрос того же диалога
    # (run_followup) мог продолжить именно этот ResearchState, а не начинать
    # с нуля.
    state: ResearchState | None = None
    # Ровно те чанки, что ушли в synthesize() — нумерация цитат [n] в ответе
    # соответствует context[n-1]. Тоже не идёт в веб-ответ: нужен оценке
    # (agent/evaluate.py и evals/), которой без исходного контекста
    # faithfulness посчитать не по чему. `sources` для этого не годится —
    # там дедуплицированные заголовки, а не текст.
    context: list[dict[str, Any]] = field(default_factory=list)


def default_sources() -> list[Source]:
    """Общий веб-источник: Tavily, если настроен `TAVILY_API_KEY` (управляемый
    поиск, без CAPTCHA/rate-limit чужих движков — см. sources/tavily.py),
    иначе локальный SearXNG (sources/web.py) — работает без ключа, но упирается
    в блокировки части движков на стороне их провайдеров (проверено вручную
    2026-08-06). Оба реализуют один и тот же протокол `Source`, funnel.py не
    видит разницы.

    CrossRef и Wikipedia — тоже без ключа, дополняют arXiv/Semantic Scholar
    непрепринтными журнальными публикациями (CrossRef, реальный
    `is-referenced-by-count`) и энциклопедическим/справочным контекстом
    (Wikipedia) для подвопросов, которые чисто научные источники не
    покрывают.

    GitHub (через MCP, sources/github_mcp.py) — опционально
    (config.GITHUB_MCP_ENABLED, по умолчанию выключен + нужен
    GITHUB_PERSONAL_ACCESS_TOKEN) — поиск по репозиториям для подвопросов
    про конкретную библиотеку/инструмент.

    `ArxivSource(categories=config.ARXIV_AI_CATEGORIES)` — ограничивает
    keyword-поиск ИИ/ML-разделами arXiv (§ пользовательский запрос: агент
    должен хорошо работать по научным статьям в области ИИ), иначе
    неоднозначные термины (например, "attention") ловят статьи из физики/
    нейронауки/econ, где то же слово значит другое. См. sources/arxiv.py."""
    web_source: Source = TavilySource() if os.environ.get("TAVILY_API_KEY") else WebSource()
    sources: list[Source] = [
        ArxivSource(categories=config.ARXIV_AI_CATEGORIES),
        SemanticScholarSource(),
        CrossrefSource(),
        WikipediaSource(),
        web_source,
    ]
    if config.GITHUB_MCP_ENABLED and os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN"):
        sources.append(GitHubMCPSource())
    return sources


def retrieve(store: QdrantStore, question: str) -> list[dict[str, Any]]:
    """Retrieval + опциональный реранк поверх уже построенного индекса.

    Общий шаг и для `ask` (индекс из corpus/), и для `research` (индекс,
    пополненный воронкой) — retrieval-механика одна и та же. Идёт через
    LangChain `VectorStoreRetriever` (`store.as_retriever()`) — стандартный
    интерфейс, совместимый с остальным LangChain-кодом, а не
    `search_hybrid()` напрямую; сама гибридная dense+FTS-логика при этом не
    меняется, см. docstring `QdrantStore`.
    """
    candidate_k = config.RERANK_CANDIDATES_K if config.RERANK_ENABLED else config.TOP_K_RETRIEVE
    retriever = store.as_retriever(search_kwargs={"k": candidate_k})
    hits = [document_to_hit(doc) for doc in retriever.invoke(question)]
    if config.RERANK_ENABLED:
        hits = rerank.rerank(question, hits, top_n=config.TOP_K_RETRIEVE)
    return hits


def retrieve_per_subquestion(
    store: QdrantStore, question: str, sub_questions: list[str]
) -> list[dict[str, Any]]:
    """Контекст синтеза, собранный по каждому подвопросу отдельно.

    Зачем. `retrieve()` отбирает TOP_K_RETRIEVE чанков по сходству с вопросом
    ЦЕЛИКОМ. Для вопроса из трёх аспектов пять чанков почти наверняка
    сгруппируются вокруг одного-двух — того, что ближе к полной строке
    запроса. Сколько бы граней воронка ни насобирала, до синтеза доезжает
    один срез: потолок покрытия подтем стоит здесь, а не в планировщике.

    Поэтому retrieval идёт на каждый подвопрос, а результаты сливаются
    ПООЧЕРЁДНО (round-robin), а не встык: если общий cap обрежет хвост, у
    каждой грани всё равно останется представительство. Дедуп по тексту
    чанка — один и тот же фрагмент часто релевантен нескольким граням.

    Один подвопрос — поведение ровно как раньше, лишних вызовов нет.
    """
    unique = [sq for sq in dict.fromkeys(sq.strip() for sq in sub_questions) if sq]
    if len(unique) <= 1:
        return retrieve(store, question)

    per_query = [retrieve(store, sq) for sq in unique]
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rank in range(max(len(h) for h in per_query)):
        for hits in per_query:
            if rank >= len(hits):
                continue
            key = hits[rank].get("text", "")
            if key in seen:
                continue
            seen.add(key)
            merged.append(hits[rank])
            if len(merged) >= config.SYNTHESIS_MAX_CHUNKS:
                return merged
    return merged


def _retrieve_context(store: QdrantStore, question: str, state: ResearchState) -> list[dict[str, Any]]:
    """Контекст для синтеза: по подвопросам, если их больше одного."""
    if not config.RETRIEVE_PER_SUBQUESTION:
        return retrieve(store, question)
    return retrieve_per_subquestion(store, question, [sq.text for sq in state.sub_questions])


def unique_sources(hits: list[dict[str, Any]]) -> list[SourceRef]:
    """Дедуп по заголовку + ссылка/цитируемость для рендера (CLI-текст,
    кликабельная ссылка в веб-интерфейсе). `citation_count == -1` — сентинел
    хранилища "неизвестно" (см. store/qdrant_store.py), превращаем в None —
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


def _candidate_summaries(state) -> list[CandidateSummary]:
    """Все найденные кандидаты, отсортированы по score триажа (лучшие
    сверху) — это и есть видимый след "как агент сузил поиск"."""
    summaries = [
        CandidateSummary(
            id=c.id,
            title=c.title,
            source=c.source,
            url=c.meta.get("url") or "",
            citation_count=c.meta.get("citation_count"),
            triage_score=c.triage_score,
            read=c.id in state.read_ids,
        )
        for c in state.candidates
    ]
    summaries.sort(key=lambda c: (c.triage_score is not None, c.triage_score or 0.0), reverse=True)
    return summaries


def run_research(
    question: str,
    store: QdrantStore,
    on_progress: ProgressCallback | None = None,
) -> ResearchResult:
    state = loop.run(question, default_sources(), store, on_progress=on_progress)
    hits = _retrieve_context(store, question, state)
    answer = synthesize(question, hits, gaps=state.gaps)
    state.add_turn(question, answer)

    return ResearchResult(
        answer=answer,
        sources=unique_sources(hits),
        context=hits,
        candidates=_candidate_summaries(state),
        gaps=state.gaps,
        iterations=state.iterations,
        read_count=len(state.read_ids),
        candidates_count=len(state.candidates),
        state=state,
    )


def run_followup(
    question: str,
    state: ResearchState,
    store: QdrantStore,
    on_progress: ProgressCallback | None = None,
    focus_candidate_id: str | None = None,
) -> ResearchResult:
    """Уточняющий вопрос / "раскрой подробнее эту тему" в том же диалоге —
    продолжает уже накопленный `state` (см. docstring `agent/state.py`)
    вместо `run_research()`'а с нуля.

    `focus_candidate_id` — id кандидата из `result.candidates` предыдущего
    хода ("раскрыть подробнее"): форсирует его deep-read (если ещё не
    прочитан) до того, как обычный follow-up-проход воронки пойдёт искать
    что-то ещё по этой теме."""
    if focus_candidate_id is not None:
        candidate = next((c for c in state.candidates if c.id == focus_candidate_id), None)
        if candidate is not None and not state.is_read(candidate.id):
            funnel.deep_read_candidate(candidate, question, state, store, on_progress=on_progress)

    loop.run(question, default_sources(), store, on_progress=on_progress, state=state)
    hits = _retrieve_context(store, question, state)
    answer = synthesize(question, hits, gaps=state.gaps, history=state.history)
    state.add_turn(question, answer)

    return ResearchResult(
        answer=answer,
        sources=unique_sources(hits),
        context=hits,
        candidates=_candidate_summaries(state),
        gaps=state.gaps,
        iterations=state.iterations,
        read_count=len(state.read_ids),
        candidates_count=len(state.candidates),
        state=state,
    )
