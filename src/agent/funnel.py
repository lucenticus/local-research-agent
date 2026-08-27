"""Прогрессивная воронка: discovery -> триаж -> deep read.

Discovery дёшево (только метаданные, все источники сразу). Триаж скорит
abstract кандидатов против подвопроса эмбеддингом и оставляет top-N — большая
часть кандидатов умирает здесь, полный текст не качается. Deep read — только
для выживших и ещё не прочитанных (`state.read_ids`): для arXiv скачивается
PDF и извлекается секция-осознанно; для остальных источников (без открытого
PDF) — полный текст страницы через MCP fetch-сервер
(`config.MCP_FETCH_ENABLED`, по умолчанию выключен — отдельный процесс
uvx/npx, см. README), а если и это недоступно/выключено — сам abstract как
единственный chunk, честный fallback, а не притворство, что full-text
недоступен.

Найдено реальным прогоном 2026-08-05: arXiv/Semantic Scholar — англоязычные
корпуса, подвопрос на русском (а агент по умолчанию русскоязычный) даёт 0
результатов дословным запросом. Добавлен bounded LLM-перевод подвопроса в
короткий английский поисковый запрос перед discovery — это расширение
"опц. bounded LLM" из плана (там разрешён для gap-check) на реальную
необходимость, без которой источники не работают для целевого пользователя.

Триаж учитывает цитируемость (§ пользовательский запрос 2026-08-06):
Semantic Scholar отдаёт `citationCount` прямо при discovery, у arXiv своей
цитируемости нет — обогащаем через `sources/citations.py` (OpenAlex, best
effort). Итоговый score триажа = косинус (семантическая релевантность
подвопросу) + небольшой логарифмический буст по цитируемости — см.
`_combined_score` и `CITATION_BOOST_SCALE` в config.py: буст — тайбрейкер
между близкими по смыслу кандидатами, а не замена семантике (иначе
популярная, но нерелевантная статья обходила бы точный, но малоцитируемый
ответ).

Триаж также учитывает свежесть (§ пользовательский запрос: агент должен
хорошо следить за свежими статьями в области ИИ) — экспоненциально
затухающий буст по `published_date` (сейчас его отдаёт только `arxiv.py`),
независимый от цитируемости: свежая статья структурно не может успеть
набрать цитирований, так что чистый citation-буст систематически топит
именно то, что нужнее всего для "что нового вышло". См. `_recency_boost` и
`RECENCY_BOOST_SCALE`/`RECENCY_HALF_LIFE_DAYS` в config.py.

Кросс-источниковый дедуп по arXiv id (найдено реальным прогоном 2026-08-06):
одна и та же статья находится и через `arxiv.py` (id вида `arxiv:XXXXvN`), и
через общий веб-поиск (id вида `web:<url>` — например, ссылка на HTML-версию
той же статьи на arxiv.org). Разные id -> `state.add_candidates` не видит,
что это один и тот же кандидат, и статья дублируется в списке источников.
`_canonical_candidate_id` нормализует любой arXiv-подобный id/URL к номеру
статьи без версии — все варианты (abs/html/pdf, разные версии, найдено через
любой источник) схлопываются в один кандидат.
"""

from __future__ import annotations

import math
import re
import uuid
from datetime import datetime, timezone
from typing import Any
from .. import config
from ..ingest.chunk import chunk_sections, chunk_text
from ..ingest.extract import Section
from ..providers import embed, llm
from ..providers.mcp_client import content_to_text, get_single_tool
from ..sources.base import DiscoveredItem, Source
from ..sources.citations import lookup_citation_count
from ..sources.langchain_tools import make_discover_tool
from ..sources.pdf import fetch_pdf_sections
from ..store.qdrant_store import Chunk, QdrantStore, chunk_id_for
from .progress import ProgressCallback, emit as _emit
from .state import Candidate, Finding, ResearchState, SubQuestion

_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(?:v\d+)?")


def _canonical_candidate_id(item: DiscoveredItem) -> str:
    """arXiv-статьи — единый id независимо от источника обнаружения и URL-варианта."""
    if item.source == "arxiv":
        haystack = item.id
    elif item.source == "semantic_scholar":
        haystack = (item.meta.get("external_ids") or {}).get("ArXiv", "") or (item.url or "")
    else:
        haystack = item.url or ""
    match = _ARXIV_ID_RE.search(haystack)
    return f"arxiv:{match.group(1)}" if match else item.id

_TRANSLATE_SYSTEM_PROMPT = (
    "Translate the user's question into a short English web-search query "
    "(3-8 keywords, no punctuation, no explanations, no quotes). "
    "Output ONLY the query, nothing else."
)


def _looks_non_english(text: str) -> bool:
    cyrillic = sum(1 for ch in text if "а" <= ch.lower() <= "я" or ch.lower() == "ё")
    return cyrillic > len(text) * 0.3


def _discovery_query(text: str) -> str:
    """Английский поисковый запрос для discovery — переводим, только если
    подвопрос явно не на английском (bounded LLM-вызов, см. docstring модуля)."""
    if not _looks_non_english(text):
        return text
    prompt = llm.build_chat_prompt(_TRANSLATE_SYSTEM_PROMPT, text)
    translated = llm.generate(prompt, max_tokens=32).strip()
    return translated or text


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _to_candidate(item: DiscoveredItem) -> Candidate:
    citation_count = item.citation_count
    if citation_count is None and item.source == "arxiv":
        # arXiv не отдаёт цитируемость сам — обогащаем через OpenAlex
        # (best effort: не найдено/недоступно -> остаётся None).
        citation_count = lookup_citation_count(item.title)
    return Candidate(
        id=_canonical_candidate_id(item),
        source=item.source,
        title=item.title,
        abstract=item.abstract,
        meta={
            **item.meta,
            "url": item.url,
            "year": item.year,
            "citation_count": citation_count,
            "published_date": item.published_date,
        },
    )


def _discover(
    sub_question: SubQuestion, sources: list[Source], discovery_limit: int
) -> list[Candidate]:
    query = _discovery_query(sub_question.text)
    candidates: list[Candidate] = []
    for source in sources:
        try:
            # Источник вызывается через LangChain tool-интерфейс
            # (`sources/langchain_tools.py`), а не напрямую `.discover()` —
            # сама логика discovery не меняется, только точка вызова.
            tool = make_discover_tool(source)
            items: list[DiscoveredItem] = tool.invoke({"query": query, "limit": discovery_limit})
        except Exception:
            # Внешний источник недоступен/троттлит — воронка продолжает с тем,
            # что нашли остальные источники, а не падает целиком.
            continue
        candidates.extend(_to_candidate(item) for item in items)
    return candidates


def _now() -> datetime:
    """Текущее время отдельной функцией, чтобы его можно было заморозить.

    Скор триажа зависит от «сейчас» (буст по свежести ниже), то есть один и
    тот же прогон в разные дни ранжирует кандидатов по-разному. Для
    воспроизводимого замера время — такой же внешний вход, как сеть, и
    подменяться должно так же (см. evals/fixtures.py).
    """
    return datetime.now(timezone.utc)


def _recency_boost(published_date: str | None) -> float:
    """Экспоненциально затухающий буст по свежести — 0, если дата неизвестна
    (не arXiv-источник) или не парсится. См. docstring модуля/config.py."""
    if not published_date:
        return 0.0
    try:
        published = datetime.fromisoformat(published_date.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    age_days = (_now() - published).total_seconds() / 86400
    age_days = max(age_days, 0.0)
    return config.RECENCY_BOOST_SCALE * math.exp(-age_days / config.RECENCY_HALF_LIFE_DAYS)


def _combined_score(
    cosine_similarity: float, citation_count: int | None, published_date: str | None
) -> float:
    """Семантическая релевантность + буст по цитируемости + буст по свежести.

    log1p для цитируемости, не сырое число — иначе статья с 10000 цитирований
    задавила бы любую семантику. Оба буста откалиброваны так, чтобы быть
    тайбрейкерами (доли от типичного разброса косинуса), а не доминирующим
    фактором — см. docstring модуля. Независимы друг от друга и суммируются:
    свежая статья без цитирований и старая цитируемая статья бустятся каждая
    по своей оси, не конкурируя за один и тот же "бюджет" буста.
    """
    score = cosine_similarity
    if citation_count and citation_count > 0:
        score += config.CITATION_BOOST_SCALE * math.log1p(citation_count)
    score += _recency_boost(published_date)
    return score


def _triage(sub_question: SubQuestion, candidates: list[Candidate]) -> list[Candidate]:
    scoreable = [c for c in candidates if c.abstract.strip()]
    if not scoreable:
        return []
    texts = [sub_question.text] + [c.abstract for c in scoreable]
    vectors = embed.embed_texts(texts)
    query_vec, candidate_vecs = vectors[0], vectors[1:]
    for candidate, vec in zip(scoreable, candidate_vecs, strict=True):
        cosine_similarity = _cosine(query_vec, vec)
        candidate.triage_score = _combined_score(
            cosine_similarity, candidate.meta.get("citation_count"), candidate.meta.get("published_date")
        )
    scoreable.sort(key=lambda c: c.triage_score or 0.0, reverse=True)
    return scoreable[: config.FUNNEL_TRIAGE_TOP_N]


def discover_candidates(topic: str, sources: list[Source], top_n: int = 5) -> list[Candidate]:
    """Discovery + triage over a single topic, standalone — no sub-question
    loop, no deep read, no synthesis. For a caller that just wants a ranked
    shortlist of candidate papers to pick from by hand (mcp_server.py's
    `find_papers` tool, used by repro-lab's discovery node over MCP), not a
    full research answer. Reuses the exact `_discover`/`_triage` stages
    `loop.run()` already uses — no separate implementation to keep in sync.
    """
    sub_question = SubQuestion(text=topic)
    candidates = _discover(sub_question, sources, config.FUNNEL_DISCOVERY_LIMIT_PER_SOURCE)
    return _triage(sub_question, candidates)[:top_n]


# Кэшируется на модуль (не на вызов) — get_mcp_tools() делает отдельный
# connect/list_tools/disconnect round-trip (см. providers/mcp_client.py),
# незачем платить его на каждый прочитанный кандидат. Сентинел "unset"
# отличает "ещё не пробовали" от "пробовали, сервер недоступен" — во втором
# случае не долбим uvx/сеть повторно на каждом кандидате воронки.
_mcp_fetch_tool: Any = "unset"


def _get_mcp_fetch_tool() -> Any:
    global _mcp_fetch_tool
    if _mcp_fetch_tool == "unset":
        connections = {
            "fetch": {
                "transport": "stdio",
                "command": config.MCP_FETCH_COMMAND,
                "args": config.MCP_FETCH_ARGS,
            }
        }
        _mcp_fetch_tool = get_single_tool(connections, "fetch")
    return _mcp_fetch_tool


def _fetch_page_text_via_mcp(url: str) -> str | None:
    """Full page text via the MCP fetch server (mcp-server-fetch) — richer
    than the abstract-only fallback for non-PDF sources (web/Tavily
    results). Best-effort like `fetch_pdf_sections`: any failure (server
    not installed, network, timeout) just returns None."""
    tool = _get_mcp_fetch_tool()
    if tool is None:
        return None
    try:
        result = tool.invoke({"url": url, "max_length": config.MCP_FETCH_MAX_CHARS})
        return content_to_text(result).strip() or None
    except Exception:
        return None


def _deep_read_sections(candidate: Candidate) -> list[Section]:
    pdf_url = candidate.meta.get("pdf_url")
    if pdf_url:
        try:
            sections = fetch_pdf_sections(pdf_url)
            if sections:
                return sections
        except Exception:
            pass  # источник недоступен -> fallback ниже

    url = candidate.meta.get("url")
    if url and config.MCP_FETCH_ENABLED:
        text = _fetch_page_text_via_mcp(url)
        if text:
            return [Section(name=candidate.title, category="body", text=text)]

    # Нет полного текста (Semantic Scholar/web без PDF, MCP fetch выключен/
    # недоступен, или скачивание не удалось) — честно используем сам
    # abstract как единственный chunk, а не притворяемся, что deep read
    # сделан на полном тексте.
    return [Section(name=candidate.title, category="abstract", text=candidate.abstract)]


def deep_read_candidate(
    candidate: Candidate,
    sub_question_text: str,
    state: ResearchState,
    store: QdrantStore,
    on_progress: ProgressCallback | None = None,
) -> None:
    """Deep-read одного кандидата (внутренний шаг `run()`, вынесен отдельной
    функцией, чтобы follow-up "раскрой подробнее эту тему"
    (`research_runner.run_followup`) мог форсировать чтение конкретного уже
    найденного кандидата напрямую, без повторного discovery/триажа."""
    if state.is_read(candidate.id):
        return

    already_indexed = store.has_source(candidate.id)
    if not already_indexed:
        _emit(on_progress, f"Reading: {candidate.title[:80]}…")
        sections = _deep_read_sections(candidate)
        raw_chunks = chunk_sections(sections) or chunk_text(candidate.abstract)
        if raw_chunks:
            vectors, sparse_vectors = embed.embed_texts_hybrid([c.text for c in raw_chunks])
            citation_count = candidate.meta.get("citation_count")
            chunks = [
                Chunk(
                    id=chunk_id_for(candidate.id, raw.section, raw.text),
                    text=raw.text,
                    source_id=candidate.id,
                    source_title=candidate.title,
                    section=raw.section,
                    vector=vector,
                    sparse=sparse,
                    url=candidate.meta.get("url") or "",
                    citation_count=citation_count if citation_count is not None else -1,
                )
                for raw, vector, sparse in zip(raw_chunks, vectors, sparse_vectors, strict=True)
            ]
            store.add_chunks(chunks)
            state.add_findings(
                [
                    Finding(text=c.text, source_id=candidate.id, sub_question=sub_question_text)
                    for c in raw_chunks
                ]
            )
    else:
        _emit(on_progress, f"Already indexed (cache hit): {candidate.title[:80]}")

    state.mark_read(candidate.id)


def run(
    sub_question: SubQuestion,
    sources: list[Source],
    state: ResearchState,
    store: QdrantStore,
    discovery_limit: int | None = None,
    on_progress: ProgressCallback | None = None,
) -> None:
    """Одна итерация воронки для одного подвопроса — расширяет state и Qdrant.

    `discovery_limit` растёт с каждой повторной попыткой (см. agent/loop.py) —
    так повторный проход реально достаёт статьи, которых не было в первой
    (более узкой) выдаче, а не просто повторяет тот же самый запрос.
    """
    _emit(on_progress, f"Searching sources for: {sub_question.text}…")
    discovered = _discover(sub_question, sources, discovery_limit or config.FUNNEL_DISCOVERY_LIMIT_PER_SOURCE)
    new_candidates = state.add_candidates(discovered)
    survivors = _triage(sub_question, new_candidates)
    _emit(
        on_progress,
        f"Found {len(discovered)} candidates, {len(survivors)} passed triage.",
    )

    for candidate in survivors:
        if state.budget_exhausted():
            break
        deep_read_candidate(candidate, sub_question.text, state, store, on_progress=on_progress)
