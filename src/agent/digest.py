"""Digest-режим: "что нового вышло" в заданных arXiv-категориях за последние
N дней — browse, не Q&A (§ пользовательский запрос: агент должен хорошо
следить за свежими статьями в области ИИ).

Сознательно отдельно от agent/funnel.py, а не поверх него: воронка всегда
организована вокруг конкретного подвопроса (`discover(query, limit)` +
триаж по релевантности этому подвопросу) — для дайджеста нет вопроса,
нужен весь свежий поток одной-нескольких категорий, отсортированный по
дате. Прогонять это через funnel/loop означало бы придумывать фиктивный
"вопрос" и потом резать по релевантности то, что и так уже отобрано по
свежести — бессмысленно и медленнее (funnel эмбеддит и реранкает каждый
подвопрос отдельно, тут это не нужно).

Опциональное bounded LLM-резюме тем (`config.DIGEST_SUMMARIZE`) — свободный
обзорный абзац по аннотациям, а не цитируемый ответ с проверкой
faithfulness/coverage, как у `research()`/`ask()`: явно помечен в выводе
как обзор, не факт с источником.

Опциональный "глубокий" анализ (`deep=True`, § пользовательский запрос) —
на каждую статью: русское саммари (bounded LLM по заголовку+аннотации, без
досочинения — см. `_ITEM_SUMMARY_SYSTEM_PROMPT`) + реальные метаданные из
OpenAlex (`sources/citations.py::lookup_paper_details`): цитируемость,
venue, институции и h-index авторов. Совсем свежие препринты у OpenAlex
почти никогда ещё не проиндексированы (задержка индексации) — в этом
случае честно `analysis.details is None`, а не выдумка через name-search
автора (см. docstring `citations.py` про тёзок с разным h-index). Не
включено по умолчанию: N дополнительных LLM-вызовов + N OpenAlex-lookup'ов
на дайджест из `limit` статей — заметно медленнее обычного browse-режима,
поэтому ограничено `config.DIGEST_DEEP_MAX_ITEMS` независимо от `limit`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .. import config
from ..providers import llm
from ..sources.arxiv import ArxivSource
from ..sources.base import DiscoveredItem
from ..sources.citations import PaperDetails, lookup_paper_details
from .progress import ProgressCallback, emit as _emit


@dataclass
class ItemAnalysis:
    summary_ru: str
    details: PaperDetails | None  # None — статья ещё не проиндексирована в OpenAlex


@dataclass
class DigestResult:
    items: list[DiscoveredItem]
    days: int
    categories: list[str]
    query: str | None = None
    summary: str | None = None
    analyses: dict[str, ItemAnalysis] = field(default_factory=dict)  # ключ — item.id


_SUMMARY_SYSTEM_PROMPT = (
    "Ты — обозреватель научных статей в области ИИ. Тебе дан список "
    "заголовков и аннотаций свежих статей. Напиши краткий (3-6 предложений) "
    "обзор основных тем и трендов, которые видны в этой подборке — не "
    "пересказывай каждую статью по отдельности, укажи на общие направления, "
    "повторяющиеся идеи и заметные результаты. Отвечай на русском языке."
)

_ITEM_SUMMARY_SYSTEM_PROMPT = (
    "Ты — обозреватель научных статей в области ИИ. Тебе дан заголовок и "
    "аннотация одной статьи на английском. Напиши краткое саммари на "
    "русском языке (3-5 предложений): какую проблему решает статья, какой "
    "подход предлагает и какие основные результаты. Пиши только на основе "
    "данного текста, не добавляй фактов, которых там нет."
)


def _summarize(items: list[DiscoveredItem]) -> str:
    listing = "\n\n".join(f"{item.title}\n{item.abstract[:400]}" for item in items)
    prompt = llm.build_chat_prompt(_SUMMARY_SYSTEM_PROMPT, listing)
    return llm.generate(prompt, max_tokens=400).strip()


def _summarize_item(item: DiscoveredItem) -> str:
    user_message = f"{item.title}\n\n{item.abstract}"
    prompt = llm.build_chat_prompt(_ITEM_SUMMARY_SYSTEM_PROMPT, user_message)
    return llm.generate(prompt, max_tokens=250).strip()


def _analyze_items(
    items: list[DiscoveredItem], on_progress: ProgressCallback | None
) -> dict[str, ItemAnalysis]:
    analyses: dict[str, ItemAnalysis] = {}
    for i, item in enumerate(items, start=1):
        _emit(on_progress, f"Анализируем статью {i}/{len(items)}: {item.title[:70]}…")
        analyses[item.id] = ItemAnalysis(
            summary_ru=_summarize_item(item), details=lookup_paper_details(item.title)
        )
    return analyses


def run_digest(
    days: int | None = None,
    categories: list[str] | None = None,
    limit: int | None = None,
    summarize: bool | None = None,
    query: str | None = None,
    deep: bool = False,
    on_progress: ProgressCallback | None = None,
) -> DigestResult:
    days = config.DIGEST_DEFAULT_DAYS if days is None else days
    categories = categories or config.ARXIV_AI_CATEGORIES
    limit = config.DIGEST_DEFAULT_LIMIT if limit is None else limit
    summarize = config.DIGEST_SUMMARIZE if summarize is None else summarize
    query = query.strip() if query and query.strip() else None

    scope = f"по «{query}» " if query else ""
    _emit(on_progress, f"Ищем статьи {scope}за последние {days} дн. в {', '.join(categories)}…")
    items = ArxivSource(categories=categories).recent(days=days, limit=limit, query=query)
    _emit(on_progress, f"Найдено {len(items)}.")

    summary = None
    if summarize and items:
        _emit(on_progress, "Собираем обзор тем…")
        summary = _summarize(items)

    analyses: dict[str, ItemAnalysis] = {}
    if deep and items:
        analyses = _analyze_items(items[: config.DIGEST_DEEP_MAX_ITEMS], on_progress)

    return DigestResult(
        items=items, days=days, categories=categories, query=query, summary=summary, analyses=analyses
    )
