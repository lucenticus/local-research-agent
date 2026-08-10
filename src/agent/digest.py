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
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import config
from ..providers import llm
from ..sources.arxiv import ArxivSource
from ..sources.base import DiscoveredItem


@dataclass
class DigestResult:
    items: list[DiscoveredItem]
    days: int
    categories: list[str]
    summary: str | None = None


_SUMMARY_SYSTEM_PROMPT = (
    "Ты — обозреватель научных статей в области ИИ. Тебе дан список "
    "заголовков и аннотаций свежих статей. Напиши краткий (3-6 предложений) "
    "обзор основных тем и трендов, которые видны в этой подборке — не "
    "пересказывай каждую статью по отдельности, укажи на общие направления, "
    "повторяющиеся идеи и заметные результаты. Отвечай на русском языке."
)


def _summarize(items: list[DiscoveredItem]) -> str:
    listing = "\n\n".join(f"{item.title}\n{item.abstract[:400]}" for item in items)
    prompt = llm.build_chat_prompt(_SUMMARY_SYSTEM_PROMPT, listing)
    return llm.generate(prompt, max_tokens=400).strip()


def run_digest(
    days: int | None = None,
    categories: list[str] | None = None,
    limit: int | None = None,
    summarize: bool | None = None,
) -> DigestResult:
    days = config.DIGEST_DEFAULT_DAYS if days is None else days
    categories = categories or config.ARXIV_AI_CATEGORIES
    limit = config.DIGEST_DEFAULT_LIMIT if limit is None else limit
    summarize = config.DIGEST_SUMMARIZE if summarize is None else summarize

    items = ArxivSource(categories=categories).recent(days=days, limit=limit)
    summary = _summarize(items) if summarize and items else None
    return DigestResult(items=items, days=days, categories=categories, summary=summary)
