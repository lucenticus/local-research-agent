"""Оборачивает `Source.discover()` в LangChain `StructuredTool` — `agent/
funnel.py::_discover()` зовёт источники через `.invoke()`, а не напрямую
через метод протокола `Source` (§7 CLAUDE.md: провайдер-шов, теперь в
LangChain-идиоме). Сама логика discovery (HTTP, парсинг, ретраи по 429 и
т.п.) не переписывается — `Source.discover()` в `sources/arxiv.py` и др.
остаётся как есть, это по-прежнему единственное место, где она живёт."""

from __future__ import annotations

from langchain_core.tools import StructuredTool

from .base import DiscoveredItem, Source


def make_discover_tool(source: Source) -> StructuredTool:
    def _discover(query: str, limit: int) -> list[DiscoveredItem]:
        return source.discover(query, limit)

    return StructuredTool.from_function(
        func=_discover,
        name=f"discover_{source.name}",
        description=f"Discover candidate documents from the '{source.name}' source for a search query.",
    )
