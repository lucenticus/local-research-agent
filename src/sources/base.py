"""Общий интерфейс источников: discovery — только метаданные, без полного текста.

Провайдер-шов (§7 CLAUDE.md): агентная логика зовёт `Source.discover(...)`, а
не HTTP-клиенты конкретных API напрямую.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class DiscoveredItem:
    id: str
    source: str
    title: str
    abstract: str
    url: str = ""
    year: int | None = None
    citation_count: int | None = None
    # ISO 8601 timestamp (day precision as a minimum) — `year` alone is too
    # coarse for recency triage (funnel.py::_recency_boost): two papers from
    # the same year but 11 months apart shouldn't score the same "freshness".
    # Only arXiv currently populates this (has a precise submission date);
    # other sources leave it None, and recency boost is simply 0 for them.
    published_date: str | None = None
    meta: dict = field(default_factory=dict)


class Source(Protocol):
    name: str

    def discover(self, query: str, limit: int) -> list[DiscoveredItem]:
        """Возвращает только метаданные (title/abstract/год/цитируемость) —
        никакого полного текста и записи в векторное хранилище на этом шаге (§1)."""
        ...
