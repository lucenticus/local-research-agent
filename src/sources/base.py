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
    meta: dict = field(default_factory=dict)


class Source(Protocol):
    name: str

    def discover(self, query: str, limit: int) -> list[DiscoveredItem]:
        """Возвращает только метаданные (title/abstract/год/цитируемость) —
        никакого полного текста и записи в векторное хранилище на этом шаге (§1)."""
        ...
