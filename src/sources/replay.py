"""Запись и воспроизведение ответов discovery — фикстуры для eval и кэш (узел 21).

Обёртка живёт на уровне `Source`, а не HTTP: ключ (`имя источника`, `запрос`,
`limit`) осмыслен, читается глазами в JSON и не зависит от того, ходит
конкретный источник через `fetch_json`, `urlopen` или MCP.

Зачем: `research` ходит в живой интернет, и выдача источников меняется день
ко дню. Без фикстур регресс-прогон меряет не наш код, а то, что сегодня
выложили на arXiv.

**Важное ограничение.** Ключ — это подвопрос, который получил `discover()`,
а не итоговый запрос к API источника. Значит фикстуры замораживают «что
источник ответил на такой подвопрос» и НЕ годятся для проверки изменений в
том, как мы формируем запрос к источнику (`arxiv.py::_keyword_query` и
подобное): при replay вернётся старая запись. Такие изменения меряются
только живым прогоном — см. `evals/run_discovery.py`.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .base import DiscoveredItem, Source

MODE_OFF = "off"
MODE_RECORD = "record"
MODE_REPLAY = "replay"


def _key(source_name: str, query: str, limit: int) -> str:
    return f"{source_name}|{limit}|{query}"


class DiscoveryCache:
    """JSON-файл с ответами discovery. Промахи в replay-режиме считаются, а не
    падают: добавить новый вопрос в golden-набор не должно ломать прогон, но
    и молча меряться на пустоте тоже нельзя — счётчик печатает раннер."""

    def __init__(self, path: Path | str, mode: str = MODE_OFF):
        self.path = Path(path)
        self.mode = mode
        self.misses: list[str] = []
        self._entries: dict[str, list[dict[str, Any]]] = {}
        if self.mode == MODE_REPLAY and self.path.exists():
            self._entries = json.loads(self.path.read_text(encoding="utf-8"))
        elif self.mode == MODE_RECORD and self.path.exists():
            # дописываем к существующим, а не затираем: записать набор можно
            # в несколько заходов (источники троттлят по-разному)
            self._entries = json.loads(self.path.read_text(encoding="utf-8"))

    def get(self, source_name: str, query: str, limit: int) -> list[DiscoveredItem] | None:
        raw = self._entries.get(_key(source_name, query, limit))
        if raw is None:
            self.misses.append(_key(source_name, query, limit))
            return None
        return [DiscoveredItem(**item) for item in raw]

    def put(self, source_name: str, query: str, limit: int, items: list[DiscoveredItem]) -> None:
        self._entries[_key(source_name, query, limit)] = [asdict(i) for i in items]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._entries, ensure_ascii=False, indent=1, sort_keys=True),
            encoding="utf-8",
        )

    def __len__(self) -> int:
        return len(self._entries)


class CachedSource:
    """`Source`, обёрнутый кэшем. В `off` — прозрачная передача."""

    def __init__(self, inner: Source, cache: DiscoveryCache):
        self._inner = inner
        self._cache = cache
        self.name = inner.name

    def discover(self, query: str, limit: int) -> list[DiscoveredItem]:
        if self._cache.mode == MODE_REPLAY:
            return self._cache.get(self.name, query, limit) or []
        items = self._inner.discover(query, limit)
        if self._cache.mode == MODE_RECORD:
            self._cache.put(self.name, query, limit, items)
        return items


def wrap(sources: list[Source], cache: DiscoveryCache) -> list[Source]:
    """Оборачивает все источники; при `off` возвращает их как есть."""
    if cache.mode == MODE_OFF:
        return sources
    return [CachedSource(s, cache) for s in sources]
