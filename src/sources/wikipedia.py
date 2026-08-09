"""Wikipedia REST API (action=query, generator=search) — discovery по
метаданным, без ключа. Даёт быстрый энциклопедический/справочный контекст
для общих подвопросов, которые arXiv/Semantic Scholar (научные статьи) и
CrossRef (DOI-регистрации) не покрывают вообще (например, "что такое X" на
базовом уровне, а не научная новизна по X).

Один запрос вместо двух: `generator=search` + `prop=extracts|info` в одном
вызове API отдаёт заголовки, интро-абзац (`exintro`+`explaintext` — plain
text, без разметки) и URL разом (проверено вручную 2026-08-06) — не нужен
отдельный round-trip за extract на каждую найденную страницу.

Английская Wikipedia всегда: подвопрос на этот момент уже переведён на
английский в `funnel._discovery_query` до вызова любого источника (см. её
docstring) — тот же переведённый query, что уходит в arXiv/Semantic
Scholar/web, приходит и сюда.
"""

from __future__ import annotations

from ._common import fetch_json
from .base import DiscoveredItem

_API_URL = "https://en.wikipedia.org/w/api.php"


class WikipediaSource:
    name = "wikipedia"

    def discover(self, query: str, limit: int) -> list[DiscoveredItem]:
        body = fetch_json(
            _API_URL,
            {
                "action": "query",
                "generator": "search",
                "gsrsearch": query,
                "gsrlimit": limit,
                "prop": "extracts|info",
                "exintro": 1,
                "explaintext": 1,
                "inprop": "url",
                "format": "json",
            },
        )
        if body is None:
            return []
        return list(self._parse(body))

    def _parse(self, body: dict):
        pages = (body.get("query") or {}).get("pages") or {}
        for page in pages.values():
            title = page.get("title")
            url = page.get("fullurl")
            if not title or not url:
                continue
            yield DiscoveredItem(
                id=f"wikipedia:{page.get('pageid')}",
                source=self.name,
                title=title,
                abstract=(page.get("extract") or "").strip(),
                url=url,
                meta={},
            )
