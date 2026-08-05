"""Web-поиск через Tavily API — опциональный источник.

# ARCH-Q: на этой машине не настроен TAVILY_API_KEY, реальный запрос ни разу
# не проверялся вживую (в отличие от arxiv.py/semantic_scholar.py). Формат
# ответа Tavily API взят из официальной документации, не верифицирован.
Без ключа `discover()` возвращает пустой список и не бросает исключение —
воронка (funnel.py) должна уметь работать при отсутствии этого источника.
"""

from __future__ import annotations

import json
import os
import urllib.request

from .base import DiscoveredItem

_API_URL = "https://api.tavily.com/search"
_TIMEOUT_SECONDS = 15


class WebSource:
    name = "web"

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.environ.get("TAVILY_API_KEY")

    def discover(self, query: str, limit: int) -> list[DiscoveredItem]:
        if not self._api_key:
            return []
        payload = json.dumps(
            {"api_key": self._api_key, "query": query, "max_results": limit}
        ).encode("utf-8")
        request = urllib.request.Request(
            _API_URL,
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "local-research-agent/0.1"},
        )
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
            body = json.loads(response.read())
        return list(self._parse(body))

    def _parse(self, body: dict):
        for i, result in enumerate(body.get("results") or []):
            url = result.get("url") or ""
            yield DiscoveredItem(
                id=f"web:{url or i}",
                source=self.name,
                title=result.get("title") or "",
                abstract=result.get("content") or "",
                url=url,
                meta={},
            )
