"""Общий веб-поиск через Tavily API — управляемый поисковый сервис, без
проблем с CAPTCHA/rate-limit чужих движков (в отличие от локального
SearXNG — `sources/web.py`, см. его docstring: DuckDuckGo/Brave там
регулярно блокируют даже через прокси).

Нужен `TAVILY_API_KEY` в `.env` (см. `.env.example`) — без ключа
`discover()` тихо возвращает пустой список, funnel.py переживает
отсутствие источника (проверено для sources/web.py, тот же принцип).

Реально проверено 2026-08-06: формат ответа — `{"results": [{"title",
"url", "content"}, ...]}`, ключ рабочий, результаты релевантные и
разнообразные (LinkedIn, DEV Community, Reddit, блоги — реально
"весь интернет", не только несколько движков SearXNG).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from ._common import is_article_url
from .base import DiscoveredItem

_API_URL = "https://api.tavily.com/search"
_TIMEOUT_SECONDS = 15


class TavilySource:
    name = "web"  # тот же логический "web"-источник, что и SearXNG-вариант

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
        try:
            with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
                body = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, OSError):
            return []
        return list(self._parse(body))

    def _parse(self, body: dict):
        for result in body.get("results") or []:
            url = result.get("url") or ""
            if not is_article_url(url):
                continue
            yield DiscoveredItem(
                id=f"web:{url}",
                source=self.name,
                title=result.get("title") or "",
                abstract=result.get("content") or "",
                url=url,
                meta={},
            )
