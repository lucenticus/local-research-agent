"""Общий веб-поиск через локальный SearXNG (Docker, свой инстанс).

Изначально планировался Tavily API (нужен ключ). На практике проверено
вручную 2026-08-05, что бесплатные варианты без своей инфраструктуры не
работают: DuckDuckGo HTML блокирует автоматические запросы CAPTCHA-
челленджем ("making sure you're not a bot"), а публичные SearXNG-инстансы
либо жёстко rate-limit'ят (429 почти сразу), либо держат JSON API
отключённым (публичные инстансы выключают его по умолчанию из-за абьюза).

Решение — свой локальный SearXNG в Docker (`docker-compose.yml` в корне
репозитория, `searxng/settings.yml` явно включает `formats: [html, json]`):
без ключа, без лимитов, полностью локально. Поднять перед использованием:

    docker compose up -d

Без поднятого контейнера `discover()` тихо возвращает пустой список (не
бросает исключение) — funnel.py и так переживает недоступность источника,
воронка просто продолжает с тем, что нашли остальные источники.

Найдено реальным прогоном 2026-08-06: общий веб-поиск иногда находит не
статьи, а страницы-листинги (например, `arxiv.org/list/cs.AI/recent` —
список последних публикаций по категории) — это навигационная страница без
собственного содержательного текста по вопросу, но она проходила как
полноценный источник. Такие URL отфильтровываются в `_parse`.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

from .. import config
from ._common import is_article_url
from .base import DiscoveredItem

_TIMEOUT_SECONDS = 15


class WebSource:
    name = "web"

    def __init__(self, base_url: str | None = None):
        self._base_url = base_url or config.SEARXNG_BASE_URL

    def discover(self, query: str, limit: int) -> list[DiscoveredItem]:
        params = urllib.parse.urlencode({"q": query, "format": "json"})
        request = urllib.request.Request(
            f"{self._base_url}/search?{params}",
            headers={"User-Agent": "local-research-agent/0.1"},
        )
        try:
            with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
                body = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            return []  # локальный SearXNG не поднят - источник просто пуст
        return list(self._parse(body))[:limit]

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
