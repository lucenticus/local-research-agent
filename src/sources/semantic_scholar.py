"""Semantic Scholar Graph API — discovery по метаданным, без ключа.

Публичный безключевой тираж жёстко троттлится (подтверждено вручную: 429
"Too Many Requests" почти на каждый запрос без ключа) — встроен retry с
экспоненциальным бэкоффом. Ключ (`S2_API_KEY` в env) поднимает лимиты
кратно, но не обязателен для работы источника.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

from .base import DiscoveredItem

_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_FIELDS = "title,abstract,year,citationCount,externalIds"
_TIMEOUT_SECONDS = 15
_MAX_RETRIES = 3
_BACKOFF_BASE_SECONDS = 2.0


class SemanticScholarSource:
    name = "semantic_scholar"

    def discover(self, query: str, limit: int) -> list[DiscoveredItem]:
        params = urllib.parse.urlencode({"query": query, "limit": limit, "fields": _FIELDS})
        headers = {"User-Agent": "local-research-agent/0.1"}
        api_key = os.environ.get("S2_API_KEY")
        if api_key:
            headers["x-api-key"] = api_key

        request = urllib.request.Request(f"{_API_URL}?{params}", headers=headers)
        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
                    body = json.loads(response.read())
                return list(self._parse(body))
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code != 429 or attempt == _MAX_RETRIES - 1:
                    raise
                time.sleep(_BACKOFF_BASE_SECONDS * (2**attempt))
        raise RuntimeError(f"Semantic Scholar API недоступен: {last_error}")

    def _parse(self, body: dict):
        for paper in body.get("data") or []:
            paper_id = paper.get("paperId")
            if not paper_id:
                continue
            yield DiscoveredItem(
                id=f"s2:{paper_id}",
                source=self.name,
                title=paper.get("title") or "",
                abstract=paper.get("abstract") or "",
                url=f"https://www.semanticscholar.org/paper/{paper_id}",
                year=paper.get("year"),
                citation_count=paper.get("citationCount"),
                meta={"external_ids": paper.get("externalIds") or {}},
            )
