"""Обогащение цитируемостью через OpenAlex — источник без ключа, разумные
лимиты (проверено вручную 2026-08-06: `cited_by_count` через title-search).

Semantic Scholar уже отдаёт `citationCount` прямо при discovery — обогащение
нужно только источникам, которые своей цитируемости не знают (arXiv).

ВАЖНО (реальный баг, найден на живом прогоне): обычный полнотекстовый
`search=` параметр OpenAlex ранжирует по общей релевантности, а не по
похожести заголовка — на запрос `"The risk of KV cache compression"` он
вернул топ-1 результатом статью **"XGBoost"** (50k+ цитирований, но
абсолютно не та статья). Обязателен `filter=title.search:...` (собственно
поиск по заголовку) + доп. проверка похожести заголовков как защита от
частичных совпадений на короткий/общий запрос.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from difflib import SequenceMatcher

_API_URL = "https://api.openalex.org/works"
_TIMEOUT_SECONDS = 10
_MIN_TITLE_SIMILARITY = 0.6


def _title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def lookup_citation_count(title: str) -> int | None:
    title = title.strip()
    if not title:
        return None
    params = urllib.parse.urlencode({"filter": f"title.search:{title}", "per_page": 1})
    request = urllib.request.Request(
        f"{_API_URL}?{params}", headers={"User-Agent": "local-research-agent/0.1"}
    )
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
            body = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None
    results = body.get("results") or []
    if not results:
        return None
    result = results[0]
    result_title = result.get("title") or ""
    if _title_similarity(title, result_title) < _MIN_TITLE_SIMILARITY:
        # title.search — не точное совпадение, а токен-поиск с ранжированием;
        # на короткие/общие запросы иногда всё равно всплывает не та статья.
        # Лучше остаться без цитируемости, чем приписать чужое число.
        return None
    count = result.get("cited_by_count")
    return count if isinstance(count, int) else None
