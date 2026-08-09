"""Общие хелперы для источников, бьющих в JSON HTTP API без ключа/с опциональным
ключом (wikipedia.py, crossref.py, web.py, tavily.py, sources/citations.py) —
у всех был один и тот же request/urlopen/json.loads/поймать-сетевую-ошибку
блок, продублированный по файлам. Semantic Scholar (retry-с-backoff-и-raise
на 429) и arXiv (Atom/XML, не JSON) сюда сознательно не входят — у них
действительно другой контракт обработки ошибок, не такая же копипаста."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request

_USER_AGENT = "local-research-agent/0.1"
_TIMEOUT_SECONDS = 15
# urllib.error.URLError (включая HTTPError) наследуется от OSError с Python
# 3.3 — как и socket.timeout/ConnectionError. Один except покрывает "сервер
# недоступен/таймаут/соединение оборвалось"; ValueError — отдельно, для
# невалидного JSON в ответе.
_FETCH_ERRORS = (OSError, ValueError)


def fetch_json(
    url: str,
    params: dict | None = None,
    *,
    json_body: dict | None = None,
    headers: dict | None = None,
    timeout: int = _TIMEOUT_SECONDS,
) -> dict | None:
    """GET (или POST, если передан `json_body`) с urlencode(params) и парсингом
    JSON-ответа. `None` на любую сетевую ошибку/невалидный JSON — источник
    просто оказывается временно пуст, `funnel.py` и так переживает
    недоступность любого отдельного источника."""
    full_url = f"{url}?{urllib.parse.urlencode(params)}" if params else url
    data = json.dumps(json_body).encode("utf-8") if json_body is not None else None
    request_headers = {"User-Agent": _USER_AGENT, **(headers or {})}
    if data is not None:
        request_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(full_url, data=data, headers=request_headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except _FETCH_ERRORS:
        return None


# Страницы-листинги/индексы, не отдельные статьи — не содержательный источник.
# Найдено реальным прогоном 2026-08-06: arxiv.org/list/cs.AI/recent (список
# последних публикаций по категории) проходил как полноценный источник.
_NON_ARTICLE_URL_RE = re.compile(r"arxiv\.org/list/")


def is_article_url(url: str) -> bool:
    return bool(url) and not _NON_ARTICLE_URL_RE.search(url)
