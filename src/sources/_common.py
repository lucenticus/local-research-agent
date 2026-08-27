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


class SourceUnavailable(Exception):
    """Источник не ответил: сеть, таймаут, 429, битый JSON.

    Отличается от «источник ответил, но ничего не нашёл» — это разные факты, и
    склеивать их в один `None` можно ровно до тех пор, пока результат никуда не
    записывается. Как только ответ попадает в фикстуры (`evals/fixtures.py`),
    разница становится критической: записанный `None` читается потом как
    «цитируемость честно неизвестна», хотя на деле мы упёрлись в лимит API.
    Поймано вживую: OpenAlex перешёл на платный тираж и отдаёт 429
    "Insufficient budget" до полуночи UTC — прогон при этом выглядел
    совершенно здоровым."""


def fetch_json(
    url: str,
    params: dict | None = None,
    *,
    json_body: dict | None = None,
    headers: dict | None = None,
    timeout: int = _TIMEOUT_SECONDS,
    strict: bool = False,
) -> dict | None:
    """GET (или POST, если передан `json_body`) с urlencode(params) и парсингом
    JSON-ответа. `None` на любую сетевую ошибку/невалидный JSON — источник
    просто оказывается временно пуст, `funnel.py` и так переживает
    недоступность любого отдельного источника.

    `strict=True` — вместо `None` поднять `SourceUnavailable`. Нужно тем
    вызовам, чей результат может быть записан в фикстуры или иначе принят за
    факт: там «не нашли» и «не смогли спросить» обязаны различаться."""
    full_url = f"{url}?{urllib.parse.urlencode(params)}" if params else url
    data = json.dumps(json_body).encode("utf-8") if json_body is not None else None
    request_headers = {"User-Agent": _USER_AGENT, **(headers or {})}
    if data is not None:
        request_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(full_url, data=data, headers=request_headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except _FETCH_ERRORS as exc:
        if strict:
            raise SourceUnavailable(f"{url}: {exc}") from exc
        return None


# Страницы-листинги/индексы, не отдельные статьи — не содержательный источник.
# Найдено реальным прогоном 2026-08-06: arxiv.org/list/cs.AI/recent (список
# последних публикаций по категории) проходил как полноценный источник.
_NON_ARTICLE_URL_RE = re.compile(r"arxiv\.org/list/")


def is_article_url(url: str) -> bool:
    return bool(url) and not _NON_ARTICLE_URL_RE.search(url)
