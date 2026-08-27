"""Semantic Scholar Graph API — discovery по метаданным + цитируемость чужих
статей батчем. Без ключа.

Публичный безключевой тираж жёстко троттлится (подтверждено вручную: 429
"Too Many Requests" почти на каждый запрос без ключа) — встроен retry с
экспоненциальным бэкоффом. Ключ (`S2_API_KEY` в env) поднимает лимиты
кратно, но не обязателен для работы источника.

**Цитируемость arXiv-статей живёт здесь, а не в `citations.py` (OpenAlex).**
Замерено 2026-08-27, и разница не косметическая:

* **один вызов вместо тридцати.** `/paper/batch` принимает список id (до 500);
  30 arXiv-id вернулись за 1.1с, тогда как OpenAlex требовал по запросу на
  кандидата — ~30 последовательных round-trip'ов на подвопрос;
* **точное сопоставление вместо поиска по заголовку.** OpenAlex искал статью
  по названию, и на запрос про KV-cache compression отдавал "XGBoost" с 50k
  цитирований (см. `citations.py`) — приходилось защищаться порогом похожести
  заголовков. По arXiv-id ошибиться статьёй нельзя, эвристика не нужна;
* **бесплатно.** OpenAlex перешёл на платный тираж и после исчерпания
  суточного бюджета отдаёт 429 до полуночи UTC — на масштабе эвала это
  наступает за один прогон.

Троттлинг у `/paper/batch` отдельный от `/paper/search`: батч отвечал 200
дважды подряд в тот момент, когда search отдавал 429 на все шесть запросов.

arXiv-id приходит с версией (`2203.16487v6`), S2 такие не принимает
(`{"error": "No valid paper ids given"}`) — суффикс срезается.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from ._common import SourceUnavailable
from .base import DiscoveredItem

_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
# Лимит S2 — 500 id на запрос; воронка приносит десятки, так что это запас,
# а не рабочий режим. Разбивать на страницы всё равно надо: молча потерять
# хвост списка хуже, чем сделать два вызова.
_BATCH_MAX_IDS = 500
_ARXIV_ID_RE = re.compile(r"v\d+$")
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


def _strip_version(arxiv_id: str) -> str:
    return _ARXIV_ID_RE.sub("", arxiv_id.strip())


def _headers() -> dict[str, str]:
    headers = {"User-Agent": "local-research-agent/0.1"}
    api_key = os.environ.get("S2_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def lookup_citation_counts(arxiv_ids: list[str]) -> dict[str, int]:
    """Цитируемость пачки arXiv-статей: `{arxiv_id: count}`, одним запросом.

    Ключи возвращаются РОВНО в том виде, в каком пришли (с версией, если она
    была) — вызывающий сопоставляет со своими кандидатами, не зная про
    внутреннюю нормализацию.

    Статьи, которых S2 не знает, просто отсутствуют в ответе: это «нет
    данных», нормальное дело для свежего препринта. А вот недоступность API —
    `SourceUnavailable`, потому что записать её как «нет данных» значило бы
    заморозить аварию в виде факта (см. `_common.SourceUnavailable`).
    """
    if not arxiv_ids:
        return {}

    counts: dict[str, int] = {}
    for start in range(0, len(arxiv_ids), _BATCH_MAX_IDS):
        page = arxiv_ids[start : start + _BATCH_MAX_IDS]
        # Один запрошенный id может встретиться дважды (разные подвопросы
        # нашли ту же статью) — S2 отдаёт ответы позиционно, так что
        # дедупликация тут сломала бы соответствие ответов запросу.
        body = _fetch_batch([f"ARXIV:{_strip_version(i)}" for i in page])
        for arxiv_id, entry in zip(page, body, strict=True):
            count = (entry or {}).get("citationCount")
            if isinstance(count, int):
                counts[arxiv_id] = count
    return counts


def _fetch_batch(ids: list[str]) -> list[dict | None]:
    request = urllib.request.Request(
        f"{_BATCH_URL}?fields=citationCount",
        data=json.dumps({"ids": ids}).encode("utf-8"),
        headers={**_headers(), "Content-Type": "application/json"},
    )
    last_error: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
                body = json.loads(response.read())
            if not isinstance(body, list):
                # S2 отдаёт объект с "error" на невалидный запрос, а не список
                raise SourceUnavailable(f"Semantic Scholar batch: {body}")
            return body
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code != 429 or attempt == _MAX_RETRIES - 1:
                raise SourceUnavailable(f"Semantic Scholar batch: {exc}") from exc
            time.sleep(_BACKOFF_BASE_SECONDS * (2**attempt))
        except (OSError, ValueError) as exc:
            raise SourceUnavailable(f"Semantic Scholar batch: {exc}") from exc
    raise SourceUnavailable(f"Semantic Scholar batch недоступен: {last_error}")
