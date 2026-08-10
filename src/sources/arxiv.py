"""arXiv API — discovery по метаданным (Atom-фид), без скачивания PDF.

Публичное API, ключ не нужен. `export.arxiv.org` редиректит http->https
(проверено вручную) — используем https сразу, чтобы не тратить round-trip.

`categories` (§ пользовательский запрос: агент должен хорошо работать по
свежим статьям в области ИИ) — опциональный `cat:` фильтр, AND'ится с
keyword-запросом. Без него keyword-поиск по всему arXiv нередко ловит
физику/математику/econ, где случайно встретились те же слова (например,
"attention" — термин и в ML, и в психологии/нейронауке). arXiv-категории:
https://arxiv.org/category_taxonomy — `config.ARXIV_AI_CATEGORIES` держит
дефолтный набор под ИИ/ML (cs.AI/cs.LG/cs.CL/cs.CV/cs.NE/stat.ML).

`sortBy=relevance` остаётся дефолтом сознательно, не `submittedDate` —
чистая сортировка по дате в узком `max_results` окне легко даёт совершенно
нерелевантные, но новые статьи; "свежесть" вместо этого учитывается позже,
в триаже воронки (`agent/funnel.py::_recency_boost`), поверх релевантной
выдачи, а не вместо неё.

`recent()` — отдельный от `discover()` метод для дайджест-режима
(`agent/digest.py`): browse "что нового в категории", а не поиск по
конкретному запросу — сортировка по дате, без keyword-фильтра вовсе.
"""

from __future__ import annotations

import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

from .base import DiscoveredItem

_API_URL = "https://export.arxiv.org/api/query"
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
_TIMEOUT_SECONDS = 15


class ArxivSource:
    name = "arxiv"

    def __init__(self, categories: list[str] | None = None):
        self._categories = categories

    def discover(self, query: str, limit: int) -> list[DiscoveredItem]:
        # AND по словам, не точная фраза — вопросы на естественном языке почти
        # никогда не совпадают с заголовком/абстрактом статьи дословно.
        words = query.split()
        keyword_query = " AND ".join(f"all:{w}" for w in words) if words else f"all:{query}"
        if self._categories:
            category_clause = " OR ".join(f"cat:{c}" for c in self._categories)
            search_query = f"({category_clause}) AND ({keyword_query})"
        else:
            search_query = keyword_query
        return self._query(search_query, limit=limit, sort_by="relevance")

    def recent(self, days: int, limit: int) -> list[DiscoveredItem]:
        """Последние публикации в `self._categories`, отсортированные по дате
        (не по релевантности — здесь нет запроса, который можно было бы с
        чем-то сравнивать). Отфильтровано клиентской стороной по `days` —
        проще и надёжнее, чем городить `submittedDate:[...]` range-синтаксис
        arXiv-запроса, а `max_results` и так достаточно мал, чтобы не тянуть
        лишнего."""
        if not self._categories:
            raise ValueError("recent() needs categories - browsing all of arXiv unfiltered isn't useful")
        category_clause = " OR ".join(f"cat:{c}" for c in self._categories)
        items = self._query(category_clause, limit=limit, sort_by="submittedDate")
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        return [item for item in items if _published_after(item, cutoff)]

    def _query(self, search_query: str, *, limit: int, sort_by: str) -> list[DiscoveredItem]:
        params = urllib.parse.urlencode(
            {
                "search_query": search_query,
                "start": 0,
                "max_results": limit,
                "sortBy": sort_by,
                **({"sortOrder": "descending"} if sort_by == "submittedDate" else {}),
            }
        )
        request = urllib.request.Request(
            f"{_API_URL}?{params}", headers={"User-Agent": "local-research-agent/0.1"}
        )
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
            body = response.read()
        return list(self._parse(body))

    def _parse(self, body: bytes):
        root = ET.fromstring(body)
        for entry in root.findall("atom:entry", _ATOM_NS):
            arxiv_id = (entry.findtext("atom:id", default="", namespaces=_ATOM_NS) or "").rsplit(
                "/", 1
            )[-1]
            title = " ".join((entry.findtext("atom:title", default="", namespaces=_ATOM_NS) or "").split())
            summary = " ".join(
                (entry.findtext("atom:summary", default="", namespaces=_ATOM_NS) or "").split()
            )
            published = entry.findtext("atom:published", default="", namespaces=_ATOM_NS) or ""
            year = int(published[:4]) if published[:4].isdigit() else None
            authors = [
                " ".join((name.text or "").split())
                for name in entry.findall("atom:author/atom:name", _ATOM_NS)
                if (name.text or "").strip()
            ]
            categories = [
                cat.get("term", "") for cat in entry.findall("atom:category", _ATOM_NS) if cat.get("term")
            ]
            yield DiscoveredItem(
                id=f"arxiv:{arxiv_id}",
                source=self.name,
                title=title,
                abstract=summary,
                url=f"https://arxiv.org/abs/{arxiv_id}",
                year=year,
                citation_count=None,  # arXiv API не отдаёт цитируемость
                published_date=published or None,
                meta={"pdf_url": f"https://arxiv.org/pdf/{arxiv_id}", "authors": authors, "categories": categories},
            )


def _published_after(item: DiscoveredItem, cutoff: datetime) -> bool:
    if not item.published_date:
        return False
    try:
        published = datetime.fromisoformat(item.published_date.replace("Z", "+00:00"))
    except ValueError:
        return False
    return published >= cutoff
