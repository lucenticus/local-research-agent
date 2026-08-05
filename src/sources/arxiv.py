"""arXiv API — discovery по метаданным (Atom-фид), без скачивания PDF.

Публичное API, ключ не нужен. `export.arxiv.org` редиректит http->https
(проверено вручную) — используем https сразу, чтобы не тратить round-trip.
"""

from __future__ import annotations

import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from .base import DiscoveredItem

_API_URL = "https://export.arxiv.org/api/query"
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
_TIMEOUT_SECONDS = 15


class ArxivSource:
    name = "arxiv"

    def discover(self, query: str, limit: int) -> list[DiscoveredItem]:
        # AND по словам, не точная фраза — вопросы на естественном языке почти
        # никогда не совпадают с заголовком/абстрактом статьи дословно.
        words = query.split()
        search_query = " AND ".join(f"all:{w}" for w in words) if words else f"all:{query}"
        params = urllib.parse.urlencode(
            {
                "search_query": search_query,
                "start": 0,
                "max_results": limit,
                "sortBy": "relevance",
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
            yield DiscoveredItem(
                id=f"arxiv:{arxiv_id}",
                source=self.name,
                title=title,
                abstract=summary,
                url=f"https://arxiv.org/abs/{arxiv_id}",
                year=year,
                citation_count=None,  # arXiv API не отдаёт цитируемость
                meta={"pdf_url": f"https://arxiv.org/pdf/{arxiv_id}"},
            )
