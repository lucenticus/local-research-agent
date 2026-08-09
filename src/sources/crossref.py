"""CrossRef REST API — discovery по метаданным (DOI-регистрации), без ключа.

Дополняет arXiv/Semantic Scholar непрепринтными/журнальными публикациями,
которых там часто нет. `is-referenced-by-count` — реальный счётчик цитирований
(в отличие от arXiv, обогащать через OpenAlex не нужно).

ВАЖНО, проверено вручную 2026-08-06: CrossRef во многих записях НЕ отдаёт
`abstract` вовсе (издатель не передал) — такие DiscoveredItem с пустым
abstract естественно отсеются на триаже (`funnel._triage`: кандидаты без
abstract не скорятся), это тот же паттерн, что и с Semantic Scholar
(`paper.get("abstract") or ""`), не отдельный кейс для CrossRef.

Когда abstract есть, он приходит JATS-XML-подобной разметкой
(`<title>Abstract</title><p>...</p>`) — снимается через BeautifulSoup
(`.get_text()`), тем же способом, что и HTML-страницы в ingest/extract.py.
"""

from __future__ import annotations

from ._common import fetch_json
from .base import DiscoveredItem

_API_URL = "https://api.crossref.org/works"


def _strip_jats(abstract: str) -> str:
    if not abstract:
        return ""
    from bs4 import BeautifulSoup  # lazy import: опциональная зависимость (см. ingest/extract.py)

    soup = BeautifulSoup(abstract, "html.parser")
    for title_tag in soup.find_all("title"):
        # JATS <title>Abstract</title> — секционная метка ("Abstract",
        # "Background", "Methods"...), не содержательный текст.
        title_tag.decompose()
    return " ".join(soup.get_text(separator=" ").split())


def _year_from_date_parts(item: dict) -> int | None:
    for key in ("published", "published-print", "published-online"):
        date_parts = (item.get(key) or {}).get("date-parts")
        if date_parts and date_parts[0]:
            return date_parts[0][0]
    return None


class CrossrefSource:
    name = "crossref"

    def discover(self, query: str, limit: int) -> list[DiscoveredItem]:
        body = fetch_json(
            _API_URL, {"query": query, "rows": limit},
            headers={"User-Agent": "local-research-agent/0.1 (mailto:noreply@example.com)"},
        )
        if body is None:
            return []
        return list(self._parse(body))

    def _parse(self, body: dict):
        for item in (body.get("message") or {}).get("items") or []:
            doi = item.get("DOI")
            titles = item.get("title") or []
            if not doi or not titles:
                continue
            citation_count = item.get("is-referenced-by-count")
            yield DiscoveredItem(
                id=f"doi:{doi}",
                source=self.name,
                title=titles[0],
                abstract=_strip_jats(item.get("abstract") or ""),
                url=item.get("URL") or f"https://doi.org/{doi}",
                year=_year_from_date_parts(item),
                citation_count=citation_count if isinstance(citation_count, int) else None,
                meta={},
            )
