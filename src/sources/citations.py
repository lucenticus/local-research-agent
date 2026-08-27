"""Обогащение метаданными через OpenAlex — без ключа (проверено вручную
2026-08-06: `cited_by_count` через title-search; 2026-08-10:
venue/авторы/институции/h-index тем же способом).

ВНИМАНИЕ (проверено 2026-08-27): OpenAlex перешёл на тираж с деньгами и
отдаёт 429 `{"error": "Rate limit exceeded", "message": "Insufficient
budget ... Resets at midnight UTC"}` после исчерпания суточного бюджета —
которого хватает примерно на один прогон эвала. Поэтому цитируемость
arXiv-кандидатов в воронке берётся не отсюда, а батчем из Semantic Scholar
(`semantic_scholar.lookup_citation_counts`): один запрос вместо тридцати,
точный id вместо поиска по заголовку, бесплатно.

Здесь остался только `lookup_paper_details` — карточка статьи в дайджесте
(venue, институции, h-index авторов), чего S2 батчем не отдаёт. Это витрина:
вызывается по одной статье по требованию пользователя, а не по десяткам
кандидатов в горячем пути, и на исчерпанном бюджете просто показывает
пустоту.

Semantic Scholar уже отдаёт `citationCount` прямо при discovery — обогащение
нужно только источникам, которые своей цитируемости не знают (arXiv).

ВАЖНО (реальный баг, найден на живом прогоне): обычный полнотекстовый
`search=` параметр OpenAlex ранжирует по общей релевантности, а не по
похожести заголовка — на запрос `"The risk of KV cache compression"` он
вернул топ-1 результатом статью **"XGBoost"** (50k+ цитирований, но
абсолютно не та статья). Обязателен `filter=title.search:...` (собственно
поиск по заголовку) + доп. проверка похожести заголовков как защита от
частичных совпадений на короткий/общий запрос.

`lookup_paper_details` (§ пользовательский запрос: дайджест должен уметь
показывать институции/venue/h-index авторов) — тот же title-search, плюс
батч-запрос `/authors?filter=id:A1|A2|...` за h-index всех авторов одним
вызовом, не по одному. Совсем свежие статьи (только что на arXiv) у OpenAlex
почти никогда не проиндексированы — задержка индексации, не баг; функция в
этом случае честно возвращает `None`, а не пытается угадать через
name-search автора (найдено вживую: поиск по имени автора неоднозначен,
"Ashish Vaswani" даёт 3 разных человека с h-index 29/5/0 — присвоить не тот
h-index не тому автору хуже, чем не показать его вовсе).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher

from ._common import fetch_json

_WORKS_URL = "https://api.openalex.org/works"
_AUTHORS_URL = "https://api.openalex.org/authors"
_TIMEOUT_SECONDS = 10
_MIN_TITLE_SIMILARITY = 0.6


def _title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def _find_work(title: str) -> dict | None:
    title = title.strip()
    if not title:
        return None
    body = fetch_json(
        _WORKS_URL, {"filter": f"title.search:{title}", "per_page": 1}, timeout=_TIMEOUT_SECONDS
    )
    if body is None:
        return None
    results = body.get("results") or []
    if not results:
        return None
    result = results[0]
    result_title = result.get("title") or ""
    if _title_similarity(title, result_title) < _MIN_TITLE_SIMILARITY:
        # title.search — не точное совпадение, а токен-поиск с ранжированием;
        # на короткие/общие запросы иногда всё равно всплывает не та статья.
        # Лучше остаться без данных, чем приписать чужие.
        return None
    return result


@dataclass
class AuthorDetails:
    name: str
    institution: str | None = None
    h_index: int | None = None


@dataclass
class PaperDetails:
    citation_count: int | None = None
    venue: str | None = None
    authors: list[AuthorDetails] = field(default_factory=list)


def _short_id(openalex_url: str | None) -> str | None:
    return openalex_url.rsplit("/", 1)[-1] if openalex_url else None


def _lookup_h_indexes(author_ids: list[str]) -> dict[str, int]:
    """Один батч-запрос за h-index всех авторов сразу — не по одному на
    каждого (OpenAlex OR-фильтр по id, `id:A1|A2|...`)."""
    if not author_ids:
        return {}
    body = fetch_json(
        _AUTHORS_URL, {"filter": f"id:{'|'.join(author_ids)}", "per_page": len(author_ids)},
        timeout=_TIMEOUT_SECONDS,
    )
    if body is None:
        return {}
    result: dict[str, int] = {}
    for author in body.get("results") or []:
        sid = _short_id(author.get("id"))
        h_index = (author.get("summary_stats") or {}).get("h_index")
        if sid and isinstance(h_index, int):
            result[sid] = h_index
    return result


def lookup_paper_details(title: str) -> PaperDetails | None:
    """Цитируемость + venue + авторы (институция, h-index) — `None`, если
    статья не найдена в OpenAlex (см. docstring модуля: обычное дело для
    только что опубликованных препринтов, не повод гадать).

    Недоступность OpenAlex здесь тоже `None`: показать нечего в обоих
    случаях, и городить различие ради UI незачем — в отличие от скоринга и
    фикстур, где «не нашли» и «не смогли спросить» обязаны различаться."""
    work = _find_work(title)
    if work is None:
        return None

    citation_count = work.get("cited_by_count")
    location = work.get("primary_location") or {}
    source = location.get("source") or {}
    venue = source.get("display_name")

    entries: list[tuple[str, str | None, str | None]] = []
    for authorship in work.get("authorships") or []:
        author = authorship.get("author") or {}
        name = author.get("display_name") or "?"
        institutions = authorship.get("institutions") or []
        institution = institutions[0].get("display_name") if institutions else None
        entries.append((name, institution, _short_id(author.get("id"))))

    h_index_by_id = _lookup_h_indexes([sid for _, _, sid in entries if sid])
    authors = [
        AuthorDetails(name=name, institution=institution, h_index=h_index_by_id.get(sid))
        for name, institution, sid in entries
    ]

    return PaperDetails(
        citation_count=citation_count if isinstance(citation_count, int) else None,
        venue=venue,
        authors=authors,
    )
