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
конкретному запросу — сортировка по дате, всегда только `cat:`-фильтр, без
`all:`-keyword'ов. Раньше опциональный `query` дайджеста AND'ился прямо в
этот search_query — на практике на некоторых запросах (частые слова + все
6 категорий в OR) arXiv отвечал за пределами `_TIMEOUT_SECONDS`, видимо
из-за дороговизны такого составного булева запроса на их стороне. Теперь
`recent()` про `query` вообще не знает: `digest.py` тянет пул статей по
категориям (сортировка по дате, без keyword-фильтра — дешёвый и
предсказуемый по времени запрос) и ранжирует его локально (гибридный
поиск + реранк), а не перекладывает это на arXiv API.

`recent(limit=None)` выгружает всё окно постранично. Пагинация опирается на
ту же сортировку по дате: страницы идут от свежих к старым, поэтому первая
же статья старше cutoff означает "дальше только старее" — и цикл
останавливается. Между страницами выдерживается `_PAGE_DELAY_SECONDS`
(рекомендация arXiv API guide), а 429/5xx ретраятся с backoff'ом:
429 Too Many Requests на выгрузке всего окна ловится реально, не
гипотетически.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

from .base import DiscoveredItem

_API_URL = "https://export.arxiv.org/api/query"
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
# 30, не 15 — digest с query выгружает всё окно постранично (см. `recent`),
# и такие крупные Atom-ответы arXiv собирает/передаёт дольше, чем обычный
# short-list запрос.
_TIMEOUT_SECONDS = 30
# Размер страницы при пагинации — подобран по реальным прогонам, не на глаз:
# на 500 и на 200 записей arXiv стабильно не укладывался в _TIMEOUT_SECONDS
# (на 200 первая страница проходила, а start=200 уже отваливалась по таймауту
# на всех ретраях), тогда как на 100 записей отдаёт за 1-10с на любом
# смещении, включая глубокие. Меньшая страница ещё и мягче к их API.
_PAGE_SIZE = 100
# Пауза между страницами — рекомендация самого arXiv (их API guide просит
# не чаще одного запроса в 3 секунды).
_PAGE_DELAY_SECONDS = 3.0
# И 429 Too Many Requests, и таймауты чтения ловятся на практике при
# выгрузке всего окна (оба подтверждены реальными прогонами) — без ретрая
# дайджест просто падает на середине пагинации.
# Терпеливо: окно 429 у arXiv заметно длиннее пары секунд — на прогоне с
# backoff'ом 5/10/20с попытки кончались раньше, чем снимался лимит.
# 10/20/40/80с даёт ~2.5 минуты ожидания, что для фоновой сборки дайджеста
# лучше, чем упасть на середине пагинации.
_RETRY_ATTEMPTS = 5
_RETRY_BACKOFF_SECONDS = 10.0


def _fetch(request: urllib.request.Request) -> bytes:
    """GET с ретраем на 429/5xx, таймауты и сетевые сбои — с backoff'ом."""
    for attempt in range(_RETRY_ATTEMPTS):
        last = attempt == _RETRY_ATTEMPTS - 1
        try:
            with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            # 4xx кроме 429 — не временная ошибка, ретраить нечего.
            if last or not (exc.code == 429 or exc.code >= 500):
                raise
        except (urllib.error.URLError, TimeoutError):
            if last:
                raise
        time.sleep(_RETRY_BACKOFF_SECONDS * (2**attempt))
    raise RuntimeError("unreachable")  # pragma: no cover


class ArxivSource:
    name = "arxiv"

    def __init__(self, categories: list[str] | None = None):
        self._categories = categories

    def discover(self, query: str, limit: int) -> list[DiscoveredItem]:
        """Поиск по релевантности с откатом на более широкий запрос.

        Пустая выдача у arXiv — обычное дело для длинного вопроса: термы
        соединены через AND, и каждый лишний сужает результат до нуля. Вместо
        того чтобы отдать воронке пустоту (она это переживёт, но подвопрос
        останется незакрытым), пробуем ещё раз, ужав запрос до
        `_RETRY_QUERY_TERMS` самых первых термов.
        """
        items = self._query_keywords(query, limit, _MAX_QUERY_TERMS)
        if items or len(_keywords(query)) <= _RETRY_QUERY_TERMS:
            return items
        return self._query_keywords(query, limit, _RETRY_QUERY_TERMS)

    def _query_keywords(self, query: str, limit: int, max_terms: int) -> list[DiscoveredItem]:
        keyword_query = _keyword_query(query, max_terms)
        category_clause = self._category_clause()
        search_query = f"({category_clause}) AND ({keyword_query})" if category_clause else keyword_query
        return self._query(search_query, limit=limit, sort_by="relevance")

    def recent(
        self,
        days: int,
        limit: int | None = None,
        on_progress: Callable[[int], None] | None = None,
        known_ids: set[str] | None = None,
    ) -> list[DiscoveredItem]:
        """Публикации в `self._categories` за последние `days` дней, по дате.

        `limit=None` — забрать всё окно целиком, постранично (§ пользовательский
        запрос: не ограничивать выдачу, выгружать все статьи за период). Выдача
        отсортирована по `submittedDate` убывающе, поэтому окно само себя
        ограничивает: как только на странице встретилась статья старше cutoff,
        дальше идут только ещё более старые — пагинация останавливается, и
        `submittedDate:[...]`-range в запросе не нужен.

        `known_ids` — инкрементальный режим (§ пользовательский запрос: не
        выкачивать одни и те же статьи по нескольку раз). Новые статьи по той
        же сортировке всегда идут в начале выдачи, поэтому первая страница,
        целиком состоящая из уже известных id, означает "догнали кэш" —
        дальше только известное, и пагинация останавливается. Возвращается
        при этом лишь то, чего в `known_ids` не было: остальное вызывающий
        код и так держит у себя.

        Вызывающий обязан убедиться, что кэш покрывает всё окно, прежде чем
        передавать `known_ids` (см. `QdrantStore.oldest_published_ts`) —
        иначе досрочная остановка срежет нижнюю, ещё не закэшированную часть
        окна.

        `limit` (не None) оставлен для дайджеста без query, где нужны просто
        N последних статей и незачем тянуть всё окно.
        """
        category_clause = self._category_clause()
        if not category_clause:
            raise ValueError("recent() needs categories - browsing all of arXiv unfiltered isn't useful")

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        items: list[DiscoveredItem] = []
        start = 0
        while True:
            page_size = _PAGE_SIZE if limit is None else min(_PAGE_SIZE, limit - len(items))
            page = self._query(category_clause, limit=page_size, sort_by="submittedDate", start=start)
            if not page:
                break
            fresh = [item for item in page if _published_after(item, cutoff)]
            unseen = [item for item in fresh if item.id not in known_ids] if known_ids else fresh
            items.extend(unseen)
            if on_progress:
                on_progress(len(items))
            # Вышли за окно (на странице попались статьи старше cutoff) либо
            # arXiv отдал неполную страницу — дальше только более старые.
            if len(fresh) < len(page) or len(page) < page_size:
                break
            # Догнали кэш: вся страница уже известна, дальше только известное.
            if known_ids and not unseen:
                break
            if limit is not None and len(items) >= limit:
                break
            start += len(page)
            time.sleep(_PAGE_DELAY_SECONDS)
        # Обрезаем сами: arXiv не обязан отдать ровно `max_results` записей.
        return items[:limit] if limit is not None else items

    def _category_clause(self) -> str:
        return " OR ".join(f"cat:{c}" for c in self._categories) if self._categories else ""

    def _query(
        self, search_query: str, *, limit: int, sort_by: str, start: int = 0
    ) -> list[DiscoveredItem]:
        params = urllib.parse.urlencode(
            {
                "search_query": search_query,
                "start": start,
                "max_results": limit,
                "sortBy": sort_by,
                **({"sortOrder": "descending"} if sort_by == "submittedDate" else {}),
            }
        )
        request = urllib.request.Request(
            f"{_API_URL}?{params}", headers={"User-Agent": "local-research-agent/0.1"}
        )
        return list(self._parse(_fetch(request)))

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


# Стоп-слова: вопросительные, служебные, связки. Отбрасываются перед сборкой
# AND-запроса — см. `_keywords`. Список намеренно короткий и консервативный:
# лучше оставить лишний терм (запрос сузится, но останется осмысленным), чем
# выбросить содержательное слово.
_STOPWORDS = frozenset("""
a an the and or of for in on to with without from by as at is are was were be been
what which who whom when where why how does do did doing done can could should would
will shall may might must have has had it its this that these those there here
i we you they he she them us our your their my me
about into over under between across through during than then so such
approach approaches method methods way ways work works use used using
exist exists need needs make makes based given
""".split())

# Сколько термов оставлять в первом, точном запросе. Больше — точнее, но выше
# шанс пустой выдачи; на этот случай есть откат ниже.
_MAX_QUERY_TERMS = 6
# На сколько термов ужимать запрос, если точный не нашёл ничего.
_RETRY_QUERY_TERMS = 3


def _keywords(query: str) -> list[str]:
    """Вопрос на естественном языке -> содержательные термы, по порядку.

    Без этого шага запрос вида "What approaches exist for KV-cache compression
    in transformers?" превращался в AND из восьми термов, включая "What",
    "for", "in" и приклеенный к последнему слову "?" — и arXiv честно
    возвращал НОЛЬ результатов (проверено живым запросом: 0 против 630 у
    "KV-cache AND compression"). Английские вопросы попадали в эту ветку
    напрямую, а русские спасал LLM-перевод в funnel.py, который и так отдаёт
    короткие ключевые слова.
    """
    words = (w.strip("?!.,:;()[]{}\"'\u00ab\u00bb") for w in query.split())
    seen: set[str] = set()
    terms: list[str] = []
    for word in words:
        low = word.lower()
        if not word or low in _STOPWORDS or low in seen:
            continue
        seen.add(low)
        terms.append(word)
    return terms


def _keyword_query(query: str, max_terms: int = _MAX_QUERY_TERMS) -> str:
    # AND по термам, не точная фраза — вопросы на естественном языке почти
    # никогда не совпадают с заголовком/абстрактом статьи дословно.
    terms = _keywords(query)[:max_terms]
    if not terms:
        # Запрос целиком из стоп-слов — отдаём как есть, пусть решает arXiv.
        return f"all:{query}" if query.strip() else "all:"
    return " AND ".join(f"all:{t}" for t in terms)


def _published_after(item: DiscoveredItem, cutoff: datetime) -> bool:
    if not item.published_date:
        return False
    try:
        published = datetime.fromisoformat(item.published_date.replace("Z", "+00:00"))
    except ValueError:
        return False
    return published >= cutoff
