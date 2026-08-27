"""Юнит-тесты sources/arxiv.py — HTTP замокан (офлайн). Раньше не было ни
одного теста на этот источник."""

from __future__ import annotations

import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

import pytest

from src.sources import arxiv as arxiv_module
from src.sources.arxiv import ArxivSource

_ATOM_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2508.12345v1</id>
    <title>Attention Is All You Need Again</title>
    <summary>We revisit attention mechanisms for transformers.</summary>
    <published>2026-08-01T12:00:00Z</published>
  </entry>
</feed>
"""


class _FakeResponse:
    def __init__(self, body: str):
        self._body = body.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


def test_discover_parses_entry_including_published_date(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout=None: _FakeResponse(_ATOM_TEMPLATE))

    items = ArxivSource().discover("attention", limit=5)
    assert len(items) == 1
    item = items[0]
    assert item.id == "arxiv:2508.12345v1"
    assert item.title == "Attention Is All You Need Again"
    assert item.abstract == "We revisit attention mechanisms for transformers."
    assert item.url == "https://arxiv.org/abs/2508.12345v1"
    assert item.year == 2026
    assert item.published_date == "2026-08-01T12:00:00Z"
    assert item.citation_count is None
    assert item.meta["pdf_url"] == "https://arxiv.org/pdf/2508.12345v1"
    assert item.source == "arxiv"


def test_discover_without_categories_sends_plain_keyword_query(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        return _FakeResponse(_ATOM_TEMPLATE)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    ArxivSource().discover("attention transformers", limit=5)
    assert "search_query=all%3Aattention+AND+all%3Atransformers" in captured["url"]
    assert "cat%3A" not in captured["url"]


def test_discover_with_categories_ands_category_filter_into_query(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        return _FakeResponse(_ATOM_TEMPLATE)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    ArxivSource(categories=["cs.AI", "cs.LG"]).discover("attention", limit=5)
    url = captured["url"]
    assert "cat%3Acs.AI+OR+cat%3Acs.LG" in url
    assert "all%3Aattention" in url


def test_discover_handles_missing_published_date():
    from src.sources.arxiv import ArxivSource as _AS

    body = _ATOM_TEMPLATE.replace(
        "<published>2026-08-01T12:00:00Z</published>", ""
    ).encode("utf-8")
    items = list(_AS()._parse(body))
    assert items[0].year is None
    assert items[0].published_date is None


def test_parse_extracts_authors_and_categories():
    body = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2508.99999v1</id>
    <title>Some Paper</title>
    <summary>An abstract.</summary>
    <published>2026-08-01T12:00:00Z</published>
    <category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.AI" scheme="http://arxiv.org/schemas/atom"/>
    <author><name>Alice Example</name></author>
    <author><name>Bob Example</name></author>
  </entry>
</feed>
""".encode("utf-8")
    item = list(ArxivSource()._parse(body))[0]
    assert item.meta["authors"] == ["Alice Example", "Bob Example"]
    assert item.meta["categories"] == ["cs.CL", "cs.AI"]


def _atom_with_published(*dates: str) -> str:
    entries = "".join(
        f"""<entry>
    <id>http://arxiv.org/abs/2508.{i:05d}v1</id>
    <title>Paper {i}</title>
    <summary>Abstract {i}.</summary>
    <published>{date}</published>
  </entry>"""
        for i, date in enumerate(dates)
    )
    return f'<?xml version="1.0" encoding="UTF-8"?>\n<feed xmlns="http://www.w3.org/2005/Atom">{entries}</feed>'


def test_recent_requires_categories():
    with pytest.raises(ValueError):
        ArxivSource().recent(days=7, limit=10)


def test_recent_sorts_by_submitted_date_not_relevance(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        return _FakeResponse(_atom_with_published(datetime.now(timezone.utc).isoformat()))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    ArxivSource(categories=["cs.AI"]).recent(days=7, limit=10)
    assert "sortBy=submittedDate" in captured["url"]
    assert "sortOrder=descending" in captured["url"]
    assert "all%3A" not in captured["url"]  # без keyword-фильтра — только категории
    assert "cat%3Acs.AI" in captured["url"]


def test_recent_filters_out_items_older_than_days(monkeypatch):
    fresh = datetime.now(timezone.utc)
    old = fresh - timedelta(days=30)
    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda request, timeout=None: _FakeResponse(_atom_with_published(fresh.isoformat(), old.isoformat())),
    )

    items = ArxivSource(categories=["cs.AI"]).recent(days=7, limit=10)
    assert len(items) == 1
    assert items[0].title == "Paper 0"


def test_recent_never_sends_keyword_clause(monkeypatch):
    """recent() больше не принимает query (см. docstring модуля) — весь пул
    всегда только по категориям, релевантность запросу ранжируется локально
    в digest.py через reranker, а не через arXiv-side AND."""
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        return _FakeResponse(_atom_with_published(datetime.now(timezone.utc).isoformat()))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    ArxivSource(categories=["cs.AI"]).recent(days=7, limit=10)
    assert "all%3A" not in captured["url"]


def _page_of(n: int, when: datetime, start_index: int = 0) -> str:
    """Страница из n записей с одинаковой датой `when`."""
    entries = "".join(
        f"""<entry>
    <id>http://arxiv.org/abs/2508.{start_index + i:05d}v1</id>
    <title>Paper {start_index + i}</title>
    <summary>Abstract.</summary>
    <published>{when.isoformat()}</published>
  </entry>"""
        for i in range(n)
    )
    return f'<?xml version="1.0" encoding="UTF-8"?>\n<feed xmlns="http://www.w3.org/2005/Atom">{entries}</feed>'


def test_recent_without_limit_pages_until_the_date_window_closes(monkeypatch):
    """limit=None — выгружаем всё окно постранично (§ пользовательский запрос:
    не ограничивать, выгружать все статьи за период). Выдача отсортирована по
    дате, поэтому первая статья старше cutoff означает "дальше только старее"
    и пагинация останавливается — range-синтаксис в запросе не нужен."""
    monkeypatch.setattr(arxiv_module, "_PAGE_SIZE", 2)
    monkeypatch.setattr(arxiv_module.time, "sleep", lambda s: None)
    fresh = datetime.now(timezone.utc)
    old = fresh - timedelta(days=30)
    # 2 полные свежие страницы, затем страница, где окно закрывается
    pages = [
        _page_of(2, fresh, 0),
        _page_of(2, fresh, 2),
        _atom_with_published(fresh.isoformat(), old.isoformat()),
    ]
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(request.full_url)
        return _FakeResponse(pages[len(calls) - 1])

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    items = ArxivSource(categories=["cs.AI"]).recent(days=7, limit=None)
    assert len(calls) == 3  # остановились сами, а не по лимиту
    assert [c.split("start=")[1].split("&")[0] for c in calls] == ["0", "2", "4"]
    assert len(items) == 5  # 2 + 2 + 1 свежая; статья старше cutoff отброшена


def test_recent_without_limit_stops_on_a_short_page(monkeypatch):
    """arXiv отдал меньше запрошенного — дальше страниц нет, второй запрос
    был бы впустую."""
    monkeypatch.setattr(arxiv_module, "_PAGE_SIZE", 5)
    monkeypatch.setattr(arxiv_module.time, "sleep", lambda s: None)
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(request.full_url)
        return _FakeResponse(_page_of(2, datetime.now(timezone.utc)))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    items = ArxivSource(categories=["cs.AI"]).recent(days=7, limit=None)
    assert len(calls) == 1
    assert len(items) == 2


def test_recent_without_limit_reports_running_total(monkeypatch):
    """Выгрузка окна — несколько запросов с паузами между ними; без
    промежуточных апдейтов это выглядит как зависший процесс."""
    monkeypatch.setattr(arxiv_module, "_PAGE_SIZE", 2)
    monkeypatch.setattr(arxiv_module.time, "sleep", lambda s: None)
    fresh = datetime.now(timezone.utc)
    pages = [_page_of(2, fresh, 0), _page_of(1, fresh, 2)]
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(1)
        return _FakeResponse(pages[len(calls) - 1])

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    seen = []
    ArxivSource(categories=["cs.AI"]).recent(days=7, limit=None, on_progress=seen.append)
    assert seen == [2, 3]  # накопительный счётчик после каждой страницы


def test_recent_with_limit_does_not_overshoot_it(monkeypatch):
    monkeypatch.setattr(arxiv_module, "_PAGE_SIZE", 2)
    monkeypatch.setattr(arxiv_module.time, "sleep", lambda s: None)
    fresh = datetime.now(timezone.utc)
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(request.full_url)
        return _FakeResponse(_page_of(2, fresh, len(calls) * 2))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    items = ArxivSource(categories=["cs.AI"]).recent(days=7, limit=3)
    assert len(items) == 3
    # Последняя страница запрошена ровно на остаток, а не на полный _PAGE_SIZE
    assert "max_results=1" in calls[-1]


def test_recent_stops_once_a_page_is_entirely_known(monkeypatch):
    """Инкрементальный режим (§ пользовательский запрос: не выкачивать одни и
    те же статьи по нескольку раз). Новые статьи по дате-desc всегда в начале
    выдачи, поэтому первая целиком известная страница = догнали кэш."""
    monkeypatch.setattr(arxiv_module, "_PAGE_SIZE", 2)
    monkeypatch.setattr(arxiv_module.time, "sleep", lambda s: None)
    fresh = datetime.now(timezone.utc)
    # Страница 1 — новые, страница 2 — уже известные, страницы 3 быть не должно
    pages = [_page_of(2, fresh, 0), _page_of(2, fresh, 2), _page_of(2, fresh, 4)]
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(1)
        return _FakeResponse(pages[len(calls) - 1])

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    known = {"arxiv:2508.00002v1", "arxiv:2508.00003v1"}
    items = ArxivSource(categories=["cs.AI"]).recent(days=7, limit=None, known_ids=known)

    assert len(calls) == 2  # третья страница не запрашивалась
    # Возвращается только то, чего не было в кэше — остальное у вызывающего есть
    assert [i.id for i in items] == ["arxiv:2508.00000v1", "arxiv:2508.00001v1"]


def test_recent_without_known_ids_does_not_stop_early(monkeypatch):
    monkeypatch.setattr(arxiv_module, "_PAGE_SIZE", 2)
    monkeypatch.setattr(arxiv_module.time, "sleep", lambda s: None)
    fresh = datetime.now(timezone.utc)
    old = fresh - timedelta(days=30)
    pages = [_page_of(2, fresh, 0), _atom_with_published(fresh.isoformat(), old.isoformat())]
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(1)
        return _FakeResponse(pages[len(calls) - 1])

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    items = ArxivSource(categories=["cs.AI"]).recent(days=7, limit=None)
    assert len(calls) == 2  # шли до границы окна, а не до знакомых статей
    assert len(items) == 3


def test_fetch_retries_on_429_then_succeeds(monkeypatch):
    """429 при выгрузке всего окна ловится реально, не гипотетически — без
    ретрая дайджест просто падает."""
    slept = []
    monkeypatch.setattr(arxiv_module.time, "sleep", slept.append)
    attempts = []

    def fake_urlopen(request, timeout=None):
        attempts.append(1)
        if len(attempts) < 3:
            raise urllib.error.HTTPError(request.full_url, 429, "Too Many Requests", None, None)
        return _FakeResponse(_atom_with_published(datetime.now(timezone.utc).isoformat()))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    items = ArxivSource(categories=["cs.AI"]).recent(days=7, limit=10)
    assert len(attempts) == 3
    assert len(items) == 1
    base = arxiv_module._RETRY_BACKOFF_SECONDS
    assert slept == [base, base * 2]  # экспоненциальный backoff


def test_fetch_gives_up_after_retry_attempts(monkeypatch):
    monkeypatch.setattr(arxiv_module.time, "sleep", lambda s: None)
    attempts = []

    def fake_urlopen(request, timeout=None):
        attempts.append(1)
        raise urllib.error.HTTPError(request.full_url, 429, "Too Many Requests", None, None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(urllib.error.HTTPError):
        ArxivSource(categories=["cs.AI"]).recent(days=7, limit=10)
    assert len(attempts) == arxiv_module._RETRY_ATTEMPTS


def test_fetch_retries_on_read_timeout(monkeypatch):
    """Таймаут чтения на крупной странице — реальный сценарий (подтверждён
    прогоном на 500 записей), не только 429."""
    monkeypatch.setattr(arxiv_module.time, "sleep", lambda s: None)
    attempts = []

    def fake_urlopen(request, timeout=None):
        attempts.append(1)
        if len(attempts) < 2:
            raise TimeoutError("The read operation timed out")
        return _FakeResponse(_atom_with_published(datetime.now(timezone.utc).isoformat()))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    items = ArxivSource(categories=["cs.AI"]).recent(days=7, limit=10)
    assert len(attempts) == 2
    assert len(items) == 1


def test_fetch_does_not_retry_on_404(monkeypatch):
    """4xx кроме 429 — не временная ошибка, ретраить нечего."""
    monkeypatch.setattr(arxiv_module.time, "sleep", lambda s: None)
    attempts = []

    def fake_urlopen(request, timeout=None):
        attempts.append(1)
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", None, None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(urllib.error.HTTPError):
        ArxivSource(categories=["cs.AI"]).recent(days=7, limit=10)
    assert len(attempts) == 1


def test_keywords_drops_stopwords_punctuation_and_duplicates():
    """Живой дефект: вопрос уходил в arXiv дословно, каждое слово через AND —
    включая "What", "for", "in" и приклеенный "?". Реальная проверка дала
    0 результатов против 630 у запроса из одних ключевых слов."""
    terms = arxiv_module._keywords("What approaches exist for KV-cache compression in transformers?")
    assert terms == ["KV-cache", "compression", "transformers"]


def test_keywords_preserves_order_and_case_of_content_words():
    assert arxiv_module._keywords("How does routing work in Mixture-of-Experts models?") == [
        "routing", "Mixture-of-Experts", "models",
    ]


def test_keyword_query_caps_the_number_of_terms():
    query = "alpha beta gamma delta epsilon zeta eta theta"
    assert arxiv_module._keyword_query(query).count(" AND ") == arxiv_module._MAX_QUERY_TERMS - 1
    assert arxiv_module._keyword_query(query, 2) == "all:alpha AND all:beta"


def test_keyword_query_falls_back_when_everything_is_a_stopword():
    """Иначе запрос из одних стоп-слов превратился бы в пустой AND."""
    assert arxiv_module._keyword_query("what is the") == "all:what is the"


def test_discover_retries_with_a_shorter_query_when_the_first_one_is_empty(monkeypatch):
    """Каждый лишний терм в AND сужает выдачу arXiv до нуля — на пустом
    ответе пробуем ещё раз более широким запросом, а не отдаём воронке
    незакрытый подвопрос."""
    sent = []
    empty = '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'

    def fake_urlopen(request, timeout=None):
        sent.append(request.full_url)
        return _FakeResponse(empty if len(sent) == 1 else _ATOM_TEMPLATE)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    items = ArxivSource().discover("alpha beta gamma delta epsilon zeta", limit=5)
    assert len(sent) == 2
    assert sent[0].count("all%3A") == arxiv_module._MAX_QUERY_TERMS
    assert sent[1].count("all%3A") == arxiv_module._RETRY_QUERY_TERMS
    assert len(items) == 1  # результат пришёл со второй попытки


def test_discover_does_not_retry_when_the_first_query_found_something(monkeypatch):
    sent = []

    def fake_urlopen(request, timeout=None):
        sent.append(request.full_url)
        return _FakeResponse(_ATOM_TEMPLATE)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    ArxivSource().discover("alpha beta gamma delta epsilon zeta", limit=5)
    assert len(sent) == 1


def test_discover_does_not_retry_a_query_that_is_already_short(monkeypatch):
    """Ужимать нечего — второй запрос был бы точной копией первого."""
    sent = []
    empty = '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'

    def fake_urlopen(request, timeout=None):
        sent.append(request.full_url)
        return _FakeResponse(empty)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    assert ArxivSource().discover("alpha beta", limit=5) == []
    assert len(sent) == 1
