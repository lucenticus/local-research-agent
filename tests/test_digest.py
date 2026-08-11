"""Юнит-тесты agent/digest.py — ArxivSource.recent() и LLM замоканы (офлайн)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.agent import digest
from src.sources.arxiv import ArxivSource
from src.sources.base import DiscoveredItem
from src.sources.citations import AuthorDetails, PaperDetails
from src.store.qdrant_store import published_timestamp


def _item(i: int, days_ago: float = 1.0) -> DiscoveredItem:
    published = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    return DiscoveredItem(
        id=f"arxiv:{i}", source="arxiv", title=f"Paper {i}", abstract=f"Abstract {i}",
        published_date=published, meta={"authors": [f"A. Uthor {i}"]},
    )


def test_run_digest_uses_defaults_when_not_specified(monkeypatch):
    captured = {}

    def fake_recent(self, days, limit, on_progress=None):
        captured["days"] = days
        captured["limit"] = limit
        captured["categories"] = self._categories
        return [_item(1)]

    monkeypatch.setattr(ArxivSource, "recent", fake_recent)
    monkeypatch.setattr(digest, "_summarize", lambda items: "summary text")

    result = digest.run_digest()
    assert captured["days"] == digest.config.DIGEST_DEFAULT_DAYS
    assert captured["limit"] == digest.config.DIGEST_DEFAULT_LIMIT
    assert captured["categories"] == digest.config.ARXIV_AI_CATEGORIES
    assert result.summary == "summary text"
    assert len(result.items) == 1


def test_run_digest_respects_explicit_overrides(monkeypatch):
    captured = {}

    def fake_recent(self, days, limit, on_progress=None):
        captured["days"] = days
        captured["limit"] = limit
        captured["categories"] = self._categories
        return []

    monkeypatch.setattr(ArxivSource, "recent", fake_recent)

    result = digest.run_digest(days=3, categories=["cs.CV"], limit=5)
    assert captured == {"days": 3, "limit": 5, "categories": ["cs.CV"]}
    assert result.days == 3
    assert result.categories == ["cs.CV"]


def test_run_digest_skips_summary_when_disabled(monkeypatch):
    monkeypatch.setattr(ArxivSource, "recent", lambda self, days, limit, on_progress=None: [_item(1)])

    def _fail(items):
        raise AssertionError("summarize must not be called when summarize=False")

    monkeypatch.setattr(digest, "_summarize", _fail)

    result = digest.run_digest(summarize=False)
    assert result.summary is None


def test_run_digest_skips_summary_when_no_items(monkeypatch):
    monkeypatch.setattr(ArxivSource, "recent", lambda self, days, limit, on_progress=None: [])

    def _fail(items):
        raise AssertionError("summarize must not be called on an empty digest")

    monkeypatch.setattr(digest, "_summarize", _fail)

    result = digest.run_digest()
    assert result.summary is None
    assert result.items == []


def test_run_digest_reports_progress(monkeypatch):
    monkeypatch.setattr(ArxivSource, "recent", lambda self, days, limit, on_progress=None: [_item(1)])
    monkeypatch.setattr(digest, "_summarize", lambda items: "summary")
    messages = []

    digest.run_digest(on_progress=messages.append)
    assert any("Ищем статьи" in m for m in messages)
    assert any("Найдено 1" in m for m in messages)
    assert any("обзор" in m for m in messages)


class _FakeStore:
    """Двойник QdrantStore для digest-пула: накопительный (add_chunks +
    has_source, как настоящий), search_hybrid отдаёт payload'ы накопленных
    чанков в исходном порядке, обрезанные до k, а load_pool/oldest_published_ts
    работают по тем же чанкам — как реальный поиск по payload'у."""

    instances: list["_FakeStore"] = []
    indexed: dict[str, object] = {}  # общий на все инстансы — переживает "рестарт"

    def __init__(self, collection_name=None):
        self.collection_name = collection_name
        self.searches = []
        self.added = []
        self.pool_loads = []
        _FakeStore.instances.append(self)

    def has_source(self, source_id):
        return source_id in _FakeStore.indexed

    def add_chunks(self, chunks):
        self.added.append(chunks)
        for chunk in chunks:
            _FakeStore.indexed[chunk.source_id] = chunk

    def search_hybrid(self, query_text, query_vector, k):
        self.searches.append({"query_text": query_text, "query_vector": query_vector, "k": k})
        chunks = list(_FakeStore.indexed.values())[:k]
        return [{"text": c.text, "source_id": c.source_id} for c in chunks]

    @staticmethod
    def _payload(chunk):
        return {
            "text": chunk.text,
            "source_id": chunk.source_id,
            "source_title": chunk.source_title,
            "url": chunk.url,
            "published_date": chunk.published_date,
            "authors": list(chunk.authors),
        }

    def load_pool(self, min_published_ts):
        self.pool_loads.append(min_published_ts)
        return [
            self._payload(chunk)
            for chunk in _FakeStore.indexed.values()
            if (published_timestamp(chunk.published_date) or 0) >= min_published_ts
        ]

    def oldest_published_ts(self):
        stamps = [
            ts
            for chunk in _FakeStore.indexed.values()
            if (ts := published_timestamp(chunk.published_date)) is not None
        ]
        return min(stamps) if stamps else None


def _patch_pool_ranking(monkeypatch):
    """Общая обвязка: подменяет эмбеддинги и QdrantStore на офлайн-двойники."""
    _FakeStore.instances = []
    _FakeStore.indexed = {}
    monkeypatch.setattr(digest, "QdrantStore", _FakeStore)
    monkeypatch.setattr(
        digest.embed, "embed_texts_hybrid",
        lambda texts: ([[0.1] * 3 for _ in texts], [{1: 0.5} for _ in texts]),
    )
    monkeypatch.setattr(digest.embed, "embed_texts", lambda texts: [[0.2] * 3 for _ in texts])


def test_run_digest_with_query_narrows_pool_by_hybrid_search_then_reranks(monkeypatch):
    """arXiv больше не видит query вообще (§ пользовательский запрос: некоторые
    формулировки с keyword AND'ом+категориями упирались в таймаут на стороне
    arXiv). Вместо этого широкий пул по категориям сужается гибридным поиском
    до DIGEST_QUERY_HYBRID_K, и только эти кандидаты идут в реранкер —
    реранкать весь пул из тысяч статей слишком медленно."""
    monkeypatch.setattr(digest.config, "DIGEST_QUERY_HYBRID_K", 3)
    monkeypatch.setattr(digest.config, "DIGEST_QUERY_TOP_K", 2)
    _patch_pool_ranking(monkeypatch)
    captured = {}
    # Явные даты: пул отдаётся отсортированным по свежести, а не в том
    # порядке, в каком его вернул источник.
    pool = [_item(i, days_ago=i) for i in (1, 2, 3, 4, 5)]

    def fake_recent(self, days, limit, on_progress=None, known_ids=None):
        captured["limit"] = limit  # без query-параметра вообще
        captured["known_ids"] = known_ids
        return pool

    def fake_rerank(query, candidates, top_n):
        captured["rerank_query"] = query
        captured["rerank_candidates"] = candidates
        captured["rerank_top_n"] = top_n
        return candidates[:top_n]

    monkeypatch.setattr(ArxivSource, "recent", fake_recent)
    monkeypatch.setattr(digest.rerank, "rerank", fake_rerank)
    monkeypatch.setattr(digest, "_summarize", lambda items: "summary")
    messages = []

    result = digest.run_digest(query="  diffusion models  ", on_progress=messages.append)

    assert captured["limit"] is None  # весь пул окна, без отдельного лимита
    assert captured["known_ids"] is None  # первый прогон — кэша ещё нет
    store = _FakeStore.instances[0]
    assert store.collection_name == digest.config.QDRANT_DIGEST_COLLECTION
    assert len(store.added[0]) == 5  # в индекс ушёл весь пул
    # Из поиска берём с запасом (коллекция накопительная), отсев до
    # DIGEST_QUERY_HYBRID_K — уже по текущему пулу
    assert store.searches[0]["k"] == 3 * digest._POOL_OVERFETCH
    assert store.searches[0]["query_text"] == "diffusion models"
    # Реранкер видит только то, что отдал гибридный поиск, а не весь пул
    assert captured["rerank_query"] == "diffusion models"  # обрезано
    assert captured["rerank_top_n"] == 2
    assert [c["item"] for c in captured["rerank_candidates"]] == pool[:3]
    assert result.query == "diffusion models"
    assert len(result.items) == 2  # обрезано реранком до DIGEST_QUERY_TOP_K
    assert any("«diffusion models»" in m for m in messages)
    assert any("Индексируем" in m for m in messages)
    assert any("Гибридный поиск" in m for m in messages)
    assert any("Реранкуем" in m for m in messages)


def test_collect_pool_first_run_fetches_whole_window(monkeypatch):
    """Кэша ещё нет — досрочно останавливать выгрузку нельзя, иначе срежется
    ещё не закэшированная нижняя часть окна."""
    _patch_pool_ranking(monkeypatch)
    captured = {}

    def fake_recent(self, days, limit, on_progress=None, known_ids=None):
        captured["known_ids"] = known_ids
        return [_item(1), _item(2)]

    monkeypatch.setattr(ArxivSource, "recent", fake_recent)

    items = digest._collect_pool(_FakeStore(), ["cs.AI"], days=7, on_progress=None)
    assert captured["known_ids"] is None  # полная выгрузка
    assert len(items) == 2


def test_collect_pool_reuses_cached_window_and_fetches_only_new(monkeypatch):
    """Главное, ради чего всё это (§ пользовательский запрос: не выкачивать
    одни и те же статьи по нескольку раз). Пул окна поднимается из индекса, а
    из arXiv добираются только статьи, появившиеся с прошлого прогона."""
    _patch_pool_ranking(monkeypatch)
    store = _FakeStore()
    # Прошлый прогон: окно закэшировано, включая статью старше его границы —
    # значит дно окна точно в индексе.
    cached = [_item(1, days_ago=1), _item(2, days_ago=3), _item(9, days_ago=30)]
    store.add_chunks(digest._pool_chunks(cached))

    captured = {}
    newcomer = _item(3, days_ago=0.1)

    def fake_recent(self, days, limit, on_progress=None, known_ids=None):
        captured["known_ids"] = known_ids
        return [newcomer]

    monkeypatch.setattr(ArxivSource, "recent", fake_recent)
    messages = []

    items = digest._collect_pool(_FakeStore(), ["cs.AI"], days=7, on_progress=messages.append)

    # Из arXiv спрашиваем только новое, передав всё уже известное
    assert captured["known_ids"] == {"arxiv:1", "arxiv:2"}  # статья вне окна не в счёт
    ids = [i.id for i in items]
    assert ids == ["arxiv:3", "arxiv:1", "arxiv:2"]  # свежие первыми, дубликатов нет
    # Закэшированные статьи восстановлены целиком, а не как заглушки
    restored = next(i for i in items if i.id == "arxiv:1")
    assert restored.title == "Paper 1"
    assert restored.abstract == "Abstract 1"
    assert restored.meta["authors"] == ["A. Uthor 1"]
    assert any("Из индекса подняты" in m for m in messages)


def test_collect_pool_refetches_window_when_cache_does_not_reach_its_bottom(monkeypatch):
    """Пользователь расширил --days: кэш покрывает только свежую часть окна,
    останавливаться на известных статьях нельзя — старое так и не подтянется."""
    _patch_pool_ranking(monkeypatch)
    store = _FakeStore()
    store.add_chunks(digest._pool_chunks([_item(1, days_ago=1)]))  # кэш только за вчера

    captured = {}

    def fake_recent(self, days, limit, on_progress=None, known_ids=None):
        captured["known_ids"] = known_ids
        return [_item(2, days_ago=20)]

    monkeypatch.setattr(ArxivSource, "recent", fake_recent)

    digest._collect_pool(_FakeStore(), ["cs.AI"], days=30, on_progress=None)
    assert captured["known_ids"] is None  # полная выгрузка, несмотря на непустой кэш


def test_rank_by_relevance_embeds_only_papers_not_already_indexed(monkeypatch):
    """Эмбеддинг пула — самая дорогая часть (~200с на 1000 абстрактов), а
    дайджест гоняется по одному и тому же недельному пулу с разными
    запросами: второй прогон не должен эмбеддить те же статьи заново."""
    _patch_pool_ranking(monkeypatch)
    monkeypatch.setattr(digest.rerank, "rerank", lambda query, candidates, top_n: candidates[:top_n])
    embedded = []
    real_pool_chunks = digest._pool_chunks
    monkeypatch.setattr(
        digest, "_pool_chunks",
        lambda items: (embedded.append([i.id for i in items]), real_pool_chunks(items))[1],
    )
    pool = [_item(1), _item(2)]

    digest._rank_by_relevance(_FakeStore(), "query", pool, top_k=2, on_progress=None)
    assert embedded == [["arxiv:1", "arxiv:2"]]  # первый прогон — весь пул

    digest._rank_by_relevance(_FakeStore(), "query", pool, top_k=2, on_progress=None)
    assert embedded == [["arxiv:1", "arxiv:2"]]  # второй — ничего нового

    digest._rank_by_relevance(_FakeStore(), "query", pool + [_item(3)], top_k=2, on_progress=None)
    assert embedded[-1] == ["arxiv:3"]  # только реально новая статья


def test_rank_by_relevance_drops_hits_outside_the_current_pool(monkeypatch):
    """Коллекция накопительная и держит статьи прошлых прогонов — дайджест
    обязан остаться в окне «за последние N дней», поэтому чужие хиты
    отсеиваются, а не подмешиваются в выдачу."""
    _patch_pool_ranking(monkeypatch)
    monkeypatch.setattr(digest.rerank, "rerank", lambda query, candidates, top_n: candidates[:top_n])

    stale = [_item(90), _item(91)]
    digest._rank_by_relevance(_FakeStore(), "query", stale, top_k=2, on_progress=None)  # осели в коллекции

    current = [_item(1)]
    result = digest._rank_by_relevance(_FakeStore(), "query", current, top_k=5, on_progress=None)
    assert [item.id for item in result] == ["arxiv:1"]  # ни одной из stale


def test_rank_by_relevance_maps_hits_back_to_items_by_source_id(monkeypatch):
    """search_hybrid возвращает payload'ы, не DiscoveredItem — сопоставление
    обратно идёт по source_id, и порядок берётся от поиска, не от пула."""
    monkeypatch.setattr(digest.config, "DIGEST_QUERY_HYBRID_K", 2)
    _patch_pool_ranking(monkeypatch)
    pool = [_item(1), _item(2), _item(3)]
    monkeypatch.setattr(digest.rerank, "rerank", lambda query, candidates, top_n: candidates[:top_n])

    result = digest._rank_by_relevance(_FakeStore(), "query", pool, top_k=2, on_progress=None)
    assert [item.id for item in result] == ["arxiv:1", "arxiv:2"]


def test_rank_by_relevance_returns_empty_when_hybrid_search_finds_nothing(monkeypatch):
    """Пустой результат поиска не должен доходить до реранкера (§1: незачем
    грузить модель, чтобы отскорить ноль кандидатов)."""
    _patch_pool_ranking(monkeypatch)
    monkeypatch.setattr(_FakeStore, "search_hybrid", lambda self, query_text, query_vector, k: [])

    def _fail(query, candidates, top_n):
        raise AssertionError("rerank must not run on an empty candidate list")

    monkeypatch.setattr(digest.rerank, "rerank", _fail)

    assert digest._rank_by_relevance(_FakeStore(), "query", [_item(1)], top_k=5, on_progress=None) == []


def test_pool_chunks_carry_item_identity_and_hybrid_vectors(monkeypatch):
    _patch_pool_ranking(monkeypatch)
    chunks = digest._pool_chunks([_item(1)])
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.source_id == "arxiv:1"  # по нему hit'ы мапятся обратно в items
    assert chunk.source_title == "Paper 1"
    assert "Paper 1" in chunk.text and "Abstract 1" in chunk.text
    assert chunk.vector == [0.1] * 3
    assert chunk.sparse == {1: 0.5}  # sparse-сторона гибридного поиска заполнена


def test_run_digest_blank_query_becomes_none_and_skips_ranking(monkeypatch):
    captured = {"ranked": False}

    def _fail_rank(query, items, top_k, on_progress):
        captured["ranked"] = True
        return items

    monkeypatch.setattr(ArxivSource, "recent", lambda self, days, limit, on_progress=None: [_item(1)])
    monkeypatch.setattr(digest, "_rank_by_relevance", _fail_rank)
    monkeypatch.setattr(digest, "_summarize", lambda items: "summary")

    result = digest.run_digest(query="   ")
    assert result.query is None
    assert captured["ranked"] is False


def test_run_digest_skips_analysis_by_default(monkeypatch):
    monkeypatch.setattr(ArxivSource, "recent", lambda self, days, limit, on_progress=None: [_item(1)])
    monkeypatch.setattr(digest, "_summarize", lambda items: "summary")

    def _fail(item):
        raise AssertionError("_summarize_item must not be called when deep=False")

    monkeypatch.setattr(digest, "_summarize_item", _fail)

    result = digest.run_digest()
    assert result.analyses == {}


def test_run_digest_deep_analyzes_each_item(monkeypatch):
    items = [_item(1), _item(2)]
    monkeypatch.setattr(ArxivSource, "recent", lambda self, days, limit, on_progress=None: items)
    monkeypatch.setattr(digest, "_summarize", lambda items: "summary")
    monkeypatch.setattr(digest, "_summarize_item", lambda item: f"резюме {item.title}")

    details = PaperDetails(citation_count=5, venue="NeurIPS", authors=[AuthorDetails(name="A. Uthor")])
    monkeypatch.setattr(digest, "lookup_paper_details", lambda title: details)

    messages = []
    result = digest.run_digest(deep=True, on_progress=messages.append)

    assert set(result.analyses.keys()) == {"arxiv:1", "arxiv:2"}
    assert result.analyses["arxiv:1"].summary_ru == "резюме Paper 1"
    assert result.analyses["arxiv:1"].details is details
    assert any("Анализируем статью 1/2" in m for m in messages)
    assert any("Анализируем статью 2/2" in m for m in messages)


def test_run_digest_deep_handles_unindexed_paper(monkeypatch):
    monkeypatch.setattr(ArxivSource, "recent", lambda self, days, limit, on_progress=None: [_item(1)])
    monkeypatch.setattr(digest, "_summarize", lambda items: "summary")
    monkeypatch.setattr(digest, "_summarize_item", lambda item: "резюме")
    monkeypatch.setattr(digest, "lookup_paper_details", lambda title: None)

    result = digest.run_digest(deep=True)
    assert result.analyses["arxiv:1"].details is None
    assert result.analyses["arxiv:1"].summary_ru == "резюме"


def test_run_digest_deep_caps_at_deep_max_items(monkeypatch):
    monkeypatch.setattr(digest.config, "DIGEST_DEEP_MAX_ITEMS", 2)
    items = [_item(i) for i in range(5)]
    monkeypatch.setattr(ArxivSource, "recent", lambda self, days, limit, on_progress=None: items)
    monkeypatch.setattr(digest, "_summarize", lambda items: "summary")
    monkeypatch.setattr(digest, "_summarize_item", lambda item: "резюме")
    monkeypatch.setattr(digest, "lookup_paper_details", lambda title: None)

    result = digest.run_digest(deep=True)
    assert len(result.items) == 5  # список статей не урезан
    assert len(result.analyses) == 2  # но проанализированы только первые DIGEST_DEEP_MAX_ITEMS


def test_summarize_item_builds_prompt_from_title_and_abstract(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        digest.llm, "build_chat_prompt", lambda system, user: (captured.setdefault("user", user), "prompt")[1]
    )
    monkeypatch.setattr(digest.llm, "generate", lambda prompt, **kw: "  резюме статьи  ")

    result = digest._summarize_item(_item(1))
    assert result == "резюме статьи"
    assert "Paper 1" in captured["user"]
    assert "Abstract 1" in captured["user"]


def test_summarize_builds_prompt_from_titles_and_abstracts(monkeypatch):
    captured = {}
    monkeypatch.setattr(digest.llm, "build_chat_prompt", lambda system, user: (captured.setdefault("user", user), "prompt")[1])
    monkeypatch.setattr(digest.llm, "generate", lambda prompt, **kw: "  overview text  ")

    result = digest._summarize([_item(1), _item(2)])
    assert result == "overview text"
    assert "Paper 1" in captured["user"]
    assert "Paper 2" in captured["user"]


def _section(name, category, text):
    from src.ingest.extract import Section
    return Section(name=name, category=category, text=text)


def test_code_links_keeps_only_code_hosts_and_dedupes():
    """Ссылки на репозитории берутся из PDF детерминированно, а не у LLM:
    правдоподобный выдуманный URL — недопустимый вид галлюцинации."""
    links = digest._code_links([
        "https://github.com/acme/repo",
        "https://arxiv.org/abs/2608.01234",       # цитата, не исходники
        "https://doi.org/10.1000/xyz",            # тоже цитата
        "https://huggingface.co/acme/model",
        "https://www.github.com/acme/repo2",      # www нормализуется
        "https://github.com/acme/repo",           # дубликат
        "https://example.com/blog",
        "https://github.com/",                    # голый хост — разорванный URL, не код
    ])
    assert [(l.kind, l.url) for l in links] == [
        ("github", "https://github.com/acme/repo"),
        ("huggingface", "https://huggingface.co/acme/model"),
        ("github", "https://github.com/acme/repo2"),
    ]


def test_insight_context_prefers_results_and_respects_the_char_budget(monkeypatch):
    """Полный текст статьи в контекст не влезает и не должен: раздутый
    контекст надувает KV-кэш (§1 CLAUDE.md)."""
    monkeypatch.setattr(digest.config, "DIGEST_PDF_CONTEXT_CHARS", 30)
    sections = [
        _section("Introduction", "introduction", "I" * 100),
        _section("Results", "results", "R" * 20),
        _section("Conclusion", "conclusion", "C" * 100),
    ]
    context, used = digest._insight_context(sections)

    assert used[0] == "Results"            # результаты вперёд введения
    assert "Introduction" not in used      # бюджет кончился раньше
    assert len(context) <= 30 + len("## Results\n") + len("## Conclusion\n") + 2


def test_insight_context_empty_when_no_sections_parsed():
    assert digest._insight_context([]) == ("", [])


def test_pdf_url_falls_back_to_arxiv_id_for_cached_items():
    """Статьи из кэша восстанавливаются без meta — а это теперь основной
    путь, так что без фолбэка глубокий анализ ломался бы почти всегда."""
    fresh = DiscoveredItem(id="arxiv:2608.1v1", source="arxiv", title="t", abstract="a",
                           meta={"pdf_url": "https://example.com/x.pdf"})
    cached = DiscoveredItem(id="arxiv:2608.1v1", source="arxiv", title="t", abstract="a")
    other = DiscoveredItem(id="web:http://x", source="web", title="t", abstract="a")

    assert digest._pdf_url(fresh) == "https://example.com/x.pdf"
    assert digest._pdf_url(cached) == "https://arxiv.org/pdf/2608.1v1"
    assert digest._pdf_url(other) is None


def test_analyze_pdf_returns_insights_from_sections_and_links(monkeypatch):
    from src.sources.pdf import FetchedPdf

    monkeypatch.setattr(digest.config, "DIGEST_PDF_CONTEXT_CHARS", 5000)
    captured = {}
    monkeypatch.setattr(
        digest, "fetch_pdf",
        lambda url: (captured.setdefault("url", url), FetchedPdf(
            sections=[_section("Results", "results", "We beat the baseline by 3 points.")],
            links=["https://github.com/acme/repo", "https://arxiv.org/abs/1"],
        ))[1],
    )
    monkeypatch.setattr(digest.llm, "build_chat_prompt", lambda system, user: user)
    monkeypatch.setattr(digest.llm, "generate", lambda prompt, **kw: "  Результаты: +3.  ")

    insights = digest._analyze_pdf(_item(1))
    assert captured["url"] == "https://arxiv.org/pdf/1"
    assert insights.findings_ru == "Результаты: +3."
    assert insights.sections_used == ["Results"]
    assert [l.url for l in insights.code_links] == ["https://github.com/acme/repo"]


def test_analyze_pdf_returns_none_when_pdf_unavailable(monkeypatch):
    def _boom(url):
        raise OSError("404")

    monkeypatch.setattr(digest, "fetch_pdf", _boom)
    assert digest._analyze_pdf(_item(1)) is None


def test_analyze_pdf_returns_none_when_no_sections_parsed(monkeypatch):
    """Двухколоночная вёрстка/нестандартные заголовки — секции не нашлись.
    Честнее показать "не разобрали", чем разбор неизвестно чего."""
    from src.sources.pdf import FetchedPdf

    monkeypatch.setattr(digest, "fetch_pdf", lambda url: FetchedPdf(sections=[], links=[]))

    def _fail(*a, **kw):
        raise AssertionError("LLM must not be called without any parsed section")

    monkeypatch.setattr(digest.llm, "generate", _fail)
    assert digest._analyze_pdf(_item(1)) is None


def test_analyze_item_includes_pdf_insights(monkeypatch):
    monkeypatch.setattr(digest, "_summarize_item", lambda item: "саммари")
    monkeypatch.setattr(digest, "lookup_paper_details", lambda title: None)
    insights = digest.PaperInsights(findings_ru="разбор", code_links=[], sections_used=["Results"])
    monkeypatch.setattr(digest, "_analyze_pdf", lambda item: insights)

    messages = []
    analysis = digest.analyze_item(_item(1), on_progress=messages.append)
    assert analysis.insights is insights
    assert any("PDF" in m for m in messages)


def test_analyze_item_skips_pdf_when_disabled(monkeypatch):
    monkeypatch.setattr(digest.config, "DIGEST_PDF_ANALYSIS", False)
    monkeypatch.setattr(digest, "_summarize_item", lambda item: "саммари")
    monkeypatch.setattr(digest, "lookup_paper_details", lambda title: None)

    def _fail(item):
        raise AssertionError("_analyze_pdf must not run when DIGEST_PDF_ANALYSIS is off")

    monkeypatch.setattr(digest, "_analyze_pdf", _fail)
    assert digest.analyze_item(_item(1)).insights is None


def test_code_links_drops_truncated_prefixes_of_a_longer_link():
    """В реальном PDF рядом с полной ссылкой на репозиторий попадался её
    огрызок, разорванный вёрсткой — показывать оба некрасиво и бесполезно."""
    links = digest._code_links([
        "https://github.com/acme/DistMoE",
        "https://github.com/acme/",          # префикс первой — уходит
        "https://github.com/other/repo",     # самостоятельная ссылка — остаётся
    ])
    assert [l.url for l in links] == [
        "https://github.com/acme/DistMoE",
        "https://github.com/other/repo",
    ]


def test_parse_authors_reads_name_affiliation_lines():
    parsed = digest._parse_authors(
        "Ashish Vaswani — Google Brain\n"
        "- Noam Shazeer — Google Brain\n"
        "Illia Polosukhin — —\n"          # аффилиации в шапке не было
    )
    assert [(a.name, a.affiliation) for a in parsed] == [
        ("Ashish Vaswani", "Google Brain"),
        ("Noam Shazeer", "Google Brain"),
        ("Illia Polosukhin", None),
    ]


def test_parse_authors_ignores_lines_without_a_separator():
    """Вводные фразы модели ("Вот список авторов:") не должны становиться
    несуществующими авторами."""
    parsed = digest._parse_authors("Вот список авторов:\nJane Doe — MIT\nСпасибо!")
    assert [a.name for a in parsed] == ["Jane Doe"]


def test_authors_from_header_skips_llm_on_empty_header(monkeypatch):
    def _fail(*a, **kw):
        raise AssertionError("нет шапки — незачем звать LLM")

    monkeypatch.setattr(digest.llm, "generate", _fail)
    assert digest._authors_from_header("   ") == []


def test_with_stars_only_queries_github_links(monkeypatch):
    asked = []
    monkeypatch.setattr(
        digest, "lookup_stars", lambda url: (asked.append(url), 42)[1]
    )
    links = [
        digest.CodeLink(url="https://github.com/acme/repo", kind="github"),
        digest.CodeLink(url="https://huggingface.co/acme/model", kind="huggingface"),
    ]
    result = digest._with_stars(links)

    assert asked == ["https://github.com/acme/repo"]  # HF не дёргаем
    assert result[0].stars == 42
    assert result[1].stars is None


def test_analyze_pdf_fills_authors_and_stars(monkeypatch):
    from src.sources.pdf import FetchedPdf

    monkeypatch.setattr(digest.config, "DIGEST_PDF_CONTEXT_CHARS", 5000)
    monkeypatch.setattr(
        digest, "fetch_pdf",
        lambda url: FetchedPdf(
            sections=[_section("Results", "results", "Beats the baseline.")],
            links=["https://github.com/acme/repo"],
            header="Paper\nJane Doe1\n1MIT",
        ),
    )
    monkeypatch.setattr(digest.llm, "build_chat_prompt", lambda system, user: user)
    monkeypatch.setattr(
        digest.llm, "generate",
        lambda prompt, **kw: "Jane Doe — MIT" if "1MIT" in prompt else "разбор",
    )
    monkeypatch.setattr(digest, "lookup_stars", lambda url: 7)

    insights = digest._analyze_pdf(_item(1))
    assert [(a.name, a.affiliation) for a in insights.authors] == [("Jane Doe", "MIT")]
    assert insights.code_links[0].stars == 7


def test_code_links_drops_references_to_someone_elses_changes():
    """Поймано на статье про бенчмарк: в её PDF десятки ссылок на PR'ы и
    коммиты в чужих репозиториях — это её датасет, а не исходники статьи."""
    links = digest._code_links([
        "https://github.com/authors/our-repo",
        "https://github.com/nasa/fprime/pull/3422",
        "https://github.com/plantuml/plantuml/commit/c910f8b",
        "https://github.com/some/proj/issues/12",
        "https://huggingface.co/datasets/team/our-dataset",
    ])
    assert [l.url for l in links] == [
        "https://github.com/authors/our-repo",
        "https://huggingface.co/datasets/team/our-dataset",  # HF-путь длиннее — это ресурс, не цитата
    ]


def test_code_links_normalizes_deep_repo_paths_to_the_repo_root():
    links = digest._code_links([
        "https://github.com/acme/repo/tree/main/src",
        "https://github.com/acme/repo",  # то же самое место
    ])
    assert [l.url for l in links] == ["https://github.com/acme/repo"]


def test_authors_prompt_shows_both_real_header_layouts():
    """4B-модель на голой инструкции теряла общую для всех авторов
    аффилиацию, стоящую отдельной строкой (проверено на реальной статье) —
    промпт обязан нести примеры обоих форматов шапки."""
    prompt = digest._AUTHORS_SYSTEM_PROMPT
    assert "Tel Aviv University" in prompt   # общая аффилиация на всех
    assert "1Singapore Management University" in prompt  # номера-сноски
