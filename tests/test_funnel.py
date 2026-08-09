"""Юнит-тесты agent/funnel.py — источники, embed и LanceDB замоканы (офлайн).

Кандидаты в тестах не несут `pdf_url` в meta, поэтому deep read всегда идёт
по fallback-ветке (abstract как единственный chunk) — без сетевых вызовов.
"""

from __future__ import annotations

from src.agent import funnel
from src.agent.state import Budget, Candidate, ResearchState, SubQuestion
from src.providers import embed, llm, mcp_client
from src.sources.base import DiscoveredItem


class _FakeSource:
    def __init__(self, name, items, raise_error=False):
        self.name = name
        self._items = items
        self._raise_error = raise_error

    def discover(self, query, limit):
        if self._raise_error:
            raise RuntimeError("source down")
        return self._items[:limit]


class _FakeStore:
    def __init__(self, already_indexed_ids=None):
        self.added_chunks = []
        self._already_indexed = set(already_indexed_ids or [])

    def has_source(self, source_id):
        return source_id in self._already_indexed

    def add_chunks(self, chunks):
        self.added_chunks.extend(chunks)


def _item(id_, title, abstract):
    return DiscoveredItem(id=id_, source="test", title=title, abstract=abstract)


def _mock_embed(monkeypatch, vector_by_keyword):
    def fake_embed_texts(texts):
        vectors = []
        for text in texts:
            for keyword, vector in vector_by_keyword.items():
                if keyword in text:
                    vectors.append(vector)
                    break
            else:
                vectors.append([0.0, 0.0])
        return vectors

    monkeypatch.setattr(embed, "embed_texts", fake_embed_texts)


def test_discovery_query_passes_through_english_text():
    assert funnel._discovery_query("cats and dogs") == "cats and dogs"


def test_discovery_query_translates_non_english_text(monkeypatch):
    monkeypatch.setattr(llm, "build_chat_prompt", lambda system, user: user)
    monkeypatch.setattr(llm, "generate", lambda prompt, **kw: "hybrid retrieval RRF")

    assert funnel._discovery_query("Что такое гибридный поиск?") == "hybrid retrieval RRF"


def test_discover_translates_query_before_calling_sources(monkeypatch):
    monkeypatch.setattr(llm, "build_chat_prompt", lambda system, user: user)
    monkeypatch.setattr(llm, "generate", lambda prompt, **kw: "hybrid retrieval")

    seen_queries = []

    class _RecordingSource:
        name = "s"

        def discover(self, query, limit):
            seen_queries.append(query)
            return []

    funnel._discover(SubQuestion(text="Что такое гибридный поиск?"), [_RecordingSource()], discovery_limit=5)
    assert seen_queries == ["hybrid retrieval"]


def test_discover_collects_from_multiple_sources_and_skips_failing_ones():
    sq = SubQuestion(text="cats")
    good = _FakeSource("good", [_item("a", "A", "about cats")])
    bad = _FakeSource("bad", [], raise_error=True)

    candidates = funnel._discover(sq, [good, bad], discovery_limit=5)
    assert [c.id for c in candidates] == ["a"]


def test_triage_keeps_top_n_by_cosine_similarity(monkeypatch):
    monkeypatch.setattr(funnel.config, "FUNNEL_TRIAGE_TOP_N", 1)
    _mock_embed(monkeypatch, {"cats": [1.0, 0.0], "dogs": [0.0, 1.0]})

    sq = SubQuestion(text="cats?")
    candidates = [
        Candidate(id="a", source="s", title="A", abstract="about dogs"),
        Candidate(id="b", source="s", title="B", abstract="about cats"),
    ]

    survivors = funnel._triage(sq, candidates)
    assert [c.id for c in survivors] == ["b"]
    assert survivors[0].triage_score == 1.0


def test_triage_skips_candidates_without_abstract(monkeypatch):
    _mock_embed(monkeypatch, {"cats": [1.0, 0.0]})
    sq = SubQuestion(text="cats?")
    candidates = [Candidate(id="a", source="s", title="A", abstract="   ")]

    assert funnel._triage(sq, candidates) == []


def test_run_deep_reads_new_candidates_and_marks_read(monkeypatch):
    _mock_embed(monkeypatch, {"cats": [1.0, 0.0]})
    sq = SubQuestion(text="cats")
    source = _FakeSource("s", [_item("a", "Cats paper", "all about cats")])
    state = ResearchState(question="cats")
    store = _FakeStore()

    funnel.run(sq, [source], state, store)

    assert state.is_read("a")
    assert len(store.added_chunks) >= 1
    assert all(c.source_id == "a" for c in store.added_chunks)
    assert len(state.findings) >= 1


def test_run_skips_deep_read_when_already_indexed(monkeypatch):
    _mock_embed(monkeypatch, {"cats": [1.0, 0.0]})
    sq = SubQuestion(text="cats")
    source = _FakeSource("s", [_item("a", "Cats paper", "all about cats")])
    state = ResearchState(question="cats")
    store = _FakeStore(already_indexed_ids={"a"})

    funnel.run(sq, [source], state, store)

    assert state.is_read("a")
    assert store.added_chunks == []  # уже в индексе (кэш-хит) — повторно не читаем


def test_run_never_deep_reads_the_same_id_twice_across_calls(monkeypatch):
    _mock_embed(monkeypatch, {"cats": [1.0, 0.0]})
    sq = SubQuestion(text="cats")
    source = _FakeSource("s", [_item("a", "Cats paper", "all about cats")])
    state = ResearchState(question="cats")
    store = _FakeStore()

    funnel.run(sq, [source], state, store)
    first_count = len(store.added_chunks)
    funnel.run(sq, [source], state, store)  # тот же кандидат "переоткрыт"

    assert len(store.added_chunks) == first_count


def test_run_stops_deep_read_when_budget_exhausted(monkeypatch):
    _mock_embed(monkeypatch, {"cats": [1.0, 0.0], "dogs": [1.0, 0.0]})
    sq = SubQuestion(text="pets")
    source = _FakeSource(
        "s", [_item("a", "Cats paper", "cats"), _item("b", "Dogs paper", "dogs")]
    )
    state = ResearchState(
        question="pets", budget=Budget(max_iterations=100, max_deep_reads=1, max_seconds=None)
    )
    store = _FakeStore()

    funnel.run(sq, [source], state, store)

    assert len(state.read_ids) == 1


def test_run_reports_progress_messages(monkeypatch):
    _mock_embed(monkeypatch, {"cats": [1.0, 0.0]})
    sq = SubQuestion(text="cats")
    source = _FakeSource("s", [_item("a", "Cats paper", "all about cats")])
    state = ResearchState(question="cats")
    store = _FakeStore()
    messages = []

    funnel.run(sq, [source], state, store, on_progress=messages.append)

    assert any("Ищем источники" in m for m in messages)
    assert any("Найдено" in m for m in messages)
    assert any("Cats paper" in m for m in messages)


def test_run_without_on_progress_does_not_raise():
    sq = SubQuestion(text="cats")
    state = ResearchState(question="cats")
    store = _FakeStore()

    funnel.run(sq, [], state, store)  # on_progress не передан - должен просто молчать


def test_combined_score_no_citations_equals_cosine():
    assert funnel._combined_score(0.42, None) == 0.42
    assert funnel._combined_score(0.42, 0) == 0.42


def test_combined_score_boosts_by_log_citations(monkeypatch):
    monkeypatch.setattr(funnel.config, "CITATION_BOOST_SCALE", 0.03)
    import math

    boosted = funnel._combined_score(0.5, 1000)
    assert boosted > 0.5
    assert boosted == 0.5 + 0.03 * math.log1p(1000)


def test_discover_enriches_arxiv_items_missing_citation_count(monkeypatch):
    monkeypatch.setattr(funnel, "lookup_citation_count", lambda title: 77)
    arxiv_item = DiscoveredItem(
        id="arxiv:1", source="arxiv", title="Some Paper", abstract="abstract", citation_count=None
    )
    source = _FakeSource("s", [arxiv_item])

    candidates = funnel._discover(SubQuestion(text="q"), [source], discovery_limit=5)
    assert candidates[0].meta["citation_count"] == 77


def test_discover_does_not_enrich_non_arxiv_items(monkeypatch):
    def _fail(title):
        raise AssertionError("lookup_citation_count must not be called for non-arxiv sources")

    monkeypatch.setattr(funnel, "lookup_citation_count", _fail)
    s2_item = DiscoveredItem(
        id="s2:1", source="semantic_scholar", title="Some Paper", abstract="abstract",
        citation_count=None,
    )
    source = _FakeSource("s", [s2_item])

    candidates = funnel._discover(SubQuestion(text="q"), [source], discovery_limit=5)
    assert candidates[0].meta["citation_count"] is None


def test_discover_does_not_enrich_when_citation_count_already_known(monkeypatch):
    def _fail(title):
        raise AssertionError("lookup_citation_count must not be called when already known")

    monkeypatch.setattr(funnel, "lookup_citation_count", _fail)
    arxiv_item = DiscoveredItem(
        id="arxiv:1", source="arxiv", title="Some Paper", abstract="abstract", citation_count=5
    )
    source = _FakeSource("s", [arxiv_item])

    candidates = funnel._discover(SubQuestion(text="q"), [source], discovery_limit=5)
    assert candidates[0].meta["citation_count"] == 5


def test_triage_citation_boost_can_flip_near_tied_ranking(monkeypatch):
    monkeypatch.setattr(funnel.config, "FUNNEL_TRIAGE_TOP_N", 2)
    monkeypatch.setattr(funnel.config, "CITATION_BOOST_SCALE", 0.03)
    # Почти одинаковая семантическая близость (0.50 против 0.51) - без буста
    # "b" был бы первым; высокая цитируемость "a" должна перевесить тайбрейк.
    _mock_embed(monkeypatch, {"query": [1.0, 0.0], "a-text": [0.995, 0.0995], "b-text": [1.0, 0.0]})

    sq = SubQuestion(text="query")
    a = Candidate(id="a", source="arxiv", title="A", abstract="a-text", meta={"citation_count": 5000})
    b = Candidate(id="b", source="arxiv", title="B", abstract="b-text", meta={"citation_count": None})

    survivors = funnel._triage(sq, [a, b])
    assert [c.id for c in survivors] == ["a", "b"]


def test_run_passes_url_and_citation_count_to_chunks(monkeypatch):
    _mock_embed(monkeypatch, {"cats": [1.0, 0.0]})
    sq = SubQuestion(text="cats")
    item = DiscoveredItem(
        id="arxiv:1", source="arxiv", title="Cats paper", abstract="all about cats",
        url="https://arxiv.org/abs/1", citation_count=42,
    )
    source = _FakeSource("s", [item])
    state = ResearchState(question="cats")
    store = _FakeStore()

    funnel.run(sq, [source], state, store)

    assert store.added_chunks[0].url == "https://arxiv.org/abs/1"
    assert store.added_chunks[0].citation_count == 42


def test_canonical_id_same_paper_from_arxiv_source():
    item = DiscoveredItem(id="arxiv:2508.11957v1", source="arxiv", title="T", abstract="a")
    assert funnel._canonical_candidate_id(item) == "arxiv:2508.11957"


def test_canonical_id_same_paper_from_web_abs_and_html_urls():
    abs_item = DiscoveredItem(
        id="web:https://arxiv.org/abs/2508.11957", source="web", title="T", abstract="a",
        url="https://arxiv.org/abs/2508.11957",
    )
    html_item = DiscoveredItem(
        id="web:https://arxiv.org/html/2508.11957v1", source="web", title="T", abstract="a",
        url="https://arxiv.org/html/2508.11957v1",
    )
    assert funnel._canonical_candidate_id(abs_item) == "arxiv:2508.11957"
    assert funnel._canonical_candidate_id(html_item) == "arxiv:2508.11957"


def test_canonical_id_semantic_scholar_via_external_ids():
    item = DiscoveredItem(
        id="s2:abc123", source="semantic_scholar", title="T", abstract="a",
        meta={"external_ids": {"ArXiv": "2508.11957"}},
    )
    assert funnel._canonical_candidate_id(item) == "arxiv:2508.11957"


def test_canonical_id_non_arxiv_item_unchanged():
    item = DiscoveredItem(id="web:https://example.com/blog", source="web", title="T", abstract="a")
    assert funnel._canonical_candidate_id(item) == "web:https://example.com/blog"


def test_discover_produces_matching_canonical_ids_for_same_paper():
    """`_discover` сам не дедуплицирует (это делает `state.add_candidates`),
    но обязан отдать ОДИНАКОВЫЙ id для одной статьи независимо от источника —
    иначе дедуп на уровне state не сработает."""
    arxiv_item = DiscoveredItem(
        id="arxiv:2508.11957v1", source="arxiv", title="A Comprehensive Review",
        abstract="review",
    )
    web_item = DiscoveredItem(
        id="web:https://arxiv.org/html/2508.11957v1", source="web",
        title="A Comprehensive Review (html mirror)", abstract="review",
        url="https://arxiv.org/html/2508.11957v1",
    )
    arxiv_source = _FakeSource("arxiv", [arxiv_item])
    web_source = _FakeSource("web", [web_item])

    candidates = funnel._discover(SubQuestion(text="q"), [arxiv_source, web_source], discovery_limit=5)
    assert [c.id for c in candidates] == ["arxiv:2508.11957", "arxiv:2508.11957"]


class _FakeMCPTool:
    def __init__(self, text=None, raise_error=False):
        self._text = text
        self._raise_error = raise_error
        self.calls = []

    def invoke(self, kwargs):
        self.calls.append(kwargs)
        if self._raise_error:
            raise RuntimeError("mcp server down")
        return [{"type": "text", "text": self._text}]


def _reset_mcp_fetch_cache(monkeypatch):
    monkeypatch.setattr(funnel, "_mcp_fetch_tool", "unset")


def test_deep_read_ignores_mcp_fetch_when_disabled(monkeypatch):
    _reset_mcp_fetch_cache(monkeypatch)
    monkeypatch.setattr(funnel.config, "MCP_FETCH_ENABLED", False)
    monkeypatch.setattr(
        mcp_client, "get_mcp_tools",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("MCP must not be reached when disabled")),
    )
    candidate = Candidate(id="a", source="web", title="T", abstract="short abstract", meta={"url": "https://x/y"})

    sections = funnel._deep_read_sections(candidate)
    assert sections == [funnel.Section(name="T", category="abstract", text="short abstract")]


def test_deep_read_uses_mcp_fetch_full_text_when_enabled(monkeypatch):
    _reset_mcp_fetch_cache(monkeypatch)
    monkeypatch.setattr(funnel.config, "MCP_FETCH_ENABLED", True)
    fake_tool = _FakeMCPTool(text="full page body, much longer than the abstract")
    monkeypatch.setattr(mcp_client, "get_mcp_tools", lambda connections: [fake_tool_with_name(fake_tool)])
    candidate = Candidate(id="a", source="web", title="T", abstract="short abstract", meta={"url": "https://x/y"})

    sections = funnel._deep_read_sections(candidate)
    assert sections == [
        funnel.Section(name="T", category="body", text="full page body, much longer than the abstract")
    ]
    assert fake_tool.calls == [{"url": "https://x/y", "max_length": funnel.config.MCP_FETCH_MAX_CHARS}]


def test_deep_read_falls_back_to_abstract_when_mcp_fetch_fails(monkeypatch):
    _reset_mcp_fetch_cache(monkeypatch)
    monkeypatch.setattr(funnel.config, "MCP_FETCH_ENABLED", True)
    fake_tool = _FakeMCPTool(raise_error=True)
    monkeypatch.setattr(mcp_client, "get_mcp_tools", lambda connections: [fake_tool_with_name(fake_tool)])
    candidate = Candidate(id="a", source="web", title="T", abstract="short abstract", meta={"url": "https://x/y"})

    sections = funnel._deep_read_sections(candidate)
    assert sections == [funnel.Section(name="T", category="abstract", text="short abstract")]


def test_get_mcp_fetch_tool_only_lists_tools_once(monkeypatch):
    _reset_mcp_fetch_cache(monkeypatch)
    monkeypatch.setattr(funnel.config, "MCP_FETCH_ENABLED", True)
    calls = {"n": 0}

    def fake_get_mcp_tools(connections):
        calls["n"] += 1
        return [fake_tool_with_name(_FakeMCPTool(text="x"))]

    monkeypatch.setattr(mcp_client, "get_mcp_tools", fake_get_mcp_tools)

    funnel._get_mcp_fetch_tool()
    funnel._get_mcp_fetch_tool()
    assert calls["n"] == 1


def fake_tool_with_name(fake_tool):
    fake_tool.name = "fetch"
    return fake_tool


def test_run_dedups_same_arxiv_paper_found_via_multiple_sources(monkeypatch):
    """Регрессия на реальный баг (2026-08-06): одна и та же статья находилась
    и через arxiv.py, и через веб-поиск (html-версия) с разными id и
    дублировалась в списке источников реального ответа."""
    _mock_embed(monkeypatch, {"review": [1.0, 0.0]})
    arxiv_item = DiscoveredItem(
        id="arxiv:2508.11957v1", source="arxiv", title="A Comprehensive Review",
        abstract="review",
    )
    web_item = DiscoveredItem(
        id="web:https://arxiv.org/html/2508.11957v1", source="web",
        title="A Comprehensive Review (html mirror)", abstract="review",
        url="https://arxiv.org/html/2508.11957v1",
    )
    arxiv_source = _FakeSource("arxiv", [arxiv_item])
    web_source = _FakeSource("web", [web_item])
    sq = SubQuestion(text="review")
    state = ResearchState(question="review")
    store = _FakeStore()

    funnel.run(sq, [arxiv_source, web_source], state, store)

    assert [c.id for c in state.candidates] == ["arxiv:2508.11957"]
    # Первый найденный источник (arxiv.py) побеждает - его заголовок сохраняется.
    assert state.candidates[0].title == "A Comprehensive Review"
