"""Юнит-тесты agent/funnel.py — источники, embed и LanceDB замоканы (офлайн).

Кандидаты в тестах не несут `pdf_url` в meta, поэтому deep read всегда идёт
по fallback-ветке (abstract как единственный chunk) — без сетевых вызовов.
"""

from __future__ import annotations

from src.agent import funnel
from src.agent.state import Budget, Candidate, ResearchState, SubQuestion
from src.providers import embed, llm
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
