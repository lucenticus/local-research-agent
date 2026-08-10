"""Юнит-тесты agent/research_runner.py — офлайн, loop/synthesize замоканы."""

from __future__ import annotations

from src.agent import research_runner
from src.agent.state import Candidate, ResearchState


def testunique_sources_dedups_by_title_and_normalizes_citation_sentinel():
    hits = [
        {"source_title": "A", "url": "https://a", "citation_count": 5},
        {"source_title": "A", "url": "https://a-dup", "citation_count": 999},
        {"source_title": "B", "url": "", "citation_count": -1},
    ]
    refs = research_runner.unique_sources(hits)
    assert [r.title for r in refs] == ["A", "B"]
    assert refs[0].citation_count == 5
    assert refs[1].citation_count is None  # -1 сентинел хранилища -> None
    assert refs[1].url == ""


def test_candidate_summaries_sorted_by_score_and_marks_read():
    state = ResearchState(question="q")
    state.add_candidates(
        [
            Candidate(id="a", source="arxiv", title="Low score", abstract="",
                      meta={"citation_count": 3}, triage_score=0.2),
            Candidate(id="b", source="web", title="High score", abstract="",
                      meta={"url": "https://b"}, triage_score=0.9),
            Candidate(id="c", source="arxiv", title="No score", abstract=""),
        ]
    )
    state.mark_read("b")

    summaries = research_runner._candidate_summaries(state)
    assert [s.title for s in summaries] == ["High score", "Low score", "No score"]
    assert summaries[0].read is True
    assert summaries[0].url == "https://b"
    assert summaries[1].citation_count == 3
    assert summaries[1].read is False
    assert summaries[2].triage_score is None


def test_run_research_assembles_result_from_state_and_hits(monkeypatch):
    state = ResearchState(question="q")
    state.add_candidates([Candidate(id="a", source="arxiv", title="Paper", abstract="", triage_score=0.5)])
    state.mark_read("a")
    state.gaps = ["unanswered part"]
    state.iterations = 2

    monkeypatch.setattr(research_runner.loop, "run", lambda *a, **kw: state)
    monkeypatch.setattr(
        research_runner, "retrieve",
        lambda store, question: [{"source_title": "Paper", "url": "https://a", "citation_count": 1}],
    )
    monkeypatch.setattr(research_runner, "synthesize", lambda question, hits, gaps=None: "the answer")

    result = research_runner.run_research("q", store=object())

    assert result.answer == "the answer"
    assert result.sources == [research_runner.SourceRef(title="Paper", url="https://a", citation_count=1)]
    assert result.gaps == ["unanswered part"]
    assert result.iterations == 2
    assert result.read_count == 1
    assert result.candidates_count == 1
    assert result.candidates[0].title == "Paper"
    assert result.candidates[0].read is True


def test_default_sources_prefers_tavily_when_key_configured(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "some-key")
    sources = research_runner.default_sources()
    web_source = sources[-1]
    assert isinstance(web_source, research_runner.TavilySource)


def test_default_sources_falls_back_to_searxng_without_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    sources = research_runner.default_sources()
    web_source = sources[-1]
    assert isinstance(web_source, research_runner.WebSource)
