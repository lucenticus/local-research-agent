"""Юнит-тесты agent/digest.py — ArxivSource.recent() и LLM замоканы (офлайн)."""

from __future__ import annotations

from src.agent import digest
from src.sources.arxiv import ArxivSource
from src.sources.base import DiscoveredItem
from src.sources.citations import AuthorDetails, PaperDetails


def _item(i: int) -> DiscoveredItem:
    return DiscoveredItem(id=f"arxiv:{i}", source="arxiv", title=f"Paper {i}", abstract=f"Abstract {i}")


def test_run_digest_uses_defaults_when_not_specified(monkeypatch):
    captured = {}

    def fake_recent(self, days, limit, query=None):
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

    def fake_recent(self, days, limit, query=None):
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
    monkeypatch.setattr(ArxivSource, "recent", lambda self, days, limit, query=None: [_item(1)])

    def _fail(items):
        raise AssertionError("summarize must not be called when summarize=False")

    monkeypatch.setattr(digest, "_summarize", _fail)

    result = digest.run_digest(summarize=False)
    assert result.summary is None


def test_run_digest_skips_summary_when_no_items(monkeypatch):
    monkeypatch.setattr(ArxivSource, "recent", lambda self, days, limit, query=None: [])

    def _fail(items):
        raise AssertionError("summarize must not be called on an empty digest")

    monkeypatch.setattr(digest, "_summarize", _fail)

    result = digest.run_digest()
    assert result.summary is None
    assert result.items == []


def test_run_digest_reports_progress(monkeypatch):
    monkeypatch.setattr(ArxivSource, "recent", lambda self, days, limit, query=None: [_item(1)])
    monkeypatch.setattr(digest, "_summarize", lambda items: "summary")
    messages = []

    digest.run_digest(on_progress=messages.append)
    assert any("Ищем статьи" in m for m in messages)
    assert any("Найдено 1" in m for m in messages)
    assert any("обзор" in m for m in messages)


def test_run_digest_passes_query_through_and_mentions_it_in_progress(monkeypatch):
    captured = {}

    def fake_recent(self, days, limit, query=None):
        captured["query"] = query
        return [_item(1)]

    monkeypatch.setattr(ArxivSource, "recent", fake_recent)
    monkeypatch.setattr(digest, "_summarize", lambda items: "summary")
    messages = []

    result = digest.run_digest(query="  diffusion models  ", on_progress=messages.append)
    assert captured["query"] == "diffusion models"  # обрезано
    assert result.query == "diffusion models"
    assert any("«diffusion models»" in m for m in messages)


def test_run_digest_blank_query_becomes_none(monkeypatch):
    captured = {}

    def fake_recent(self, days, limit, query=None):
        captured["query"] = query
        return []

    monkeypatch.setattr(ArxivSource, "recent", fake_recent)

    result = digest.run_digest(query="   ")
    assert captured["query"] is None
    assert result.query is None


def test_run_digest_skips_analysis_by_default(monkeypatch):
    monkeypatch.setattr(ArxivSource, "recent", lambda self, days, limit, query=None: [_item(1)])
    monkeypatch.setattr(digest, "_summarize", lambda items: "summary")

    def _fail(item):
        raise AssertionError("_summarize_item must not be called when deep=False")

    monkeypatch.setattr(digest, "_summarize_item", _fail)

    result = digest.run_digest()
    assert result.analyses == {}


def test_run_digest_deep_analyzes_each_item(monkeypatch):
    items = [_item(1), _item(2)]
    monkeypatch.setattr(ArxivSource, "recent", lambda self, days, limit, query=None: items)
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
    monkeypatch.setattr(ArxivSource, "recent", lambda self, days, limit, query=None: [_item(1)])
    monkeypatch.setattr(digest, "_summarize", lambda items: "summary")
    monkeypatch.setattr(digest, "_summarize_item", lambda item: "резюме")
    monkeypatch.setattr(digest, "lookup_paper_details", lambda title: None)

    result = digest.run_digest(deep=True)
    assert result.analyses["arxiv:1"].details is None
    assert result.analyses["arxiv:1"].summary_ru == "резюме"


def test_run_digest_deep_caps_at_deep_max_items(monkeypatch):
    monkeypatch.setattr(digest.config, "DIGEST_DEEP_MAX_ITEMS", 2)
    items = [_item(i) for i in range(5)]
    monkeypatch.setattr(ArxivSource, "recent", lambda self, days, limit, query=None: items)
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
