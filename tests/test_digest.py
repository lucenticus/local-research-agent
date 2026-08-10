"""Юнит-тесты agent/digest.py — ArxivSource.recent() и LLM замоканы (офлайн)."""

from __future__ import annotations

from src.agent import digest
from src.sources.base import DiscoveredItem
from src.sources.arxiv import ArxivSource


def _item(i: int) -> DiscoveredItem:
    return DiscoveredItem(id=f"arxiv:{i}", source="arxiv", title=f"Paper {i}", abstract=f"Abstract {i}")


def test_run_digest_uses_defaults_when_not_specified(monkeypatch):
    captured = {}

    def fake_recent(self, days, limit):
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

    def fake_recent(self, days, limit):
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
    monkeypatch.setattr(ArxivSource, "recent", lambda self, days, limit: [_item(1)])

    def _fail(items):
        raise AssertionError("summarize must not be called when summarize=False")

    monkeypatch.setattr(digest, "_summarize", _fail)

    result = digest.run_digest(summarize=False)
    assert result.summary is None


def test_run_digest_skips_summary_when_no_items(monkeypatch):
    monkeypatch.setattr(ArxivSource, "recent", lambda self, days, limit: [])

    def _fail(items):
        raise AssertionError("summarize must not be called on an empty digest")

    monkeypatch.setattr(digest, "_summarize", _fail)

    result = digest.run_digest()
    assert result.summary is None
    assert result.items == []


def test_summarize_builds_prompt_from_titles_and_abstracts(monkeypatch):
    captured = {}
    monkeypatch.setattr(digest.llm, "build_chat_prompt", lambda system, user: (captured.setdefault("user", user), "prompt")[1])
    monkeypatch.setattr(digest.llm, "generate", lambda prompt, **kw: "  overview text  ")

    result = digest._summarize([_item(1), _item(2)])
    assert result == "overview text"
    assert "Paper 1" in captured["user"]
    assert "Paper 2" in captured["user"]
