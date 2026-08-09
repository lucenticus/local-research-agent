"""Юнит-тесты providers/tracing.py — только env var / config эффекты, без
реального обращения к LangSmith."""

from __future__ import annotations

from src.providers import tracing


def _clear(monkeypatch):
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)


def test_does_nothing_when_disabled_even_with_key(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setattr(tracing.config, "LANGSMITH_TRACING_ENABLED", False)
    monkeypatch.setenv("LANGSMITH_API_KEY", "fake-key")

    tracing.enable_if_configured()

    assert "LANGSMITH_TRACING" not in __import__("os").environ


def test_does_nothing_when_enabled_but_no_key(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setattr(tracing.config, "LANGSMITH_TRACING_ENABLED", True)

    tracing.enable_if_configured()

    import os

    assert "LANGSMITH_TRACING" not in os.environ


def test_sets_tracing_env_vars_when_enabled_and_key_present(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setattr(tracing.config, "LANGSMITH_TRACING_ENABLED", True)
    monkeypatch.setattr(tracing.config, "LANGSMITH_PROJECT", "test-project")
    monkeypatch.setenv("LANGSMITH_API_KEY", "fake-key")

    tracing.enable_if_configured()

    import os

    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_PROJECT"] == "test-project"


def test_does_not_override_explicitly_set_project(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setattr(tracing.config, "LANGSMITH_TRACING_ENABLED", True)
    monkeypatch.setattr(tracing.config, "LANGSMITH_PROJECT", "default-project")
    monkeypatch.setenv("LANGSMITH_API_KEY", "fake-key")
    monkeypatch.setenv("LANGSMITH_PROJECT", "user-chosen-project")

    tracing.enable_if_configured()

    import os

    assert os.environ["LANGSMITH_PROJECT"] == "user-chosen-project"
