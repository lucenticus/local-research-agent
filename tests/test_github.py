"""Юнит-тесты sources/github.py — HTTP замокан (офлайн)."""

from __future__ import annotations

import pytest

from src.sources import github


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://github.com/acme/repo", "acme/repo"),
        ("https://www.github.com/acme/repo", "acme/repo"),
        ("https://github.com/acme/repo.git", "acme/repo"),
        ("https://github.com/acme/repo/tree/main/src", "acme/repo"),  # звёзды у репо целиком
        ("https://github.com/acme/repo#readme", "acme/repo"),
        ("https://github.com/acme/", None),   # голый профиль — не репозиторий
        ("https://github.com/", None),
        ("https://huggingface.co/acme/model", None),
    ],
)
def test_repo_slug(url, expected):
    assert github.repo_slug(url) == expected


def test_lookup_stars_returns_count(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        github, "fetch_json",
        lambda url, **kw: (captured.setdefault("url", url), {"stargazers_count": 17460})[1],
    )
    assert github.lookup_stars("https://github.com/tensorflow/tensor2tensor") == 17460
    assert captured["url"].endswith("/repos/tensorflow/tensor2tensor")


def test_lookup_stars_keeps_a_real_zero(monkeypatch):
    """0 звёзд у нового репозитория — настоящее значение, не "не узнали"."""
    monkeypatch.setattr(github, "fetch_json", lambda url, **kw: {"stargazers_count": 0})
    assert github.lookup_stars("https://github.com/acme/brand-new") == 0


def test_lookup_stars_none_when_api_unavailable(monkeypatch):
    """Лимит исчерпан/404/сеть — None, а не 0: подменять ноль неизвестностью
    нельзя, это разные вещи."""
    monkeypatch.setattr(github, "fetch_json", lambda url, **kw: None)
    assert github.lookup_stars("https://github.com/acme/repo") is None


def test_lookup_stars_none_for_non_github_url(monkeypatch):
    def _fail(*a, **kw):
        raise AssertionError("не GitHub — сетевого запроса быть не должно")

    monkeypatch.setattr(github, "fetch_json", _fail)
    assert github.lookup_stars("https://huggingface.co/acme/model") is None


def test_lookup_stars_none_when_field_missing_or_not_int(monkeypatch):
    monkeypatch.setattr(github, "fetch_json", lambda url, **kw: {"message": "Not Found"})
    assert github.lookup_stars("https://github.com/acme/repo") is None
