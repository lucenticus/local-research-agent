"""Звёзды GitHub-репозитория по ссылке из статьи.

Публичный REST API, ключ не нужен, но без него лимит низкий (60 запросов в
час на IP). Поэтому: best effort, как и `citations.py` — на любую ошибку,
403 (лимит исчерпан) или 404 возвращается `None`, и UI честно показывает
ссылку без числа звёзд, а не ноль. Ноль — это реальное значение
("репозиторий никому не интересен"), и подменять им "не смогли узнать"
нельзя.

Звёзды — грубый, но честно измеримый прокси востребованности кода статьи:
в отличие от цитируемости, у свежего препринта они появляются сразу.
"""

from __future__ import annotations

import re

from ._common import fetch_json

_API_URL = "https://api.github.com/repos"
_TIMEOUT_SECONDS = 10
# github.com/<owner>/<repo> — остальное (ветки, файлы, issues в глубине пути)
# отбрасывается: звёзды считаются у репозитория целиком.
_REPO_PATH_RE = re.compile(r"^https?://(?:www\.)?github\.com/([^/\s]+)/([^/\s#?]+)", re.IGNORECASE)


def repo_slug(url: str) -> str | None:
    """URL -> "owner/repo". `None`, если это не ссылка на репозиторий."""
    match = _REPO_PATH_RE.match(url.strip())
    if not match:
        return None
    owner, repo = match.group(1), match.group(2).removesuffix(".git")
    if not repo:
        return None
    return f"{owner}/{repo}"


def lookup_stars(url: str) -> int | None:
    """Число звёзд репозитория. `None` — не GitHub, нет такого репозитория,
    либо API недоступен/исчерпан лимит (см. docstring модуля)."""
    slug = repo_slug(url)
    if not slug:
        return None
    payload = fetch_json(f"{_API_URL}/{slug}", timeout=_TIMEOUT_SECONDS)
    if not payload:
        return None
    stars = payload.get("stargazers_count")
    return stars if isinstance(stars, int) else None
