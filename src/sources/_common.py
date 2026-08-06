"""Общие хелперы для веб-источников (sources/web.py, sources/tavily.py)."""

from __future__ import annotations

import re

# Страницы-листинги/индексы, не отдельные статьи — не содержательный источник.
# Найдено реальным прогоном 2026-08-06: arxiv.org/list/cs.AI/recent (список
# последних публикаций по категории) проходил как полноценный источник.
_NON_ARTICLE_URL_RE = re.compile(r"arxiv\.org/list/")


def is_article_url(url: str) -> bool:
    return bool(url) and not _NON_ARTICLE_URL_RE.search(url)
