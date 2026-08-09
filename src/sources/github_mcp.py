"""GitHub repository search via the official GitHub MCP server
(`ghcr.io/github/github-mcp-server`, run through Docker), through
`providers/mcp_client.py` — for subquestions about a specific library/tool/
codebase, where arXiv/Semantic Scholar/CrossRef (papers) and Wikipedia
(encyclopedic) don't have anything useful.

Needs `GITHUB_PERSONAL_ACCESS_TOKEN` in `.env` (read-only scopes are
enough: `public_repo`/`repo:status`) — without it, `discover()` returns an
empty list without ever touching Docker, same contract as
`sources/tavily.py` without a key. Also gated by
`config.GITHUB_MCP_ENABLED` (off by default, see `funnel.py`'s
`MCP_FETCH_ENABLED` for the same rationale — spinning a Docker container
per call is real per-question latency, not something to pay unconditionally
on every `research()` run).

`stargazers_count` is used as `citation_count` — the same log-scaled triage
boost (`config.CITATION_BOOST_SCALE`, see `funnel._combined_score`) that
ranks well-cited papers higher ranks well-starred repos higher, a
reasonable authority proxy for the same tie-breaking purpose.

Found on a real run 2026-08-06: `search_repositories`'s `query` matches
poorly against a full natural-language question ("Is there a good
open-source library for KV cache compression?" -> 0 results) but works
fine with a short keyword query ("KV cache compression library" -> real
results) — same class of issue as arXiv/Semantic Scholar's literal-query
sensitivity, not a bug here. No query rewriting added for this source
specifically; `funnel._discovery_query`'s bounded-LLM translation only
triggers for non-English text, not for reformatting verbose English
questions into keywords.
"""

from __future__ import annotations

import json
import os
from typing import Any

from ..providers.mcp_client import content_to_text, get_single_tool
from .base import DiscoveredItem

_CONNECTIONS_TEMPLATE = {
    "transport": "stdio",
    "command": "docker",
    "args": ["run", "-i", "--rm", "-e", "GITHUB_PERSONAL_ACCESS_TOKEN", "ghcr.io/github/github-mcp-server"],
}


class GitHubMCPSource:
    name = "github"

    def __init__(self, token: str | None = None):
        self._token = token or os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN")
        # Кэш на инстанс, не на модуль: default_sources() создаёт список
        # источников один раз на весь research()-прогон (все проходы
        # воронки переиспользуют один и тот же список) — кэш живёт ровно
        # столько, сколько нужно, без глобального состояния между прогонами.
        self._tool: Any = "unset"

    def _get_tool(self) -> Any:
        if self._tool == "unset":
            if not self._token:
                self._tool = None
            else:
                connections = {
                    "github": {**_CONNECTIONS_TEMPLATE, "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": self._token}}
                }
                self._tool = get_single_tool(connections, "search_repositories")
        return self._tool

    def discover(self, query: str, limit: int) -> list[DiscoveredItem]:
        tool = self._get_tool()
        if tool is None:
            return []
        try:
            result = tool.invoke({"query": query, "perPage": min(max(limit, 1), 100)})
        except Exception:
            return []
        return list(self._parse(content_to_text(result)))

    def _parse(self, text: str):
        try:
            body = json.loads(text)
        except (ValueError, TypeError):
            return
        for repo in body.get("items") or []:
            full_name = repo.get("full_name")
            url = repo.get("html_url")
            if not full_name or not url:
                continue
            description = (repo.get("description") or "").strip()
            topics = repo.get("topics") or []
            abstract = description
            if topics:
                abstract = f"{abstract} (topics: {', '.join(topics)})" if abstract else f"Topics: {', '.join(topics)}"
            stars = repo.get("stargazers_count")
            yield DiscoveredItem(
                id=f"github:{full_name}",
                source=self.name,
                title=full_name,
                abstract=abstract,
                url=url,
                citation_count=stars if isinstance(stars, int) else None,
                meta={},
            )
