"""LangSmith tracing — observability for the whole system, not a separate
integration to instrument. This project already routes nearly everything
through LangChain/LangGraph (`ChatMLX`, the LCEL synthesis chain,
`LanceDBStore` as a `VectorStore`, each `Source` wrapped as a
`StructuredTool`, MCP tools, and the `agent/loop.py` research graph itself)
— LangSmith hooks into LangChain's callback system, so turning it on traces
every one of those automatically. Nothing here calls LangSmith directly;
this module's only job is to set the env vars LangChain/LangSmith read.

Off by default (`config.LANGSMITH_TRACING_ENABLED`), same shape as
`MCP_FETCH_ENABLED`/`GITHUB_MCP_ENABLED`, but a different reason: not
latency/deps, but not shipping question text + retrieved chunks + answers to
a third-party cloud service without an explicit opt-in. Also gated on
`LANGSMITH_API_KEY` actually being set in the environment (`.env`) — same
"flag AND credential" pattern as `GITHUB_MCP_ENABLED` + the GitHub token.
"""

from __future__ import annotations

import os

from .. import config


def enable_if_configured() -> None:
    """Call once at process start (see `cli.py::main`, `web/app.py`,
    `mcp_server.py::serve`) — before that, no LangChain call in this process
    has run yet, so setting the env vars here is early enough for LangSmith
    to pick them up on the very first traced call."""
    if not config.LANGSMITH_TRACING_ENABLED:
        return
    if not os.environ.get("LANGSMITH_API_KEY"):
        return
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ.setdefault("LANGSMITH_PROJECT", config.LANGSMITH_PROJECT)
