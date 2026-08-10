# CLAUDE.md — research-agent

A local deep-research agent: arXiv / Semantic Scholar / web → iterative
search → synthesis with citations and self-verification. Runs entirely
locally on a MacBook M4 Air, 16GB. See `README.md` for the user-facing
feature tour; this file holds the standing engineering rules.

## How to work
- **Implement → stop → show the diff and how to run it → wait for
  review.** Don't bundle unrelated changes into one pass.
- **Definition of done.** Verified on real hardware (real models, real
  external calls where relevant), not "seems to work" from reading the
  code — see `## Tests` below.
- **`# ARCH-Q:` instead of guessing.** Any assumption that can't be
  verified without the real hardware (MPS device, MLX load call, Qdrant
  hybrid API, memory peaks) — mark it with this tag, don't silently guess.
- Small commits per task. Type hints and docstrings on public functions.

## Hard memory constraints (16GB — do not violate)
- **One resident LLM instance per process.** Its own copy of Qwen3.5,
  loaded once via MLX. Never instantiate the model twice (e.g. in
  synthesis and in gap-check) — reuse the instance.
- **Reranker — on demand.** Load → score → release. Never resident next to
  the LLM.
- **Never hold two resident models at once.**
- **Synthesis context under a hard cap** (in config). Long context
  inflates the KV-cache — the main OOM risk.
- **Don't index everything.** Extraction is a progressive funnel
  (discovery → triage → deep read). Full text and embedding only for what
  passed triage.
- **Corpus via generators.** Never hold all documents in memory at once.
- Embeddings: MPS with a CPU fallback under pressure. `# ARCH-Q:` verify
  on-device.

## Code architecture
- **Provider seams.** External dependencies (LLM, embed, rerank, sources)
  live behind thin interfaces in `src/providers/` and `src/sources/`.
  Agent logic never calls an SDK directly.
- **A single `ResearchState`** — the planner, funnel, loop, and
  synthesizer all read and write it. Invariants: never read one id twice;
  the loop always terminates on budget.
- **Control lives in code, not in the model.** The planner and gap-check
  are deterministic code (a 4B model handles multi-step meta-reasoning
  poorly). The LLM is only used for synthesis (+ optionally a bounded
  gap-check).

## Tests
- Offline and fast: mock the LLM/embeddings.
- Unit: chunking, `ResearchState` transitions, `read_ids` dedup, budget
  stop, hybrid merge.
- One end-to-end integration smoke test on a tiny corpus.
- Run before closing out a milestone.

## Stack
Qwen3.5-4B (Q4, MLX, own copy) · bge-m3 (dense+sparse) ·
bge-reranker-v2-m3 (on demand) · Qdrant (Docker) · arXiv / Semantic
Scholar / CrossRef / Wikipedia / web (+ GitHub via MCP, off by default) ·
LangGraph (orchestration in `agent/loop.py`) · LangChain:
`providers/langchain_llm.py`/`langchain_embeddings.py` wrap the resident MLX
models (not a separate implementation) as `BaseChatModel`/`Embeddings`;
`sources/langchain_tools.py` wraps each `Source.discover()` as a
`StructuredTool` (`agent/funnel.py` calls it via `.invoke()`);
`store/qdrant_store.py::QdrantStore` implements `VectorStore`
(`agent/research_runner.retrieve()` goes through `.as_retriever()`);
`agent/synthesize.py` is an LCEL chain (`prompt | ChatMLX() |
StrOutputParser()`). The reranker (`providers/rerank.py`) and the
gap-check/faithfulness heuristics in `agent/loop.py` stay on their existing
dict-based interfaces on purpose — no functional need to route them through
LangChain, and it would break their test doubles for no benefit.

MCP: `providers/mcp_client.py` is a sync bridge over `langchain-mcp-adapters`
(async-only, `asyncio.run()` per call — see its docstring for why a
persistent connection isn't worth it at this scale). Built on it: an MCP
fetch server as a deep-read fallback (`funnel.py`, off by default —
`config.MCP_FETCH_ENABLED`), an MCP filesystem server for `index --mcp-dir`,
and `mcp_server.py` exposing this agent's own `ask`/`research` as MCP tools
over stdio (`mcp-serve`). See README's "MCP: tools this agent uses, and using this agent as a tool" section.

Tracing: `providers/tracing.py` sets the LangSmith env vars at process start
(off by default — `config.LANGSMITH_TRACING_ENABLED` + `LANGSMITH_API_KEY`).
No other code talks to LangSmith — it instruments every LangChain/LangGraph
component above automatically via LangChain's callback system. See README's
"Tracing: LangSmith" section.

## Running it
```
docker compose up -d qdrant       # required — Qdrant vector store (see README)
python -m src.cli index          # build the index from corpus/
python -m src.cli ask "question"  # ask
```
