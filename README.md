# local-research-agent

A local deep-research agent that runs entirely on-device (MacBook M4 Air,
16GB, MLX). It searches arXiv, Semantic Scholar, and the general web
(Tavily or a local SearXNG instance), reads and cites what it finds, and
self-checks its own answers before returning them.

The full build history, every real bug found along the way, and the
hardware assumptions behind each design choice live in
[DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) and [CLAUDE.md](CLAUDE.md). This
file describes the system as it stands today.

## Quickstart

```bash
uv venv
uv pip install -r requirements.txt

python -m src.cli index
python -m src.cli ask "What is RAG?"

cp .env.example .env   # optional: add TAVILY_API_KEY for broader web search
docker compose up -d   # optional: local SearXNG fallback if no Tavily key
python -m src.cli research "What approaches exist for KV-cache compression in transformers?"

python -m src.cli serve   # web UI at http://127.0.0.1:8000

uv run pytest -q
```

## Architecture

```mermaid
flowchart TB
    browser(["browser"])

    subgraph web["web/app.py (serve)"]
        fastapi["FastAPI: background job<br/>+ polling"]
    end

    subgraph cli["cli.py"]
        index["index"]
        ask["ask"]
        research["research"]
    end

    subgraph runner["agent/research_runner.py"]
        rr["run_research()"]
    end

    subgraph providers["providers/ (resident or on-demand)"]
        embed["embed.py<br/>bge-m3 dense, resident"]
        llm["llm.py<br/>Qwen3.5-4B MLX, resident"]
        rerank["rerank.py<br/>Qwen3-Reranker MLX,<br/>load→score→release"]
    end

    subgraph store["store/lancedb_store.py"]
        lance[("LanceDB<br/>dense + FTS(BM25)<br/>hybrid = RRF")]
    end

    subgraph ingest["ingest/"]
        extract["extract.py<br/>sections, drop refs/acks"]
        chunk["chunk.py<br/>chunk_sections"]
    end

    subgraph agent["agent/"]
        state["state.py<br/>ResearchState"]
        planner["planner.py<br/>question → subquestions"]
        funnel["funnel.py<br/>discovery→triage→deep read"]
        loop["loop.py<br/>gap-check, budget,<br/>faithfulness-retry"]
        synth["synthesize.py<br/>answer + [n] citations"]
        eval["evaluate.py<br/>coverage + faithfulness"]
    end

    subgraph sources["sources/ (metadata only)"]
        arxiv["arxiv.py"]
        s2["semantic_scholar.py"]
        websearch["web.py / tavily.py<br/>(SearXNG or Tavily, see below)"]
    end

    browser <--> fastapi
    fastapi --> rr
    research --> rr
    rr --> loop
    rr --> synth

    index --> ingest --> lance
    index --> embed
    ask --> embed & rerank & lance & synth
    loop --> planner & funnel & synth & eval
    funnel --> sources
    funnel --> ingest
    funnel --> lance
    loop --> lance
    eval --> rerank
    synth --> llm
```

Key memory invariant: `embed`/`llm` stay resident; `rerank` loads and
releases on every call — two heavy models are never resident at once next
to the reranker.

## Local RAG: `index` / `ask`

The simple path: build an embedding index from local files, then ask
questions grounded in that index only (no internet access).

- `ingest/extract.py` — section-aware extraction (markdown/HTML/PDF), drops
  References/Bibliography/Acknowledgments/Appendix sections.
- `ingest/chunk.py` — `chunk_sections` chunks along section boundaries so a
  chunk never straddles two sections.
- `store/lancedb_store.py` — hybrid search (`search_hybrid`): dense vector +
  LanceDB full-text search (BM25), merged via RRF. `search_dense` is kept
  as a baseline for comparison.
- `providers/rerank.py` — `mlx-community/Qwen3-Reranker-0.6B-4bit` (pure
  MLX, 331MB). `ask` takes `RERANK_CANDIDATES_K` hybrid-search hits, the
  reranker loads, scores, releases, and returns the top
  `TOP_K_RETRIEVE`. Toggle off with `config.RERANK_ENABLED = False`.

```bash
python -m src.cli index
python -m src.cli ask "What is RAG?"
```

`index` reads `.txt`/`.md`/`.html`/`.pdf` files from `corpus/` (the repo
ships a small sample corpus: three notes on RAG/vector DBs/local LLMs, plus
one sample "paper" with Abstract/Introduction/Method/Results/Conclusion/
References/Acknowledgments to exercise the reference-stripping logic).

## Deep research: `research`

```bash
python -m src.cli research "What approaches exist for KV-cache compression in transformers?"
```

Unlike `ask`, `research` actively goes out and finds new material instead
of only searching a pre-built index:

- `agent/state.py` — `ResearchState`: subquestions, candidates, `read_ids`,
  findings, budget.
- `sources/arxiv.py`, `sources/semantic_scholar.py`, `sources/web.py` /
  `sources/tavily.py` — metadata-only discovery. arXiv and Semantic Scholar
  work without a key; the web source picks Tavily if `TAVILY_API_KEY` is
  set, otherwise falls back to a local SearXNG instance (see below).
- `agent/planner.py` — question → subquestions via a deterministic
  heuristic, not an LLM call (small local models are unreliable
  multi-step planners, so planning logic lives in code).
- `agent/funnel.py` — discovery → embedding-based triage → deep read (PDF
  fetch + section extraction for arXiv, abstract-only fallback for sources
  without full text). Non-English subquestions get translated to a short
  English search query via a bounded LLM call first — arXiv/Semantic
  Scholar otherwise return zero results for Russian queries.
- `agent/loop.py` — the iterative controller. A subquestion is "covered"
  when retrieval clears a reranker score threshold (`FUNNEL_MIN_RERANK_SCORE
  = 0.5`) across at least `FUNNEL_MIN_SOURCES_TO_COVER = 3` distinct
  sources; each retry pass asks sources for more candidates than the last.

Search breadth and budget are tunable in `config.py`:
`FUNNEL_DISCOVERY_LIMIT_PER_SOURCE` (candidates requested per source),
`FUNNEL_TRIAGE_TOP_N` (how many go to deep read), and
`DEFAULT_BUDGET_MAX_ITERATIONS` / `_MAX_DEEP_READS` / `_MAX_SECONDS`.

### Citation-aware triage

- Semantic Scholar reports `citationCount` directly; arXiv doesn't, so it's
  backfilled via the [OpenAlex](https://openalex.org) Works API
  (`sources/citations.py`, no key required). The triage score is semantic
  similarity to the subquestion plus a small log-scaled citation boost
  (`config.CITATION_BOOST_SCALE`) — a tie-breaker between similarly
  relevant candidates, not a substitute for relevance.
- `url` and `citation_count` flow through the whole pipeline (discovery →
  deep read → LanceDB → retrieval) and show up as clickable links (CLI:
  a plain URL most terminals auto-link; web UI: a real `<a href>` with a
  citation-count badge when known).

### Self-evaluation

`agent/evaluate.py` checks the generated answer before it ships:

- **Citation coverage** — share of claim-sentences carrying at least one
  `[n]`.
- **Faithfulness** — share of cited claims actually supported by the text
  of their cited source, checked via `providers/rerank.score_pairs`
  (reuses the existing reranker instead of a separate NLI model or another
  LLM call).

`agent/loop.py` wires this in: once every subquestion is covered, a draft
answer is scored, and if faithfulness is low, subquestions reopen for one
more forced-discovery pass within budget — `force_discovery` guarantees a
real new search happens on that pass even if the persistent index would
otherwise consider the subquestion already "covered" by stale content from
an earlier question.

### Process visibility

The answer isn't a black box — after completion, the page/CLI output keeps:

- **Progress log** — stays visible next to the answer (not hidden once
  done): how many subquestions, how many iterations, what got read.
- **All discovered candidates** — a table of every candidate the funnel
  found (not just the ones cited in the final answer), sorted by triage
  score: score, citation count, source (`arxiv`/`semantic_scholar`/`web`),
  a clickable link, and a checkmark if it was actually deep-read
  (`agent/research_runner.py::CandidateSummary`).

## Web search: Tavily (recommended) or local SearXNG

`agent/research_runner.py::default_sources()` picks the web source
automatically:

- **`TAVILY_API_KEY` set** (in `.env`, see `.env.example`) → `sources/tavily.py`,
  a managed search API with no third-party engine reliability problems.
  Gives noticeably broader, more authoritative coverage in practice.
- **No key** → `sources/web.py`, a local [SearXNG](https://docs.searxng.org/)
  instance in Docker — no key, no rate limits of its own, but some of its
  underlying engines (DuckDuckGo, Brave) get blocked by their own
  providers (CAPTCHA/rate-limit) even through a local proxy.

```bash
cp .env.example .env
# put TAVILY_API_KEY=... in .env — free tier is roughly 1000 requests/month
```

Without a Tavily key, use local SearXNG instead:

```bash
docker compose up -d          # start (once, runs in the background)
docker compose down           # stop
curl "http://localhost:8888/search?q=test&format=json"   # sanity check
```

Without `docker compose up -d` running, `WebSource.discover()` just
returns an empty list — `research` doesn't crash, the funnel carries on
with whatever arXiv/Semantic Scholar found.

## Web UI: `serve`

```bash
python -m src.cli serve                    # http://127.0.0.1:8000
python -m src.cli serve --port 8080         # different port
```

FastAPI plus a single plain-JS HTML page (no build step, no Node/npm). A
`research()` run can take minutes (real models, external APIs), so it runs
in a background thread; the client polls `GET /api/jobs/{id}` once a
second rather than using SSE/WebSockets — unnecessary complexity for a
single local user. Only one job runs at a time (loading multiple heavy
models concurrently isn't safe on 16GB) — a second request while one is
running gets `409`, not a silent queue.

## Tests and eval scripts

```bash
uv run pytest -q
```

Unit tests are offline and fast — embeddings and the LLM are mocked, no
real model load needed to run the suite.

A few scripts run against real models (not part of `pytest`, since
measuring retrieval/rerank/faithfulness quality without real models would
be meaningless):

```bash
python -m src.cli index
python -m scripts.eval_retrieval      # dense vs hybrid retrieval@k
python -m scripts.eval_rerank         # hybrid vs hybrid+rerank hit@1
python -m scripts.eval_faithfulness   # citation coverage + faithfulness
```

## Known assumptions (`ARCH-Q`)

Assumptions that couldn't be verified without real hardware are marked
`# ARCH-Q:` directly in the code (`src/config.py`, `src/providers/embed.py`,
`src/providers/llm.py`) — LLM HF repo/tag, the FlagEmbedding device kwarg,
`enable_thinking` support in the chat template, and so on. See
`DEVELOPMENT_PLAN.md` for which of these have since been confirmed by a
real run on the target hardware, including peak memory numbers.
