# Deep-Research Agent — development plan

A local agent for scientific research and trends. Iteratively searches
arXiv / Semantic Scholar / the web, reads selectively, synthesizes an
answer with citations and self-verification.

**Target hardware:** MacBook M4 Air, 16GB. Fully local.
**How to proceed:** implement **one milestone at a time**, each its own
branch/commit series, stop and review after each. Don't start the next
milestone until the previous one's acceptance criteria are met.

---

## 1. Hard constraints (do not violate)

These rules outrank features. Any code that breaks them is broken code.

- **One resident LLM instance per process.** The project keeps its own
  copy of Qwen3.5 (copied into the repo) and loads it once via MLX —
  independent of storyreel. Never instantiate the model twice in one
  process (e.g. separately in synthesis and in gap-check); reuse the single
  instance. `# ARCH-Q:` the MLX loading code can be modeled on storyreel's.
- **Reranker — on demand only.** Load before reranking, release after.
  Never keep it resident next to the LLM.
- **Context under a hard cap.** A hard ceiling on synthesis context
  characters/tokens (config). Long context inflates the KV-cache — the
  main OOM risk.
- **Embeddings survive on CPU.** `# ARCH-Q:` bge-m3 defaults to MPS; if MPS
  is under pressure next to the LLM, fall back to CPU (query embedding is
  cheap).
- **We don't index everything.** No bulk corpus downloads. Extraction is a
  progressive funnel (see §3), full text and embedding only for what
  passed triage.
- **Streaming processing.** Process the corpus with generators, never hold
  all documents in memory at once.

---

## 2. Stack

| Layer | Choice | Note |
|---|---|---|
| Generation | Qwen3.5-4B (Q4), MLX, own copy in the project | resident, single instance |
| Embeddings | bge-m3 (dense + sparse), FlagEmbedding | resident; MPS→CPU fallback |
| Reranker | bge-reranker-v2-m3 | on demand |
| Storage | LanceDB (embedded, on disk) | no separate server |
| Sources | arXiv API, Semantic Scholar Graph API, web (Tavily or SearXNG) | keys/limits in config |
| Text extraction | arXiv HTML/TeX if available, else PDF→text (pymupdf) | section-aware |

---

## 3. Target architecture

A progressive extraction funnel — pay for the expensive stuff only for
what clears the filter:

1. **Discovery** — query the sources, only metadata comes back (title,
   abstract, year, citation count). Cheap, dozens-to-hundreds of
   candidates per subquestion.
2. **Triage** — score candidates by abstract against the subquestion
   (embedding/rerank on the abstract). Keep top-N (10–20). Most articles
   die here — full text is never fetched.
3. **Deep read** — only for survivors: full text → section-aware
   extraction (prioritize abstract/intro/method/results/conclusion, drop
   references/acks) → chunking → embedding → LanceDB.
4. **Cache** — what's been read persists in LanceDB, reused by future
   questions. The corpus grows organically from what's been researched.

The control loop is an **iterative loop** around a state object (§5). The
planner and gap-assessment live in code, not in the 4B model's head. The
LLM is only used for synthesis (and, optionally, a bounded gap-check).

---

## 4. Repository layout

```
research-agent/
  CLAUDE.md                 # conventions (see §7)
  DEVELOPMENT_PLAN.md       # this file
  README.md
  requirements.txt
  config.py                 # paths, model tags, memory/context limits, API keys
  corpus/                   # local .txt/.md for milestone 0
  src/
    providers/
      llm.py                # loads the project's own Qwen3.5 copy via MLX, single instance
      embed.py               # bge-m3: dense + sparse, MPS/CPU
      rerank.py               # bge-reranker, load-on-demand + release
    store/
      lancedb_store.py      # schema, upsert, hybrid search
    ingest/
      extract.py             # PDF/HTML/TeX -> clean text, section-aware
      chunk.py                # chunking
    sources/
      base.py                 # common Source interface: discover() -> [Candidate]
      arxiv.py
      semantic_scholar.py
      web.py
    retrieval/
      retriever.py            # hybrid (dense+sparse) + optional rerank
    agent/
      state.py                 # ResearchState (§5)
      planner.py               # question -> subquestions
      funnel.py                 # discovery -> triage -> deep read
      loop.py                   # iterative controller (gap, stop condition)
      synthesize.py             # answer with [n] citations
      evaluate.py               # faithfulness / groundedness
    cli.py                    # entrypoints: index, ask
  tests/
```

Each module is a thin interface (provider seams), so components stay
swappable and testable with mocks.

---

## 5. Research state object

`ResearchState` — a single object read and written by the planner, funnel,
loop, and synthesizer. Fields (guideline):

- `question: str` — the original question
- `sub_questions: list[SubQ]` — each with status `open | covered`
- `candidates: list[Candidate]` — found so far (id, source, title, abstract,
  meta, triage_score)
- `read_ids: set[str]` — what's already been deep-read (dedup, never read
  twice)
- `findings: list[Finding]` — extracted chunks (text, source_id, which
  sub_q they belong to)
- `gaps: list[str]` — unclosed gaps (drive follow-up)
- `iterations: int`, `budget: Budget` — counter and limits (max
  iterations, max deep-reads, time)

Invariants: never read the same `id` twice; `candidates`/`findings` only
ever grow within a request; the loop must terminate on `budget` even with
gaps remaining (then say so honestly in the answer).

---

## 6. Milestones

### Milestone 0 — dumb end-to-end path
**Goal:** end-to-end on a preloaded mini-corpus, single-pass, no loop or
funnel.
**Tasks:** `config.py`; `providers/embed.py` (bge-m3 dense);
`store/lancedb_store.py` (index + vector search); `ingest/chunk.py` (naive
fixed-size chunking); `providers/llm.py` (load Qwen3.5 via MLX, single
instance); `agent/synthesize.py` (prompt with citations); `cli.py`
(`index`, `ask`). Can start from a ready-made `rag_skeleton.py`.
**Done when:**
- [ ] `python -m src.cli index` builds an index from `corpus/`
- [ ] `python -m src.cli ask "..."` returns an answer with `[n]` citations
      and a source list
- [x] peak memory measured and within budget: real run on M4 Air 16GB,
      2026-08-04, `ask` (bge-m3 dense on MPS + Qwen3.5-4B-4bit resident at
      the same time) — **peak memory footprint ~5.92GB**, `index`
      (embedding only) — ~3.2GB. Both comfortably within the 16GB budget.
**Not doing:** sources, reranking, the loop, the funnel.

### Milestone 1 — hybrid retrieval + clean extraction ✅ (2026-08-05)
**Goal:** make retrieval good.
**Tasks:** `embed.py` — add bge-m3's sparse output; `lancedb_store.py` —
FTS index + hybrid search merged via RRF, `# ARCH-Q:` confirm the current
LanceDB hybrid API; `ingest/extract.py` — section-aware extraction
(PDF/HTML), drop references/acks; smart chunking along section boundaries.

**Decision on sparse (closes the ARCH-Q):** confirmed by hand on lancedb
0.36.0 — the native `table.create_index("text", config=FTS(language="Russian",
stem=True))` + `table.search(query_type="hybrid").vector(...).text(...)`,
default reranker `RRFReranker`. tantivy's Russian stemmer works correctly
on Cyrillic (confirmed). So the "sparse" signal is implemented via LanceDB
FTS/BM25 rather than a separate bge-m3 sparse output (lexical weights) —
keeping both would be a duplicate signal with no proven benefit at this
project's scale; documented in `lancedb_store.py`'s docstring.

**Done when:**
- [x] the retriever returns a merged dense+FTS result set
      (`LanceDBStore.search_hybrid`, RRF) — dense-only kept as
      `search_dense` for comparison
- [x] 8 reference pairs (question → source_id), `scripts/eval_retrieval.py`,
      real bge-m3 embeddings. Recall@3: dense 8/8, hybrid 8/8 — hybrid is
      not worse than dense (the corpus is still too small for the
      difference to show in recall; ranking is genuinely different though —
      confirmed by hand on a lexical query, hybrid swaps ranks 2/3 relative
      to dense)
- [x] extraction from an article contains no bibliography/acknowledgments —
      confirmed on `corpus/hybrid_retrieval_paper.md` (References/
      Acknowledgments absent from the index after `index`)
**Not doing:** reranking, the loop.

### Milestone 2 — reranker (on demand) ✅ (2026-08-05)
**Goal:** higher retrieval precision, memory under control.
**Tasks:** `providers/rerank.py` (bge-reranker, load→score→release); wire
into `retriever.py` as an optional stage on top of top-N.

**Model swap (closes an unexpected ARCH-Q):** BAAI/bge-reranker-v2-m3 via
FlagEmbedding conflicts by version with mlx_lm — `FlagReranker` requires
`tokenizer.prepare_for_model`, a method removed in `transformers>=5`,
while `mlx_lm` (our LLM provider) requires `transformers>=5`
(`TokenizersBackend`). Confirmed by a real run: with `transformers<5` the
reranker works but `ask` fails loading the LLM; with `transformers>=5` it's
the other way around. Switched to `mlx-community/Qwen3-Reranker-0.6B-4bit`
— pure MLX (loads via the same `mlx_lm.load()` as the main LLM), no
conflict at all. Multilingual, Russian confirmed by hand. Implemented in
`providers/rerank.py`.

**Done when:**
- [x] reranking is not worse than hybrid on the reference pairs —
      `scripts/eval_rerank.py`, hit@1: hybrid 8/8, +rerank 8/8 (the corpus
      is too small to show a file-level win; a manual check of chunk
      ordering within one article shows reranking agrees with the hybrid
      ranking, no regression)
- [x] the memory measurement confirms the reranker isn't resident and gets
      released — `mx.get_active_memory()`/`get_cache_memory()` on a real
      run: 0MB → 335MB (weights) / +360MB (compute cache) while scoring →
      **0/0 after `_release()`** (`del` + `gc.collect()` + `mx.clear_cache()`)
- [x] reranking can be disabled with a config flag — `config.RERANK_ENABLED`;
      when `False`, `cmd_ask` takes top-k straight from hybrid search

### Milestone 3 — funnel + iterative loop + state (core) ✅ (2026-08-05)
**Goal:** turn this from RAG into deep research.
**Tasks:** `agent/state.py` (§5); `sources/*` (metadata-only discovery,
a single `base.Source`); `agent/funnel.py` (discovery→triage→deep read,
growing LanceDB on the fly); `agent/planner.py` (question→subquestions);
`agent/loop.py` (for each open subquestion: retrieve → gap-check →
follow up/discover more → until covered or out of budget). Gap-check:
start with a heuristic (score threshold + subquestion coverage),
optionally a bounded LLM yes/no.

**Implemented:** `agent/state.py` (ResearchState/Candidate/Finding/Budget),
`sources/base.py`+`arxiv.py`+`semantic_scholar.py`+`web.py` (arXiv and
Semantic Scholar genuinely work without a key; web originally used
Tavily/a key — replaced with local SearXNG, see the post-M4 update below),
`agent/planner.py` (deterministic heuristic: compound questions are split
on "?"/";"/"and also", otherwise one subquestion), `agent/funnel.py`
(discovery → embedding-based triage top-N → deep read: PDF +
`extract_pdf_sections` for arXiv, an honest fallback to the abstract for
sources without full text), `agent/loop.py` (iterative controller with a
growing `discovery_limit` on retry), `store/lancedb_store.py::
add_chunks/has_source` (incremental upsert + cache hit), `cli.py research`.

**Three real bugs/gaps found by running the code, not by guessing:**
1. **Off-by-one in the iteration budget.** `state.iterations` was
   incremented BEFORE the inner per-subquestion loop → `budget_exhausted()`
   inside that same loop was tripped by that same increment, and the last
   permitted pass always got wasted. Caught by a unit test on mocks
   (`tests/test_loop.py`), fixed by incrementing after the pass.
2. **arXiv/Semantic Scholar are English-language, subquestions are in
   Russian.** A literal Russian query returned 0 results on the real API.
   Added a bounded LLM translation of the subquestion into a short English
   search query before discovery (`agent/funnel.py::_discovery_query`) —
   extends the plan's allowance for "optional bounded LLM" beyond
   gap-check, since without this the sources simply don't work for the
   target (Russian-speaking) user.
3. **Gap-check without a score threshold falsely "covered" irrelevant
   subquestions.** The first version considered a subquestion covered by a
   single criterion (≥2 distinct source_ids in the top-k), with no
   relevance check — the plan explicitly required "score threshold +
   coverage". On a real run, a topically adjacent but non-answering local
   corpus about LLM quantization falsely "covered" a question about
   KV-cache compression. Fix: gate on a calibrated reranker score
   (`rerank.score(...) >= 0.5`) on top of the distinct-source count — reuse
   the existing reranker instead of a separate LLM yes/no.

**Done when:**
- [x] on an open-ended question, the agent runs >1 iterations and finds
      articles that weren't preloaded — real run (isolated index, a
      question about KV-cache compression, `FUNNEL_TRIAGE_TOP_N=1` to force
      multiple passes): 3 iterations, 20 real candidates from
      arXiv/Semantic Scholar (none were in the local corpus), 44 findings
- [x] `read_ids` genuinely deduplicates — one article is never read twice —
      unit tests (`test_run_never_deep_reads_the_same_id_twice_across_calls`)
      + a real run: a repeated `funnel.run` with an already-known id
      touches the store not at all
- [x] the loop always terminates on `budget`; the answer reflects any
      remaining gaps — unit tests on mocks + a real run with a deliberately
      unreachable coverage threshold: the loop stopped on `max_seconds`,
      `state.gaps == [question]`, `synthesize()` receives the gaps and
      honestly flags the uncovered part in the answer
- [x] what's been read persists in LanceDB and gets reused by the next
      question (cache hit) — real run: a second `loop.run()` call for the
      same question against the same index — `iterations=1`, `read_ids`
      empty, `candidates` empty — coverage came from the persistent index
      with zero external requests
**Not doing:** the eval layer, polish.

### Milestone 4 — evaluation + packaging ✅ (2026-08-05)
**Goal:** self-verification and showcase readiness.
**Tasks:** `agent/evaluate.py` (faithfulness: every claim is derivable from
its source; citation coverage: every claim has a `[n]`); wire into
`loop.py` (low score → one more iteration within budget); optional figures
via VLM Qwen; README + diagram + reproducibility.

**Implemented:** `agent/evaluate.py` (citation coverage + faithfulness —
faithfulness via the existing `providers/rerank.score_pairs`, not a
separate NLI model or another LLM call: each cited claim gets its own
query, a single reranker load/release for the whole answer);
`providers/rerank.py::score_pairs` (a new function for different queries
per pair, alongside `score`/`rerank`); `agent/loop.py` — once every
subquestion is covered, a draft synthesis runs through `evaluate()`, and
if faithfulness is below `EVAL_FAITHFULNESS_THRESHOLD` the subquestions
reopen for exactly ONE more pass (not unlimited — otherwise a
systematically weak index could burn the entire budget for nothing);
`scripts/eval_faithfulness.py`.

**Optional figures via VLM Qwen — not implemented** (marked "optional" in
the tasks, deliberately out of scope: the project already juggles 3 models
resident/on-demand, VLM figures aren't a critical showcase feature).

**Done when:**
- [x] faithfulness is computed and printed on reference questions —
      `scripts/eval_faithfulness.py`, a real run over 8 questions: mean
      citation coverage 0.59, mean faithfulness 0.75 (real numbers from
      real models, not invented)
- [x] found and demonstrated a case where the eval caught an unsupported
      claim — **two independent cases**: (1) unit test
      `test_catches_unsupported_claim` — a controlled injection of a made-up
      fact ("whales have three hearts and breathe through gills"), correctly
      caught by the eval; (2) a real run on a question about the numeric
      recall improvement from hybrid retrieval — the model (honestly)
      wrote a meta-claim about missing data, citing the range `[1]–[5]`,
      and the eval flagged it unsupported (the reranker score fell below
      threshold for any single source). The second case isn't a fabricated
      fact but an honestly-found limitation of the method: the reranker
      doesn't validate negations/meta-claims well against a single source.
      Attempts to provoke a genuine fabricated number (asking for exact
      GB/ms/seconds not present in the corpus) failed — the model honestly
      answered "not in the context" every time, itself a good sign for
      synthesis quality
- [x] `README` lets you run the project from scratch; there's an
      architecture diagram
- [x] there's an honest quality metric on several reference questions —
      see above, real numbers, not massaged

---

## Post-M4: a working general web search (2026-08-05)

All 4 plan milestones are closed, but `sources/web.py` from Milestone 3 was
optional and never verified live (Tavily needs a key that wasn't available
on this machine). Brought to a genuinely working state at the user's
request.

**Confirmed by hand: neither free option without its own infrastructure
works:**
- DuckDuckGo HTML (`html.duckduckgo.com/html/`) — blocks automated
  requests with a CAPTCHA challenge ("making sure you're not a bot",
  `cc=botnet` in the response's tracking pixel). Working around this would
  be bot-detection bypass — not doing that.
- Public SearXNG instances (searx.be, priv.au, searx.tiekoetter.com,
  search.inetol.net, baresearch.org, opnxng.com) — either `429 Too Many
  Requests` almost immediately, or HTML-only (public instances disable the
  JSON API by default against abuse), or their own bot-check.

**Solution — a local SearXNG in Docker**, our own instance with no key and
no limits: `docker-compose.yml` + `searxng/settings.yml`
(`use_default_settings: true` + overriding `search.formats: [html, json]`,
otherwise JSON is disabled in the default image too). A real gotcha: the
`searxng/searxng` image listens on port **8080 inside the container**, not
8888 (even though the default `settings.yml` says `server.port: 8888`, the
actual uwsgi process in this image comes up on 8080) — `docker port` showed
the `8888/tcp` mapping empty until `ports: ["8888:8080"]` was fixed in the
compose file.

`sources/web.py` was rewritten around
`GET {SEARXNG_BASE_URL}/search?q=...&format=json` (no key). A real run of
`loop.run()` on "what local embedded vector databases exist for Apple
Silicon" (a topic where arXiv/Semantic Scholar are naturally empty — not a
paper topic, tools/blogs instead): web found 5 candidates (GitHub repos,
blog posts), 3 got read, the subquestion closed in 1 iteration — the
source genuinely participates in the funnel alongside arXiv/Semantic
Scholar.

---

## Post-M4: web UI (2026-08-05)

At the user's request — a convenient web UI for `research`. Stack:
FastAPI + a single plain-JS HTML page (no build step, no Node/npm —
matches the project's spirit of minimal dependencies).

**Architecture:** `agent/research_runner.py` — the shared
`loop.run() → retrieve → synthesize` glue, pulled out of `cli.py` so the
CLI and the web layer don't duplicate it. `agent/progress.py` — a shared
progress-callback type (`ProgressCallback`); `funnel.run()`/`loop.run()`
now take an optional `on_progress`, which the CLI doesn't use (it just
prints via `print`), while the web UI accumulates it into a message list.

`src/web/app.py` (FastAPI) + `src/web/static/index.html`: a single
`research()` run executes in a background thread (can take minutes — real
models and external APIs), the client polls `GET /api/jobs/{id}` once a
second (no WebSocket/SSE — unnecessary complexity for one local user).
Only one `research()` run is allowed at a time (§1: can't load several
heavy models concurrently on 16GB) — a second request while one is running
gets rejected with `409`.

`python -m src.cli serve` — starts the server (defaults to
`127.0.0.1:8000`). 8 unit tests on `TestClient` (offline, `run_research`
mocked) — job lifecycle, 409 on a concurrent request, error reflected in
status.

**Verified for real in the browser** (not just tests): form → job creation
→ live progress ("Subquestions: 1.", "Iteration 1…", "Searching
sources…", "Reading: …") → a finished answer with `[1]`–`[4]` citations, a
source list (the LoRC paper and others), and stats ("Iterations: 1 ·
sources read: 3 · candidates found: 5") on a KV-cache compression question
— the same real run previously only exercised via the CLI.

---

## Post-M4: citation-aware triage + clickable links (2026-08-06)

At the user's request: (1) clickable links to articles and (2) factoring
citation counts into candidate analysis.

**Citation sources — confirmed for real:** Semantic Scholar already
reports `citationCount` at discovery time (already had this). arXiv
doesn't give its own citation count — backfilled via the **OpenAlex Works
API** (`api.openalex.org`, no key, `cited_by_count` by title) —
`sources/citations.py`.

**A real bug, found on a live run and fixed:** OpenAlex's plain
full-text `search=` parameter ranks by general relevance, not title
similarity — a query for `"The risk of KV cache compression"` (an
uncited paper published a few weeks earlier) came back with the top hit
being the **"XGBoost"** paper (50,424 citations!) — the titles aren't
remotely alike, just some overlapping vocabulary. Discovered visually in a
real answer (`[2] The risk of KV cache compression (citations: 50424)` —
a physically impossible number for a brand-new paper). Fixed:
`filter=title.search:...` (an actual title search) plus a title-similarity
guard (`difflib.SequenceMatcher`, threshold 0.6) — if the top result isn't
similar enough to the requested title, return `None` instead of someone
else's number. Regression tests `test_lookup_rejects_mismatched_title_result`
and `test_lookup_accepts_closely_matching_title` lock in both cases.
Confirmed by a repeat real run: the same paper now correctly shows
`citations: 0`.

**Use in triage:** `agent/funnel.py::_combined_score` — the final score =
cosine (semantic relevance to the subquestion) +
`CITATION_BOOST_SCALE * log1p(citation_count)` (log-scaled, not raw —
otherwise a paper with thousands of citations would drown out any
semantics; the 0.03 scale is calibrated so the boost is a tie-breaker
between similarly relevant candidates, not a substitute for semantics).
Unit test `test_triage_citation_boost_can_flip_near_tied_ranking` confirms:
with near-equal cosine scores, the more-cited candidate outranks the less-
cited one.

**Clickable links:** `url` and `citation_count` are now part of the
LanceDB schema (`store/lancedb_store.py::Chunk`) and flow through the
whole pipeline — discovery → triage → deep read → chunk → retrieval →
`agent/research_runner.py::SourceRef` → the CLI (the URL is printed under
the source, most terminals auto-link it) and the web UI (a real `<a href>`,
confirmed by hand via `read_page` in the browser: `link "LoRC: ..." href=
"https://arxiv.org/abs/2410.03111v1"`).

**A real LanceDB schema gotcha, found and locked in by a test:** sentinels
`url=""` / `citation_count=-1` instead of `None` — mixing `None`/`int` in
one column across different inserts (`rebuild` starts with `None`,
`add_chunks` later adds a real `int`) produces a null-typed column in
LanceDB, which then fails with `Invalid input, cannot cast field ... from
Int64 to Null` on the first attempt to add a real number — confirmed by
hand, locked in by
`test_mixed_sentinel_and_real_citation_count_across_add_chunks`.

---

## Post-M4: three bugs from real usage (2026-08-06)

The user sent a real web UI answer to a question about "the most-cited
recent AI agent papers" — its sources revealed three problems, all found
and fixed:

1. **Duplicates of the same paper under different URLs.** `[2508.11957]`
   showed up both as `arxiv:2508.11957v1` (found by `arxiv.py`) and
   separately as `web:<html-URL of the same paper>` (found by web search)
   — different ids, `state.add_candidates` never saw them as the same
   candidate. Fix: `agent/funnel.py::_canonical_candidate_id` — any arXiv
   id/URL (abs/html/pdf, versioned or not, from any source, including
   Semantic Scholar's `externalIds.ArXiv`) gets normalized to
   `arxiv:<version-less-id>` BEFORE dedup runs.
2. **A listing page counted as a source.**
   `arxiv.org/list/cs.AI/recent` (a browse page of recent publications by
   category, not an individual article) ended up in the sources via web
   search. Fix: `sources/web.py` filters such URLs.
3. **The demo corpus leaked into real answers.** `research` and `ask`
   shared one LanceDB table — a test fixture note,
   `hybrid_retrieval_paper.md` (written to test Milestone 1), showed up as
   a source in an answer about AI agents purely because it lived in the
   same index and cleared the score threshold. Fix:
   `config.RESEARCH_INDEX_TABLE` — a separate table for `research`/`serve`;
   `ask`/`index` stay on the local `corpus/`.

All three confirmed by a repeat real run in the browser (`read_page`): on
a similar AI-agents question, the sources are three genuine on-topic
articles/pages, no duplicates, no listing pages, no demo corpus.

---

## Post-M4: Tavily as the preferred web source (2026-08-06)

The user asked why `research` doesn't search "the whole internet". The
real cause — confirmed by hand from the local SearXNG container's own logs
(`docker logs local-research-agent-searxng`), not a guess:

```
searx.engines.duckduckgo: CAPTCHA (wt-wt) (suspended_time=0)
searx.engines.brave: Too many request (suspended_time=180)
```

Of 83 engines enabled in SearXNG, in practice only Google CSE and
Wikipedia reliably respond — DuckDuckGo gets CAPTCHA-blocked even through
the local proxy (the same class of problem already found with direct
DuckDuckGo HTML, see Milestone 3), Brave rate-limits. This explains why
web search coverage felt narrow in practice (mostly tag/listing pages from
a handful of sites).

The user provided a working **Tavily API** key — a managed search service
with no third-party engine reliability problems. `sources/tavily.py`
implements the same `Source` protocol as `sources/web.py` (SearXNG).
`agent/research_runner.py::default_sources()` picks Tavily if
`TAVILY_API_KEY` is set in the environment, otherwise local SearXNG (both
paths stay fully working and covered by tests).

**The key is a secret, handled accordingly:** written to `.env` (added to
`.gitignore` BEFORE anything was written there, confirmed with
`git check-ignore`), `config.py` loads `.env` via `python-dotenv` at
import time (no-op if the file/package is missing). `.env.example` is a
committed template with no real value. The real key never made it into
code/commits/this file.

**Verified for real in the browser** (not just tests, with a non-cached
question — otherwise you might accidentally verify a cache hit instead of
real discovery): a question about "approaches to evaluating hallucinations
in language models" surfaced sources via Tavily including **PMC (PubMed
Central), Nature.com, IEEE Xplore** — noticeably broader and more
authoritative than the earlier SearXNG-only results (mostly tag pages from
TechCrunch/Medium/aiagentstore.ai).

---

## Post-M4: faithfulness retry found nothing — force_discovery (2026-08-06)

The user got two funnel passes on the same "most-cited AI agent papers"
question, but `sources read: 0, candidates found: 0` — the loop honestly
printed "Low faithfulness — gathering more sources" but genuinely gathered
nothing.

**Cause, found by reading `agent/loop.py`, not guessed:** by this point the
persistent `research_chunks` table already held chunks from several of MY
OWN earlier test questions in this session (about AI agents, KV-cache,
LLM hallucinations — 68 chunks, `source_id`s confirmed directly via
`LanceDBStore._ensure_table().to_pandas()`). The retry reopened the
subquestions, but `_is_covered()` immediately looked at the same index
again, saw the same passing score from the old chunks (TechCrunch/Medium/
aiagentstore + an incidentally embedding-relevant LoRC) and instantly
"covered" the subquestion again — `funnel.run()` (real discovery) was
NEVER called on the retry pass at all. The culprit line:
`if not _is_covered(store, sq): funnel.run(...)` — on retry, `_is_covered`
almost always returns True again from the same persistent index.

**Fix:** `force_discovery` — a boolean flag, `True` for exactly one pass
right after a low-faithfulness retry triggers, bypasses the
`_is_covered()` check and guarantees a real call to `funnel.run()`. Unit
test `test_loop_reopens_once_when_draft_is_not_faithful` inverted (it used
to wrongly assert `funnel.run` is NOT called on retry — that assertion
was literally the bug, codified as a test).

**Confirmed by a real run** (deterministically, `_draft_is_faithful` forced
to `False` on the first draft to guarantee hitting the retry path —
otherwise LLM stochasticity makes the retry unreliable to verify by hand):
after "Low faithfulness — gathering more sources", iteration 2 now
genuinely shows "Searching sources…" → "Found 8 candidates, 3 passed
triage" → 3 new articles read via Tavily, including the actual paper
`[2508.11957] A Comprehensive Review of AI Agents`.

---

## Post-M4: wider search at the user's request (2026-08-06)

The user asked for more aggressive/broader search in `research`. Changes
in `config.py` (config-only, no funnel code rewrite):

| Parameter | Before | After |
|---|---|---|
| `FUNNEL_DISCOVERY_LIMIT_PER_SOURCE` | 5 | 10 |
| `FUNNEL_TRIAGE_TOP_N` | 3 | 6 |
| `FUNNEL_MIN_SOURCES_TO_COVER` | 2 | 3 |
| `DEFAULT_BUDGET_MAX_ITERATIONS` | 3 | 5 |
| `DEFAULT_BUDGET_MAX_DEEP_READS` | 10 | 20 |
| `DEFAULT_BUDGET_MAX_SECONDS` | 120 | 240 |

The higher `FUNNEL_MIN_SOURCES_TO_COVER` bar (2→3) doesn't just "search
wider" — it also incidentally lowers the odds of repeating the bug from
the previous section: the more distinct sources are needed for
"coverage", the less often a couple of stale chunks already in the
persistent index are enough to skip a new discovery pass.

**Confirmed by a real run** (question: "neural network model compression
methods besides quantization", never asked before — a genuine check, not a
cache hit): **19 candidates found, 6 read** (typical earlier numbers were
~5-15 candidates, 3 read) — a real, diverse mix of web (SoftServe,
Xailient, a PMC survey, a GitHub awesome-list) and Semantic Scholar
results.

---

## 7. Conventions for Claude Code (in CLAUDE.md)

- **Incrementally.** One milestone per run. Then stop, show the diff, wait
  for review. Don't get ahead of yourself.
- **`# ARCH-Q:`** — mark every assumption that can't be verified without
  the real hardware (MPS device, MLX provider call, LanceDB hybrid API,
  memory peaks) with this marker instead of silently guessing.
- **Provider seams.** External dependencies (LLM, embed, rerank, sources)
  live behind thin interfaces. No direct SDK calls from agent logic.
- **Tests offline and fast.** Mock the LLM/embeddings in tests. Unit
  tests: chunking, `ResearchState` transitions, `read_ids` dedup, budget
  stop, hybrid merge. One end-to-end integration smoke test on a tiny
  corpus.
- **Memory is a first-class citizen.** Comment where models load/release.
  LLM — one instance per process; reranker — on demand; never hold two
  resident models at once. Process the corpus with generators.
- **Small commits** per task, type hints and docstrings on public
  functions.

---

## 8. Starting prompt for Claude Code

> Read `DEVELOPMENT_PLAN.md` and `CLAUDE.md`. Implement **only Milestone
> 0**. Follow §1 (hard constraints) and §7 (conventions). Where a
> hardware assumption is needed, mark it `# ARCH-Q:` instead of guessing.
> Stop after Milestone 0 and show the diff and how to run it.
