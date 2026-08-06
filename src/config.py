"""Пути, теги моделей, лимиты памяти/контекста для research-agent.

Единая точка правды по конфигурации — конкретные провайдеры (§7 CLAUDE.md)
читают эти значения, а не хардкодят свои.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# .env — секреты (API-ключи), НИКОГДА не коммитится (.gitignore). Не падаем,
# если python-dotenv не установлен или файла нет — ключи тогда просто не
# подхватятся, источники, которым они нужны (TavilySource), тихо отключатся.
try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass
CORPUS_DIR = REPO_ROOT / "corpus"
INDEX_DIR = REPO_ROOT / ".index" / "lancedb"
INDEX_TABLE = "chunks"
# Отдельная таблица для `research` (Milestone 3+), в той же LanceDB-базе.
# Найдено реальным прогоном 2026-08-06: `research` и `ask` изначально делили
# одну таблицу — тестовый demo-корпус (corpus/) просачивался в источники
# реальных research-ответов на никак не связанные вопросы (например,
# заметка про hybrid retrieval попала в ответ про "AI агентов" только потому,
# что была в том же индексе). `ask`/`index` работают с локальными файлами
# corpus/, `research`/`serve` — с находками воронки; общий кэш между ними не
# нужен и вреден.
RESEARCH_INDEX_TABLE = "research_chunks"

# Подтверждено реальным запуском на M4 Air 2026-08-04: репо существует и
# грузится через mlx_lm.load (~2.9ГБ, уже был в HF-кэше этой машины). Проект
# по плану должен держать "свою копию" в репозитории — сейчас грузим из HF
# кэша, копирование в репо не сделано (не в скоупе Milestone 0).
LLM_HF_REPO = "mlx-community/Qwen3.5-4B-4bit"
LLM_MAX_TOKENS = 512
LLM_TEMPERATURE = 0.2

EMBED_MODEL_NAME = "BAAI/bge-m3"

# ARCH-Q: MPS выбран по умолчанию для эмбеддингов.
# Фолбэк на CPU под давлением памяти рядом с резидентной LLM в Milestone 0 не
# реализован (эмбеддинг корпуса и LLM в этом милестоне не грузятся одновременно
# долго) — добавить явный фолбэк в Milestone 1/2, когда это станет актуальным.
EMBED_DEVICE = "mps"

CHUNK_SIZE_CHARS = 800
CHUNK_OVERLAP_CHARS = 100

# Жёсткий потолок контекста синтеза (§1: длинный контекст раздувает KV-cache —
# главный риск OOM на 16 ГБ).
MAX_SYNTHESIS_CONTEXT_CHARS = 6000

TOP_K_RETRIEVE = 5

# Изначально планировался BAAI/bge-reranker-v2-m3 (FlagEmbedding), но он
# конфликтует по версии с mlx_lm: FlagReranker вызывает
# tokenizer.prepare_for_model, метод удалён в transformers>=5, а mlx_lm
# (LLM-провайдер) требует transformers>=5 (TokenizersBackend) — подтверждено
# реальным запуском на M4 Air 2026-08-05. Переключились на чистый MLX
# реранкер: конфликта нет, т.к. в пути скоринга нет torch/transformers вовсе.
RERANK_HF_REPO = "mlx-community/Qwen3-Reranker-0.6B-4bit"
# Флаг для отключения реранка (§1: реранкер по требованию, не резидентно
# рядом с LLM) — при выключении cmd_ask берёт top-k прямо из hybrid-поиска.
RERANK_ENABLED = True
# Сколько кандидатов из hybrid-поиска отдать реранкеру перед тем, как он
# урежет их до TOP_K_RETRIEVE.
RERANK_CANDIDATES_K = 10

# --- Milestone 3: воронка + итеративный цикл ---

# Сколько кандидатов запрашивать у каждого источника на подвопрос (discovery,
# только метаданные — §1: полный текст только для прошедших триаж).
# Увеличено 2026-08-06 по запросу пользователя ("более агрессивный/широкий
# поиск") с 5 — воронка была слишком быстро довольна малой выдачей.
FUNNEL_DISCOVERY_LIMIT_PER_SOURCE = 10
# Сколько кандидатов после триажа (скоринг abstract против подвопроса)
# реально идут на deep read. Увеличено 2026-08-06 с 3 — та же причина.
FUNNEL_TRIAGE_TOP_N = 6
# Вес буста по цитируемости в триаже: score = cosine + CITATION_BOOST_SCALE *
# log1p(citation_count). При 0.03: 10 цитирований -> +0.07, 1000 -> +0.21,
# 10000 -> +0.28 — заметный тайбрейкер между близкими по смыслу кандидатами,
# но не способный вытащить нерелевантную статью выше релевантной (типичный
# разброс косинуса шире). См. docstring funnel.py.
CITATION_BOOST_SCALE = 0.03
# Подвопрос считается закрытым, когда retrieval из LanceDB отдаёт чанки
# минимум от стольких РАЗНЫХ источников (не просто N чанков — иначе одна
# скачанная статья с 5 чанками тривиально "закрывала" бы подвопрос).
# Эвристика gap-оценки (§7: старт с порога покрытия, не LLM). Увеличено
# 2026-08-06 с 2 — по запросу пользователя на более широкий поиск: чем выше
# планка, тем реже воронка довольствуется парой старых чанков из
# персистентного индекса вместо реального нового discovery.
FUNNEL_MIN_SOURCES_TO_COVER = 3
# Порог score для gap-оценки ("порог score + покрытие подвопросов", не
# только счётчик источников). Сырой RRF-скор гибридного
# поиска некалиброван (узкий диапазон ~0.01-0.03, порог пришлось бы гадать),
# поэтому берём готовый реранкер: P(yes) >= 0.5 — только чанки, которые
# реранкер сам считает более релевантными, чем нет, идут в счёт покрытия.
# Найдено реальным прогоном 2026-08-05: без этого фильтра тематически
# смежный, но нерелевантный локальный корпус ложно "закрывал" подвопрос.
FUNNEL_MIN_RERANK_SCORE = 0.5

# sources/web.py — локальный SearXNG (docker compose up -d, см. README).
# Публичные инстансы/DuckDuckGo HTML не работают без ключа (проверено вручную
# 2026-08-05: CAPTCHA у DDG, rate-limit/отключённый JSON у публичных SearXNG).
SEARXNG_BASE_URL = "http://localhost:8888"

# --- Milestone 4: самопроверка ---

# Порог faithfulness (agent/evaluate.py) ниже которого loop.py считает
# черновой ответ недостаточно обоснованным и переоткрывает подвопросы для
# ещё одного прохода воронки (в пределах budget). Тот же 0.5, что и
# FUNNEL_MIN_RERANK_SCORE — обе метрики читают один калиброванный скор.
EVAL_FAITHFULNESS_THRESHOLD = 0.5
# Потолок на размер скачиваемого PDF (байты) — не тянуть произвольно большие
# файлы с внешних источников без ограничения.
FUNNEL_MAX_PDF_BYTES = 30_000_000

# Увеличено 2026-08-06 по запросу пользователя на более агрессивный/широкий
# поиск: больше проходов, больше прочитанных источников, больше времени на
# это (шире discovery и триаж, см. выше, сами по себе требуют больше
# времени — PDF/embed/Tavily-вызовов больше на каждый проход).
DEFAULT_BUDGET_MAX_ITERATIONS = 5
DEFAULT_BUDGET_MAX_DEEP_READS = 20
DEFAULT_BUDGET_MAX_SECONDS = 240.0

# --- MCP integrations (providers/mcp_client.py) ---

# Off by default: spawns an external `uvx`/`npx` MCP-server subprocess per
# call, an extra dependency (uvx/npx must be installed, network egress on
# every deep-read) beyond this project's own HTTP clients. Opt in explicitly
# once the MCP server(s) below are available on the machine — see README.
MCP_FETCH_ENABLED = False
# mcp-server-fetch pinned to `mcp<2`: its 2026.7.10 release imports
# `McpError` from `mcp.shared.exceptions`, renamed to `MCPError` in `mcp`
# 2.0 - confirmed by a real `uvx mcp-server-fetch --help` crash on this
# machine (mcp 2.0.0 resolved by uvx by default at time of writing).
MCP_FETCH_COMMAND = "uvx"
MCP_FETCH_ARGS = ["--from", "mcp-server-fetch", "--with", "mcp<2", "mcp-server-fetch"]
# Per-page cap fed into chunking - same rationale as FUNNEL_MAX_PDF_BYTES,
# just character-based since mcp-server-fetch already returns text, not bytes.
MCP_FETCH_MAX_CHARS = 8000

# Off by default, same rationale as MCP_FETCH_ENABLED - spawns a Docker
# container per call (real per-question latency), an extra dependency
# (Docker + GITHUB_PERSONAL_ACCESS_TOKEN) beyond this project's own HTTP
# clients. Also gated on the token actually being set - see
# sources/github_mcp.py.
GITHUB_MCP_ENABLED = False
