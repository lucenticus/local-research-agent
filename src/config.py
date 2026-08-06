"""Пути, теги моделей, лимиты памяти/контекста для research-agent.

Единая точка правды по конфигурации — конкретные провайдеры (§7 CLAUDE.md)
читают эти значения, а не хардкодят свои.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "corpus"
INDEX_DIR = REPO_ROOT / ".index" / "lancedb"
INDEX_TABLE = "chunks"

# Подтверждено реальным запуском на M4 Air 2026-08-04: репо существует и
# грузится через mlx_lm.load (~2.9ГБ, уже был в HF-кэше этой машины). Проект
# по плану должен держать "свою копию" в репозитории — сейчас грузим из HF
# кэша, копирование в репо не сделано (не в скоупе Milestone 0).
LLM_HF_REPO = "mlx-community/Qwen3.5-4B-4bit"
LLM_MAX_TOKENS = 512
LLM_TEMPERATURE = 0.2

EMBED_MODEL_NAME = "BAAI/bge-m3"

# ARCH-Q: MPS выбран по умолчанию для эмбеддингов (см. DEVELOPMENT_PLAN §1).
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
FUNNEL_DISCOVERY_LIMIT_PER_SOURCE = 5
# Сколько кандидатов после триажа (скоринг abstract против подвопроса)
# реально идут на deep read.
FUNNEL_TRIAGE_TOP_N = 3
# Вес буста по цитируемости в триаже: score = cosine + CITATION_BOOST_SCALE *
# log1p(citation_count). При 0.03: 10 цитирований -> +0.07, 1000 -> +0.21,
# 10000 -> +0.28 — заметный тайбрейкер между близкими по смыслу кандидатами,
# но не способный вытащить нерелевантную статью выше релевантной (типичный
# разброс косинуса шире). См. docstring funnel.py.
CITATION_BOOST_SCALE = 0.03
# Подвопрос считается закрытым, когда retrieval из LanceDB отдаёт чанки
# минимум от стольких РАЗНЫХ источников (не просто N чанков — иначе одна
# скачанная статья с 5 чанками тривиально "закрывала" бы подвопрос).
# Эвристика gap-оценки (§7: старт с порога покрытия, не LLM).
FUNNEL_MIN_SOURCES_TO_COVER = 2
# Порог score для gap-оценки (план §6 M2/M3: "порог score + покрытие
# подвопросов", не только счётчик источников). Сырой RRF-скор гибридного
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

DEFAULT_BUDGET_MAX_ITERATIONS = 3
DEFAULT_BUDGET_MAX_DEEP_READS = 10
DEFAULT_BUDGET_MAX_SECONDS = 120.0
