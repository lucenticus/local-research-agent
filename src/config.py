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
