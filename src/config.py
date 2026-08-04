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

# ARCH-Q: "своя копия Qwen3.5" по плану должна лежать в репозитории — точный
# HF-репо/тег квантованной MLX-версии не проверен на этой машине. Дефолт ниже
# взят по образцу ~/Dev/storyreel/storyreel/providers/plan/local_mlx.py
# (там mlx-community/Qwen3.5-4B-4bit) — подтвердить при первом реальном запуске
# на M4 Air и заменить на путь к локальной копии, если она уже скачана.
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
