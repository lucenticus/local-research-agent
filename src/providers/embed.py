"""bge-m3 dense embedding provider.

Milestone 0: только dense-выход (sparse — Milestone 1). Резидентная модель —
грузится один раз при первом вызове и переиспользуется процессом (§1
CLAUDE.md: не инстанцировать модель повторно).
"""

from __future__ import annotations

from typing import Any

from .. import config

_model: Any = None


def _load() -> Any:
    global _model
    if _model is not None:
        return _model
    from FlagEmbedding import BGEM3FlagModel  # lazy import: тяжёлая torch-зависимость

    # Подтверждено реальным запуском на M4 Air 2026-08-04: kwarg device=
    # принимается этой версией FlagEmbedding, модель реально грузится на MPS
    # (next(model.model.parameters()).device == mps:0). Варианты ниже оставлены
    # как защита от дрейфа версии библиотеки.
    #
    # ВАЖНО: BGEM3FlagModel сам вызывает snapshot_download БЕЗ allow_patterns и
    # докачивает весь репозиторий (~4.3ГБ: pytorch_model.bin + дублирующий onnx/
    # + imgs/), даже если нужные для инференса файлы уже скачаны отдельно.
    # Тот же паттерн, что и с ltx-2-mlx в storyreel — библиотека не разделяет
    # "что реально используется" от "что лежит в репозитории".
    last_error: Exception | None = None
    for kwargs in ({"device": config.EMBED_DEVICE}, {"devices": [config.EMBED_DEVICE]}, {}):
        try:
            _model = BGEM3FlagModel(config.EMBED_MODEL_NAME, use_fp16=True, **kwargs)
            return _model
        except TypeError as exc:
            last_error = exc
            continue
    raise RuntimeError(f"Не удалось загрузить {config.EMBED_MODEL_NAME}: {last_error}")


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Dense-эмбеддинги для списка текстов. Пустой вход -> пустой выход."""
    if not texts:
        return []
    model = _load()
    out = model.encode(texts, return_dense=True, return_sparse=False, return_colbert_vecs=False)
    return [vec.tolist() for vec in out["dense_vecs"]]
