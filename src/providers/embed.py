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

    # ARCH-Q: точное имя kwarg устройства в установленной версии FlagEmbedding
    # (device= vs devices=) не проверено на этой машине — пробуем варианты,
    # последний фолбэк отдаёт выбор устройства библиотеке по умолчанию.
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
