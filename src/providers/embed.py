"""bge-m3 embedding provider — dense (всегда) и sparse (по запросу).

Раньше (Milestone 1) sparse-выход bge-m3 сознательно не использовался —
лексический сигнал давал LanceDB FTS (tantivy) поверх сырого текста, без
эмбеддинга запроса. При переходе на Qdrant для гибридного поиска нужен
именно sparse-вектор (learned lexical weights, не BM25-по-корпусу), и raw
FlagEmbedding.encode() уже умеет отдавать его в том же forward pass, что и
dense — просто `return_sparse=True` было `False`.

Резидентная модель — грузится один раз при первом вызове и переиспользуется
процессом (§1 CLAUDE.md: не инстанцировать модель повторно).
"""

from __future__ import annotations

from typing import Any

from .. import config
from . import metrics

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
    with metrics.track("embed.embed_texts", items=len(texts)):
        model = _load()
        out = model.encode(texts, return_dense=True, return_sparse=False, return_colbert_vecs=False)
        return [vec.tolist() for vec in out["dense_vecs"]]


def _to_sparse_dict(lexical_weights: dict) -> dict[int, float]:
    """bge-m3 отдаёт lexical_weights как {str(token_id): np.float16(weight)} —
    Qdrant SparseVector хочет int-индексы и обычный float."""
    return {int(token_id): float(weight) for token_id, weight in lexical_weights.items()}


def embed_texts_hybrid(texts: list[str]) -> tuple[list[list[float]], list[dict[int, float]]]:
    """Dense + sparse одним forward pass'ом — используется при индексации
    чанков (store/qdrant_store.py), где текст и так уже под рукой и оба
    представления нужны сразу, чтобы не гонять модель дважды на одном тексте."""
    if not texts:
        return [], []
    with metrics.track("embed.embed_texts_hybrid", items=len(texts)):
        model = _load()
        out = model.encode(texts, return_dense=True, return_sparse=True, return_colbert_vecs=False)
        dense = [vec.tolist() for vec in out["dense_vecs"]]
        sparse = [_to_sparse_dict(w) for w in out["lexical_weights"]]
        return dense, sparse


def embed_sparse(texts: list[str]) -> list[dict[int, float]]:
    """Только sparse — используется для query-стороны гибридного поиска
    (store/qdrant_store.py::search_hybrid получает dense-вектор запроса от
    вызывающего кода уже готовым, как и раньше для LanceDB, и сам добирает
    sparse-сторону здесь, чтобы не менять сигнатуру search_hybrid и все её
    вызовы по всему проекту)."""
    if not texts:
        return []
    with metrics.track("embed.embed_sparse", items=len(texts)):
        model = _load()
        out = model.encode(texts, return_dense=False, return_sparse=True, return_colbert_vecs=False)
        return [_to_sparse_dict(w) for w in out["lexical_weights"]]
