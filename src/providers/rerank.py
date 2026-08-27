"""Qwen3-Reranker-0.6B-4bit (чистый MLX): load-on-demand, score, release.

Изначально планировался BAAI/bge-reranker-v2-m3 через FlagEmbedding, но он
требует `tokenizer.prepare_for_model`, метод удалён в transformers>=5, а
mlx_lm (наш LLM-провайдер) требует transformers>=5 (TokenizersBackend) —
конфликт версий в одном venv подтверждён реальным запуском. Qwen3-Reranker
грузится через тот же `mlx_lm.load()`, что и основная LLM, без torch/
transformers в пути скоринга — конфликта нет. Мультиязычный, русский
проверен вручную (relevant≈1.0 / irrelevant≈0.00005 на тестовой паре).

По требованию (§1 CLAUDE.md): в отличие от providers/embed.py и
providers/llm.py, эта модель НЕ резидентна между вызовами — rerank() грузит,
скорит все кандидаты, освобождает и только потом возвращает результат.
"""

from __future__ import annotations

import gc
from typing import Any

from .. import config
from . import metrics

_INSTRUCT = "Given a search query, retrieve relevant passages that answer the query"
_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements "
    'based on the Query and the Instruct provided. Note that the answer can '
    'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _load() -> tuple[Any, Any]:
    from mlx_lm import load  # lazy import

    # Считается отдельно от скоринга: по инварианту памяти (§1) реранкер
    # грузится заново на КАЖДЫЙ вызов, и эта цена должна быть видна в
    # отчёте, а не растворяться во времени скоринга.
    with metrics.track("rerank.load"):
        return load(config.RERANK_HF_REPO)


def _release(model: Any, tokenizer: Any) -> None:
    del model, tokenizer
    gc.collect()
    import mlx.core as mx

    mx.clear_cache()


def _score_one(model: Any, tokenizer: Any, query: str, text: str) -> float:
    """Одна (query, text) пара -> P(yes) на последнем токене промпта."""
    import mlx.core as mx

    hf = getattr(tokenizer, "_tokenizer", tokenizer)
    true_id = hf.convert_tokens_to_ids("yes")
    false_id = hf.convert_tokens_to_ids("no")
    pre = hf.encode(_PREFIX, add_special_tokens=False)
    suf = hf.encode(_SUFFIX, add_special_tokens=False)

    content = f"<Instruct>: {_INSTRUCT}\n<Query>: {query}\n<Document>: {text}"
    ids = pre + hf.encode(content, add_special_tokens=False) + suf
    logits = model(mx.array([ids]))[:, -1, :]
    pair = mx.stack([logits[0, false_id], logits[0, true_id]])
    return float(mx.exp((pair - mx.logsumexp(pair))[1]))


def _score_batch(model: Any, tokenizer: Any, query: str, texts: list[str]) -> list[float]:
    """Один и тот же query против нескольких текстов (retrieval-реранк)."""
    return [_score_one(model, tokenizer, query, text) for text in texts]


def score(query: str, candidates: list[dict[str, Any]]) -> list[tuple[dict[str, Any], float]]:
    """Скорит candidates по query, возвращает (candidate, score) в исходном порядке.

    Модель грузится и освобождается в пределах одного вызова (§1). Используется
    и `rerank()` (сортировка + top_n), и agent/loop.py (gap-оценка по порогу
    score, без сортировки/обрезки).
    """
    if not candidates:
        return []
    model, tokenizer = _load()
    try:
        with metrics.track("rerank.score", items=len(candidates)):
            scores = _score_batch(model, tokenizer, query, [c["text"] for c in candidates])
    finally:
        _release(model, tokenizer)
    return list(zip(candidates, scores, strict=True))


def rerank(query: str, candidates: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    """Скорит candidates по query, возвращает top_n по убыванию скора."""
    scored = sorted(score(query, candidates), key=lambda pair: pair[1], reverse=True)
    return [candidate for candidate, _ in scored[:top_n]]


def score_pairs(pairs: list[tuple[str, str]]) -> list[float]:
    """Скорит произвольные (query, text) пары одним load/release.

    В отличие от `score()` (один query, много кандидатов), здесь у каждой
    пары свой query — нужно для faithfulness-проверки (agent/evaluate.py):
    у каждого утверждения в ответе свой текст, который проверяется против
    текста именно того чанка, на который оно ссылается.
    """
    if not pairs:
        return []
    model, tokenizer = _load()
    try:
        with metrics.track("rerank.score_pairs", items=len(pairs)):
            return [_score_one(model, tokenizer, query, text) for query, text in pairs]
    finally:
        _release(model, tokenizer)
