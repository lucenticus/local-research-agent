"""Ручной eval retrieval@k + MRR: dense-only vs hybrid (dense+FTS/RRF).

Recall@k — бинарная метрика (нашли/не нашли), она насыщается быстро на
небольшом корпусе (см. eval_data.py) и не различает "нашли на #1" от "нашли
на #3". MRR (Mean Reciprocal Rank, 1/rank, 0 если не нашли) использует уже
вычисленные ранги, которые recall@k просто отбрасывает — не требует новых
меток, только более честно показывает качество ранжирования.

Не pytest-тест: гоняет реальные bge-m3 эмбеддинги (см. CLAUDE.md — юнит-тесты
обязаны быть офлайн/мокнутыми, а измерение качества ретривала без реальной
модели бессмысленно). Индекс должен быть уже построен (`python -m src.cli index`).

Запуск:
    python -m scripts.eval_retrieval
"""

from __future__ import annotations

from src.providers import embed
from src.store.lancedb_store import LanceDBStore

from .eval_data import EVAL_CASES

K = 3


def recall_at_k(hits: list[dict], expected_source_id: str) -> bool:
    return any(hit.get("source_id") == expected_source_id for hit in hits)


def reciprocal_rank(hits: list[dict], expected_source_id: str) -> float:
    for rank, hit in enumerate(hits, start=1):
        if hit.get("source_id") == expected_source_id:
            return 1.0 / rank
    return 0.0


def main() -> None:
    store = LanceDBStore()
    dense_hits_count = 0
    hybrid_hits_count = 0
    dense_rr_sum = 0.0
    hybrid_rr_sum = 0.0

    print(f"{'Вопрос':<55} {'dense':>7} {'hybrid':>7} {'d-RR':>6} {'h-RR':>6}")
    print("-" * 84)
    for case in EVAL_CASES:
        query_vector = embed.embed_texts([case.question])[0]
        dense_hits = store.search_dense(query_vector, k=K)
        hybrid_hits = store.search_hybrid(case.question, query_vector, k=K)

        dense_ok = recall_at_k(dense_hits, case.expected_source_id)
        hybrid_ok = recall_at_k(hybrid_hits, case.expected_source_id)
        dense_hits_count += dense_ok
        hybrid_hits_count += hybrid_ok

        dense_rr = reciprocal_rank(dense_hits, case.expected_source_id)
        hybrid_rr = reciprocal_rank(hybrid_hits, case.expected_source_id)
        dense_rr_sum += dense_rr
        hybrid_rr_sum += hybrid_rr

        short_q = case.question if len(case.question) <= 55 else case.question[:52] + "..."
        print(
            f"{short_q:<55} {'OK' if dense_ok else 'MISS':>7} {'OK' if hybrid_ok else 'MISS':>7} "
            f"{dense_rr:>6.2f} {hybrid_rr:>6.2f}"
        )

    n = len(EVAL_CASES)
    print("-" * 84)
    print(f"Recall@{K}: dense = {dense_hits_count}/{n}, hybrid = {hybrid_hits_count}/{n}")
    print(f"MRR: dense = {dense_rr_sum / n:.3f}, hybrid = {hybrid_rr_sum / n:.3f}")
    if hybrid_hits_count < dense_hits_count:
        print("ВНИМАНИЕ: hybrid хуже dense по recall — критерий Milestone 1 не выполнен.")
    else:
        print("hybrid не хуже dense — критерий Milestone 1 выполнен.")


if __name__ == "__main__":
    main()
