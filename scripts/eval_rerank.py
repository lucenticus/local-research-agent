"""Ручной eval: улучшает ли реранк порядок выдачи (hit@1) поверх hybrid-поиска.

Использует тот же golden-set, что и остальные eval-скрипты (eval_data.py).
Recall@3 в eval_retrieval.py уже насыщен (корпус мал), поэтому здесь смотрим
на более строгую метрику: попадает ли ожидаемый источник на позицию #1.
Именно это реранкер и должен улучшать поверх грубого RRF-слияния.

Не pytest — реальные bge-m3 + Qwen3-Reranker-0.6B-4bit, требует построенного
индекса.

Запуск:
    python -m src.cli index
    python -m scripts.eval_rerank
"""

from __future__ import annotations

from src import config
from src.providers import embed, rerank
from src.store.qdrant_store import QdrantStore

from .eval_data import EVAL_CASES


def hit_at_1(hits: list[dict], expected_source_id: str) -> bool:
    return bool(hits) and hits[0].get("source_id") == expected_source_id


def main() -> None:
    store = QdrantStore()
    hybrid_hits_count = 0
    reranked_hits_count = 0

    print(f"{'Вопрос':<55} {'hybrid':>8} {'+rerank':>8}")
    print("-" * 73)
    for case in EVAL_CASES:
        query_vector = embed.embed_texts([case.question])[0]
        candidates = store.search_hybrid(case.question, query_vector, k=config.RERANK_CANDIDATES_K)
        reranked = rerank.rerank(case.question, candidates, top_n=config.TOP_K_RETRIEVE)

        hybrid_ok = hit_at_1(candidates, case.expected_source_id)
        reranked_ok = hit_at_1(reranked, case.expected_source_id)
        hybrid_hits_count += hybrid_ok
        reranked_hits_count += reranked_ok

        short_q = case.question if len(case.question) <= 55 else case.question[:52] + "..."
        print(f"{short_q:<55} {'OK' if hybrid_ok else 'MISS':>8} {'OK' if reranked_ok else 'MISS':>8}")

    n = len(EVAL_CASES)
    print("-" * 73)
    print(f"Hit@1: hybrid = {hybrid_hits_count}/{n}, +rerank = {reranked_hits_count}/{n}")
    if reranked_hits_count < hybrid_hits_count:
        print("ВНИМАНИЕ: реранк ухудшил порядок — критерий Milestone 2 не выполнен.")
    else:
        print("Реранк не хуже (или лучше) hybrid по hit@1.")


if __name__ == "__main__":
    main()
