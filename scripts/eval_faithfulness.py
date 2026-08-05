"""Ручной eval: citation coverage + faithfulness на эталонных вопросах.

Не pytest — реальные bge-m3 + Qwen3.5 + Qwen3-Reranker, требует построенного
индекса (`python -m src.cli index`). Переиспользует вопросы из
eval_retrieval.py (без ожидаемого source_id — здесь оценивается не
retrieval, а сам сгенерированный ответ).

Запуск:
    python -m src.cli index
    python -m scripts.eval_faithfulness
"""

from __future__ import annotations

from src import config
from src.agent.evaluate import evaluate
from src.agent.synthesize import synthesize
from src.providers import embed, rerank
from src.store.lancedb_store import LanceDBStore

from .eval_retrieval import EVAL_PAIRS


def main() -> None:
    store = LanceDBStore()
    coverage_sum = 0.0
    faithfulness_sum = 0.0

    print(f"{'Вопрос':<55} {'coverage':>9} {'faithful':>9}")
    print("-" * 75)
    for question, _expected_source in EVAL_PAIRS:
        query_vector = embed.embed_texts([question])[0]
        hits = store.search_hybrid(question, query_vector, k=config.RERANK_CANDIDATES_K)
        if config.RERANK_ENABLED:
            hits = rerank.rerank(question, hits, top_n=config.TOP_K_RETRIEVE)

        answer = synthesize(question, hits)
        result = evaluate(answer, hits)
        coverage_sum += result.citation_coverage
        faithfulness_sum += result.faithfulness

        short_q = question if len(question) <= 55 else question[:52] + "..."
        print(f"{short_q:<55} {result.citation_coverage:>9.2f} {result.faithfulness:>9.2f}")
        for claim in result.unsupported:
            print(f"    НЕПОДТВЕРЖДЕНО: {claim.sentence}")

    n = len(EVAL_PAIRS)
    print("-" * 75)
    print(f"Среднее citation coverage: {coverage_sum / n:.2f}")
    print(f"Среднее faithfulness: {faithfulness_sum / n:.2f}")


if __name__ == "__main__":
    main()
