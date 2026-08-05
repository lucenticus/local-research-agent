"""Ручной eval retrieval@k: dense-only vs hybrid (dense+FTS/RRF).

Не pytest-тест: гоняет реальные bge-m3 эмбеддинги (см. CLAUDE.md — юнит-тесты
обязаны быть офлайн/мокнутыми, а измерение качества ретривала без реальной
модели бессмысленно). Индекс должен быть уже построен (`python -m src.cli index`).

Запуск:
    python -m scripts.eval_retrieval
"""

from __future__ import annotations

from src import config
from src.providers import embed
from src.store.lancedb_store import LanceDBStore

# (вопрос, ожидаемый source_id) — по одному-двум вопросу на каждый документ
# корпуса, плюс вопросы с точными терминами/аббревиатурами, где лексический
# сигнал (FTS) может опередить чистый dense.
EVAL_PAIRS: list[tuple[str, str]] = [
    ("Что такое Retrieval-Augmented Generation?", "retrieval_augmented_generation.md"),
    ("Почему RAG снижает галлюцинации модели?", "retrieval_augmented_generation.md"),
    ("Чем векторный поиск отличается от поиска по ключевым словам?", "vector_databases.md"),
    ("Что такое LanceDB и зачем она нужна?", "vector_databases.md"),
    ("Что такое квантование весов языковой модели?", "quantized_local_llm.md"),
    ("Почему нельзя держать несколько резидентных моделей на 16 ГБ памяти?", "quantized_local_llm.md"),
    ("Что такое BM25 и как оно связано с RRF?", "hybrid_retrieval_paper.md"),
    ("Что показали результаты эксперимента с гибридным поиском?", "hybrid_retrieval_paper.md"),
]

K = 3


def recall_at_k(hits: list[dict], expected_source_id: str) -> bool:
    return any(hit.get("source_id") == expected_source_id for hit in hits)


def main() -> None:
    store = LanceDBStore()
    dense_hits_count = 0
    hybrid_hits_count = 0

    print(f"{'Вопрос':<55} {'dense':>7} {'hybrid':>7}")
    print("-" * 71)
    for question, expected in EVAL_PAIRS:
        query_vector = embed.embed_texts([question])[0]
        dense_hits = store.search_dense(query_vector, k=K)
        hybrid_hits = store.search_hybrid(question, query_vector, k=K)

        dense_ok = recall_at_k(dense_hits, expected)
        hybrid_ok = recall_at_k(hybrid_hits, expected)
        dense_hits_count += dense_ok
        hybrid_hits_count += hybrid_ok

        short_q = question if len(question) <= 55 else question[:52] + "..."
        print(f"{short_q:<55} {'OK' if dense_ok else 'MISS':>7} {'OK' if hybrid_ok else 'MISS':>7}")

    n = len(EVAL_PAIRS)
    print("-" * 71)
    print(f"Recall@{K}: dense = {dense_hits_count}/{n}, hybrid = {hybrid_hits_count}/{n}")
    if hybrid_hits_count < dense_hits_count:
        print("ВНИМАНИЕ: hybrid хуже dense — критерий Milestone 1 не выполнен.")
    else:
        print("hybrid не хуже dense — критерий Milestone 1 выполнен.")


if __name__ == "__main__":
    main()
