"""RAGAS context-quality eval: ContextPrecision + ContextRecall — a finer
grain than eval_retrieval.py's recall@k/MRR. Recall@k/MRR ask "did the
retriever return the right *document*"; context precision/recall ask "of
the chunks actually fed to synthesis, how many are useful (precision), and
how much of what's needed is covered (recall)" — a document can be the
right one and still contribute a chunk that isn't actually relevant to the
question, which recall@k/MRR can't see.

Judge is `ChatMLX` (`providers/langchain_llm.py`) wrapped in RAGAS's
`LangchainLLMWrapper` — the resident Qwen3.5, same model as every other
judge in this project (eval_correctness.py, agent/evaluate.py's
faithfulness check), not a new model/API dependency. Verified working
end-to-end against this project's real retrieval pipeline before adding
(RAGAS's prompts ask the LLM for structured JSON verdicts — confirmed
Qwen3.5 produces output RAGAS can parse).

RAGAS version pin matters here: `ragas>=0.4,<0.5`'s `ragas/llms/base.py`
unconditionally imports `langchain_community.chat_models.vertexai` at
module load time — removed in `langchain-community>=0.4` (which says it's
"being sunset" upstream in favor of standalone integration packages), so
`langchain-community==0.3.30` is pinned in requirements.txt too, or `import
ragas` itself crashes. Also: `LangchainLLMWrapper` + the plain
`LLMContextPrecisionWithReference`/`LLMContextRecall` metric classes used
here are RAGAS's *legacy* API — the docs mark them deprecated in favor of a
new "collections" API built around raw OpenAI clients (`ragas.llms.
llm_factory` + `AsyncOpenAI`), which has no documented path for a custom
LangChain `BaseChatModel` like `ChatMLX`. The legacy path is what actually
works without an OpenAI dependency; if a future RAGAS major version drops
it, this script needs revisiting, not the rest of the codebase.

Не pytest — реальные bge-m3 + Qwen3.5, требует построенного индекса.

Запуск:
    python -m src.cli index
    python -m scripts.eval_ragas
"""

from __future__ import annotations

import warnings

from src import config
from src.providers import embed
from src.store.lancedb_store import LanceDBStore

from .eval_data import EVAL_CASES

K = 3


def main() -> None:
    # RAGAS's LangchainLLMWrapper/metric classes are its own deprecated-but-
    # functional legacy API (see module docstring) — this warning is
    # expected noise, not something wrong with this integration.
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")

    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall

    from src.providers.langchain_llm import ChatMLX

    judge = LangchainLLMWrapper(ChatMLX())
    precision_metric = LLMContextPrecisionWithReference(llm=judge)
    recall_metric = LLMContextRecall(llm=judge)

    from ragas.dataset_schema import SingleTurnSample

    store = LanceDBStore()
    precision_sum = 0.0
    recall_sum = 0.0

    print(f"{'Вопрос':<55} {'precision':>10} {'recall':>8}")
    print("-" * 76)
    for case in EVAL_CASES:
        query_vector = embed.embed_texts([case.question])[0]
        hits = store.search_hybrid(case.question, query_vector, k=K)
        contexts = [hit["text"] for hit in hits]

        sample = SingleTurnSample(
            user_input=case.question, retrieved_contexts=contexts, reference=case.reference_answer
        )
        precision = precision_metric.single_turn_score(sample)
        recall = recall_metric.single_turn_score(sample)
        precision_sum += precision
        recall_sum += recall

        short_q = case.question if len(case.question) <= 55 else case.question[:52] + "..."
        print(f"{short_q:<55} {precision:>10.2f} {recall:>8.2f}")

    n = len(EVAL_CASES)
    print("-" * 76)
    print(f"Среднее context precision: {precision_sum / n:.2f}")
    print(f"Среднее context recall: {recall_sum / n:.2f}")


if __name__ == "__main__":
    main()
