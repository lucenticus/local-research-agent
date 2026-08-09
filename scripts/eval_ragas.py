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
faithfulness check), not a new model/API dependency.

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

Uploads to the same LangSmith dataset as eval_correctness.py
(`config.LANGSMITH_EVAL_DATASET`) as a separate experiment — same dataset,
different target (retrieved contexts, not a generated answer), so both are
browsable/comparable side by side in the LangSmith UI instead of only a
console table. Needs `LANGSMITH_API_KEY` in `.env`.

Не pytest — реальные bge-m3 + Qwen3.5 + LangSmith API, требует построенного
индекса.

Запуск:
    python -m src.cli index
    python -m scripts.eval_ragas
"""

from __future__ import annotations

import warnings

from src.providers import embed
from src.store.lancedb_store import LanceDBStore

from . import eval_data
from .eval_data import require_langsmith_api_key

K = 3


def main() -> None:
    require_langsmith_api_key()

    # RAGAS's LangchainLLMWrapper/metric classes are its own deprecated-but-
    # functional legacy API (see module docstring) — this warning is
    # expected noise, not something wrong with this integration.
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")

    from langsmith import Client, evaluate
    from ragas.dataset_schema import SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall

    from src import config
    from src.providers.langchain_llm import ChatMLX

    judge = LangchainLLMWrapper(ChatMLX())
    precision_metric = LLMContextPrecisionWithReference(llm=judge)
    recall_metric = LLMContextRecall(llm=judge)

    client = Client()
    dataset_name = eval_data.ensure_dataset(
        client, config.LANGSMITH_EVAL_DATASET,
        description="local-research-agent: golden Q&A eval over corpus/ (correctness + RAGAS context quality)",
    )
    store = LanceDBStore()

    def target(inputs: dict) -> dict:
        question = inputs["question"]
        query_vector = embed.embed_texts([question])[0]
        hits = store.search_hybrid(question, query_vector, k=K)
        return {"retrieved_contexts": [hit["text"] for hit in hits]}

    def _sample(run, example) -> "SingleTurnSample":
        return SingleTurnSample(
            user_input=example.inputs["question"],
            retrieved_contexts=run.outputs["retrieved_contexts"],
            reference=example.outputs["reference_answer"],
        )

    def context_precision(run, example) -> dict:
        return {"key": "context_precision", "score": precision_metric.single_turn_score(_sample(run, example))}

    def context_recall(run, example) -> dict:
        return {"key": "context_recall", "score": recall_metric.single_turn_score(_sample(run, example))}

    results = evaluate(
        target,
        data=dataset_name,
        evaluators=[context_precision, context_recall],
        experiment_prefix="ragas-context",
        description="RAGAS context precision/recall (judge: resident ChatMLX) over corpus/ golden Q&A",
    )

    def _scores_by_key(rows: list, key: str) -> list[float]:
        return [
            r.score
            for row in rows
            for r in row["evaluation_results"]["results"]
            if r.key == key
        ]

    rows = list(results)
    precision_scores = _scores_by_key(rows, "context_precision")
    recall_scores = _scores_by_key(rows, "context_recall")
    print(f"\nСреднее context precision: {sum(precision_scores) / len(precision_scores):.2f}")
    print(f"Среднее context recall: {sum(recall_scores) / len(recall_scores):.2f}")
    if results.url:
        print(f"LangSmith: {results.url}")


if __name__ == "__main__":
    main()
