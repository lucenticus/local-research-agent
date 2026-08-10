"""LLM-as-judge correctness eval — closes a gap the other eval scripts don't
cover. `eval_retrieval.py`/`eval_rerank.py` check whether the right chunk was
found; `eval_faithfulness.py` checks whether cited claims are grounded in the
chunk they cite. None of them check whether the generated answer actually
gets the question *right* — a fully-cited, fully-grounded answer can still
miss the point of the question.

Judge is the resident Qwen3.5 (`llm.generate`) — same bounded-LLM pattern as
`funnel.py`'s query translation, not a separate model/API call. Binary
yes/no verdict against a hand-written reference answer, not a 1-5 score —
matches the yes/no calibration already used for faithfulness/gap-check
elsewhere in this project, and is more reproducible than a numeric scale a
4B judge model would apply inconsistently.

Uploads to a LangSmith dataset/experiment (`config.LANGSMITH_EVAL_DATASET`)
via `langsmith.evaluate()` instead of just printing a table like the other
eval scripts — the value here is comparing runs over time in the UI (did
correctness regress after a synthesis-prompt change?), not a one-off number.
Needs `LANGSMITH_API_KEY` in `.env`; independent of
`config.LANGSMITH_TRACING_ENABLED` (that flag gates passive production
tracing — this script's entire point is an explicit LangSmith upload).

Не pytest — реальные bge-m3 + Qwen3.5 + Qwen3-Reranker + LangSmith API,
требует построенного индекса.

Запуск:
    python -m src.cli index
    python -m scripts.eval_correctness
"""

from __future__ import annotations

from src import config
from src.agent.synthesize import synthesize
from src.providers import embed, llm, rerank
from src.store.qdrant_store import QdrantStore

from .eval_data import ensure_dataset, require_langsmith_api_key

_JUDGE_SYSTEM_PROMPT = (
    "Ты — строгий эксперт-проверяющий. Тебе дан вопрос, эталонный ответ и "
    "проверяемый ответ. Определи, верно ли проверяемый ответ отвечает на "
    "вопрос и не противоречит эталонному ответу по сути (дословное "
    "совпадение не требуется, парафраз — это нормально). Ответь РОВНО одним "
    'словом: "yes", если ответ по сути верный, "no" — если неверный, '
    "противоречит эталону или не отвечает на вопрос. Больше ничего не пиши."
)


def _judge(question: str, reference: str, generated: str) -> bool:
    user_message = (
        f"Вопрос: {question}\n\nЭталонный ответ: {reference}\n\nПроверяемый ответ: {generated}"
    )
    prompt = llm.build_chat_prompt(_JUDGE_SYSTEM_PROMPT, user_message)
    verdict = llm.generate(prompt, max_tokens=8, temperature=0.0).strip().lower()
    return verdict.startswith("yes")


def _answer(store: QdrantStore, question: str) -> str:
    query_vector = embed.embed_texts([question])[0]
    hits = store.search_hybrid(question, query_vector, k=config.RERANK_CANDIDATES_K)
    if config.RERANK_ENABLED:
        hits = rerank.rerank(question, hits, top_n=config.TOP_K_RETRIEVE)
    return synthesize(question, hits)


def main() -> None:
    require_langsmith_api_key()

    from langsmith import Client, evaluate

    client = Client()
    dataset_name = ensure_dataset(
        client, config.LANGSMITH_EVAL_DATASET,
        description="local-research-agent: golden Q&A eval over corpus/ (correctness + RAGAS context quality)",
    )
    store = QdrantStore()

    def target(inputs: dict) -> dict:
        return {"answer": _answer(store, inputs["question"])}

    def correctness(run, example) -> dict:
        is_correct = _judge(
            example.inputs["question"], example.outputs["reference_answer"], run.outputs["answer"]
        )
        return {"key": "correctness", "score": int(is_correct)}

    results = evaluate(
        target,
        data=dataset_name,
        evaluators=[correctness],
        experiment_prefix="correctness",
        description="LLM-judge (resident Qwen3.5) correctness over corpus/ golden Q&A",
    )

    scores = [row["evaluation_results"]["results"][0].score for row in results]
    n_correct = sum(scores)
    print(f"\nCorrectness: {n_correct}/{len(scores)} ({n_correct / len(scores):.0%})")
    if results.url:
        print(f"LangSmith: {results.url}")


if __name__ == "__main__":
    main()
