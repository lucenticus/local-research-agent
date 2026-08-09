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

import os
from typing import Any

from src import config
from src.agent.synthesize import synthesize
from src.providers import embed, llm, rerank
from src.store.lancedb_store import LanceDBStore

from .eval_data import EVAL_CASES

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


def _answer(store: LanceDBStore, question: str) -> str:
    query_vector = embed.embed_texts([question])[0]
    hits = store.search_hybrid(question, query_vector, k=config.RERANK_CANDIDATES_K)
    if config.RERANK_ENABLED:
        hits = rerank.rerank(question, hits, top_n=config.TOP_K_RETRIEVE)
    return synthesize(question, hits)


def _ensure_dataset(client: Any) -> str:
    """Создаёт датасет, если его ещё нет, и докидывает в него любые кейсы из
    `eval_data.EVAL_CASES`, которых там ещё нет (сравнение по тексту
    вопроса) — так расширение golden-set (`eval_data.py`) само подхватывается
    без ручной синхронизации с уже существующим датасетом в LangSmith."""
    dataset_name = config.LANGSMITH_EVAL_DATASET
    if not client.has_dataset(dataset_name=dataset_name):
        client.create_dataset(
            dataset_name,
            description="local-research-agent: golden Q&A correctness eval over corpus/",
        )
        existing_questions: set[str] = set()
    else:
        existing_questions = {
            example.inputs["question"] for example in client.list_examples(dataset_name=dataset_name)
        }

    new_cases = [c for c in EVAL_CASES if c.question not in existing_questions]
    if new_cases:
        client.create_examples(
            dataset_name=dataset_name,
            examples=[
                {"inputs": {"question": c.question}, "outputs": {"reference_answer": c.reference_answer}}
                for c in new_cases
            ],
        )
    return dataset_name


def main() -> None:
    if not os.environ.get("LANGSMITH_API_KEY"):
        raise SystemExit(
            "LANGSMITH_API_KEY не задан в .env — нужен для загрузки датасета "
            "и результатов эксперимента в LangSmith (см. README 'Tracing: LangSmith')."
        )

    from langsmith import Client, evaluate

    client = Client()
    dataset_name = _ensure_dataset(client)
    store = LanceDBStore()

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
