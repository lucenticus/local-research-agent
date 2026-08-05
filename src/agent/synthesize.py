"""Single-pass синтез: найденные чанки -> ответ с цитатами [n].

Один вызов LLM на переиспользуемом провайдере (§1: не инстанцировать модель
повторно). `gaps` (Milestone 3, agent/loop.py) — подвопросы, не закрытые до
исчерпания бюджета: честно передаём их модели, чтобы ответ отражал реальные
пробелы, а не делал вид, что покрыто всё (§5: "цикл обязан завершаться по
budget... тогда честно сказать в ответе").
"""

from __future__ import annotations

from typing import Any

from .. import config
from ..providers import llm

SYSTEM_PROMPT = (
    "Ты — исследовательский ассистент. Отвечай ТОЛЬКО на основе приведённого "
    "контекста. После каждого фактического утверждения ставь номер блока "
    "контекста в квадратных скобках, например [1]. Если ответа нет в "
    "контексте — прямо скажи об этом."
)


def _format_context(chunks: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    total = 0
    for i, chunk in enumerate(chunks, start=1):
        title = chunk.get("source_title") or chunk.get("source_id") or "?"
        block = f"[{i}] (источник: {title})\n{chunk['text']}"
        if total + len(block) > config.MAX_SYNTHESIS_CONTEXT_CHARS:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)


def synthesize(question: str, chunks: list[dict[str, Any]], gaps: list[str] | None = None) -> str:
    context = _format_context(chunks)
    user_message = f"Контекст:\n{context}\n\nВопрос: {question}"
    if gaps:
        gaps_text = "\n".join(f"- {g}" for g in gaps)
        user_message += (
            "\n\nВАЖНО: бюджет исследования исчерпан, следующие подвопросы "
            f"остались непокрытыми (найденных источников недостаточно):\n{gaps_text}\n"
            "Ответь на основе того, что есть в контексте, и явно укажи, какая "
            "часть вопроса осталась без достаточного подтверждения."
        )
    prompt = llm.build_chat_prompt(SYSTEM_PROMPT, user_message)
    return llm.generate(prompt)
