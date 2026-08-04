"""Single-pass синтез: найденные чанки -> ответ с цитатами [n].

Milestone 0: без итеративного цикла и gap-оценки (см. Milestone 3) — один
вызов LLM на переиспользуемом провайдере (§1: не инстанцировать модель
повторно).
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


def synthesize(question: str, chunks: list[dict[str, Any]]) -> str:
    context = _format_context(chunks)
    user_message = f"Контекст:\n{context}\n\nВопрос: {question}"
    prompt = llm.build_chat_prompt(SYSTEM_PROMPT, user_message)
    return llm.generate(prompt)
