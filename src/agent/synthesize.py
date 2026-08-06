"""Single-pass синтез: найденные чанки -> ответ с цитатами [n].

Один вызов LLM на переиспользуемом провайдере (§1: не инстанцировать модель
повторно) — через LCEL-цепочку `prompt | ChatMLX() | StrOutputParser()`.
`ChatMLX` (providers/langchain_llm.py) сам сворачивает system+human обратно в
пару для MLX chat-template, резидентная модель по-прежнему одна на процесс,
эта цепочка не заводит вторую.

`gaps` (Milestone 3, agent/loop.py) — подвопросы, не закрытые до исчерпания
бюджета: честно передаём их модели, чтобы ответ отражал реальные пробелы, а
не делал вид, что покрыто всё (§5: "цикл обязан завершаться по budget...
тогда честно сказать в ответе").
"""

from __future__ import annotations

from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .. import config
from ..providers.langchain_llm import ChatMLX

SYSTEM_PROMPT = (
    "Ты — исследовательский ассистент. Отвечай ТОЛЬКО на основе приведённого "
    "контекста. После каждого фактического утверждения ставь номер блока "
    "контекста в квадратных скобках, например [1]. Если ответа нет в "
    "контексте — прямо скажи об этом."
)

# Сам текст context/gaps/history подставляется значением переменной
# `user_message`, а не встраивается в тело шаблона — f-string-форматирование
# ChatPromptTemplate не переинтерпретирует фигурные скобки ВНУТРИ значения
# (в чанках из веба их сколько угодно: код, LaTeX, JSON), только в самом
# шаблоне.
_PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{user_message}")])
_CHAIN = _PROMPT | ChatMLX() | StrOutputParser()


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


def _format_history(history: list[tuple[str, str]]) -> str:
    turns = [f"Вопрос: {q}\nОтвет: {a}" for q, a in history]
    return "\n\n".join(turns)


def synthesize(
    question: str,
    chunks: list[dict[str, Any]],
    gaps: list[str] | None = None,
    history: list[tuple[str, str]] | None = None,
) -> str:
    context = _format_context(chunks)
    user_message = ""
    if history:
        # Follow-up-вопрос в том же диалоге (agent/research_runner.run_followup)
        # — модель должна видеть, что уже спрашивали и отвечали, чтобы
        # "а что насчёт X" разрешалось в контекст предыдущего ответа, а не
        # повисало без антецедента.
        user_message += f"Предыдущий диалог:\n{_format_history(history)}\n\n"
    user_message += f"Контекст:\n{context}\n\nВопрос: {question}"
    if gaps:
        gaps_text = "\n".join(f"- {g}" for g in gaps)
        user_message += (
            "\n\nВАЖНО: бюджет исследования исчерпан, следующие подвопросы "
            f"остались непокрытыми (найденных источников недостаточно):\n{gaps_text}\n"
            "Ответь на основе того, что есть в контексте, и явно укажи, какая "
            "часть вопроса осталась без достаточного подтверждения."
        )
    return _CHAIN.invoke({"user_message": user_message})
