"""Вопрос -> подвопросы. Детерминированный код, не LLM (§7 CLAUDE.md: 4B плохо
держит многошаговое мета-рассуждение — планирование живёт в коде).

Эвристика минимальна: составные вопросы (несколько "?" или явные
разделители "и"/"а также"/";") режутся на части; иначе весь вопрос — один
подвопрос. Более умная декомпозиция — за рамками Milestone 3 (ядро цикла
важнее качества планирования).
"""

from __future__ import annotations

import re

from .state import SubQuestion

_SPLIT_RE = re.compile(r"\s*(?:\?|;|(?:,\s*)?\bа также\b)\s*", re.IGNORECASE)


def plan(question: str) -> list[SubQuestion]:
    question = question.strip()
    if not question:
        return []

    parts = [p.strip() for p in _SPLIT_RE.split(question) if p.strip()]
    if len(parts) <= 1:
        return [SubQuestion(text=question)]

    return [SubQuestion(text=p if p.endswith("?") else f"{p}?") for p in parts]
