"""Счётчики вызовов, времени и токенов по провайдерам — узел 22 (cost/latency).

Инструментированы три шва, через которые проходит вся тяжёлая работа:
`llm.generate`, `embed.*`, `rerank.*`. Этого достаточно, чтобы ответить «куда
ушло время и во что обошёлся прогон», и при этом не пришлось трогать логику
агента (`loop.py`/`funnel.py`) — разбивка по ресурсу (LLM / эмбеддинги /
реранк / сеть) отвечает на вопрос точнее, чем разбивка по стадиям: стадия
«триаж» это и есть эмбеддинги, а «синтез» — это LLM.

Percentiles считаются по распределению ВЫЗОВОВ (их сотни за прогон), а не по
вопросам золотого набора (их десятки) — p95 по 18 точкам это просто максимум.

Счётчики глобальные и не потокобезопасные: eval-прогон однопоточный, а в
вебе один тяжёлый job за раз (§1 CLAUDE.md). Накладные расходы — инкремент и
`perf_counter`, поэтому включены всегда, без флага.
"""

from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Counter:
    calls: int = 0
    seconds: float = 0.0
    items: int = 0  # тексты для embed, пары для rerank
    prompt_tokens: int = 0
    completion_tokens: int = 0
    durations: list[float] = field(default_factory=list)


_counters: dict[str, Counter] = {}


def reset() -> None:
    _counters.clear()


def _counter(name: str) -> Counter:
    return _counters.setdefault(name, Counter())


@contextmanager
def track(name: str, items: int = 0):
    """Замеряет один вызов провайдера."""
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started
        c = _counter(name)
        c.calls += 1
        c.seconds += elapsed
        c.items += items
        c.durations.append(elapsed)


def add_tokens(name: str, prompt: int = 0, completion: int = 0) -> None:
    c = _counter(name)
    c.prompt_tokens += prompt
    c.completion_tokens += completion


def snapshot() -> dict[str, dict[str, Any]]:
    """Срез счётчиков. `p50`/`p95` — по вызовам этого провайдера."""
    out: dict[str, dict[str, Any]] = {}
    for name, c in sorted(_counters.items()):
        row: dict[str, Any] = {
            "calls": c.calls,
            "seconds": round(c.seconds, 3),
            "items": c.items,
        }
        if c.prompt_tokens or c.completion_tokens:
            row["prompt_tokens"] = c.prompt_tokens
            row["completion_tokens"] = c.completion_tokens
        if c.durations:
            ordered = sorted(c.durations)
            row["p50_seconds"] = round(statistics.median(ordered), 3)
            # ceil-индекс: на малых выборках percentile честнее брать так,
            # чем интерполяцией, которая создаёт значения, которых не было.
            idx = max(0, min(len(ordered) - 1, int(len(ordered) * 0.95 + 0.5) - 1))
            row["p95_seconds"] = round(ordered[idx], 3)
        out[name] = row
    return out
