"""ResearchState — единый объект, который читают и пишут планировщик, воронка,
цикл и синтезатор (§5 DEVELOPMENT_PLAN.md).

Инварианты (обязаны соблюдаться вызывающим кодом, а не только структурой
данных):
- один `id` не читается дважды — `mark_read` идемпотентен, а вызывающий код
  обязан проверять `is_read(id)` перед дорогим deep-read;
- `candidates`/`findings` только растут в пределах запроса — нет методов
  удаления;
- цикл обязан завершаться по `budget`, даже если остались пробелы —
  `budget_exhausted()` даёт на это однозначный ответ.

Follow-up-вопросы (уточнения в рамках того же диалога) переиспользуют один и
тот же `ResearchState` вместо создания нового — `candidates`/`findings`/
`read_ids` из предыдущих ходов остаются доступны (реальный кэш-хит: то, что
уже нашли и прочитали, не ищется и не читается заново), а `history` даёт
синтезу увидеть предыдущие вопросы и ответы диалога. `agent/loop.py::run()`
при передаче существующего `state` не пересоздаёт его, а только добавляет
новые подвопросы и выдаёт им свежий `budget` — см. docstring `loop.run`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from .. import config


class SubQuestionStatus(str, Enum):
    OPEN = "open"
    COVERED = "covered"


@dataclass
class SubQuestion:
    text: str
    status: SubQuestionStatus = SubQuestionStatus.OPEN


@dataclass
class Candidate:
    """Найденный, но необязательно ещё прочитанный источник (после discovery)."""

    id: str
    source: str
    title: str
    abstract: str
    meta: dict = field(default_factory=dict)
    triage_score: float | None = None


@dataclass
class Finding:
    """Извлечённый чанк, привязанный к подвопросу, который он закрывает."""

    text: str
    source_id: str
    sub_question: str


@dataclass
class Budget:
    max_iterations: int = config.DEFAULT_BUDGET_MAX_ITERATIONS
    max_deep_reads: int = config.DEFAULT_BUDGET_MAX_DEEP_READS
    max_seconds: float | None = config.DEFAULT_BUDGET_MAX_SECONDS


@dataclass
class ResearchState:
    question: str
    budget: Budget = field(default_factory=Budget)
    sub_questions: list[SubQuestion] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    read_ids: set[str] = field(default_factory=set)
    findings: list[Finding] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    history: list[tuple[str, str]] = field(default_factory=list)
    iterations: int = 0
    _candidate_ids: set[str] = field(default_factory=set, repr=False)
    _started_at: float = field(default_factory=time.monotonic, repr=False)

    def add_candidates(self, new_candidates: list[Candidate]) -> list[Candidate]:
        """Добавляет только неизвестные ранее id, возвращает реально добавленные."""
        added = []
        for candidate in new_candidates:
            if candidate.id in self._candidate_ids:
                continue
            self._candidate_ids.add(candidate.id)
            self.candidates.append(candidate)
            added.append(candidate)
        return added

    def is_read(self, candidate_id: str) -> bool:
        return candidate_id in self.read_ids

    def mark_read(self, candidate_id: str) -> None:
        self.read_ids.add(candidate_id)

    def add_findings(self, new_findings: list[Finding]) -> None:
        self.findings.extend(new_findings)

    def add_turn(self, question: str, answer: str) -> None:
        self.history.append((question, answer))

    def start_new_turn(self, budget: Budget | None = None) -> None:
        """Follow-up-ход того же диалога получает свой budget/таймер
        (`iterations`/`_started_at` сбрасываются) — `candidates`/`findings`/
        `read_ids` из прошлых ходов не трогаются, см. docstring класса."""
        self.budget = budget or Budget()
        self.iterations = 0
        self._started_at = time.monotonic()

    def add_gap(self, text: str) -> None:
        if text not in self.gaps:
            self.gaps.append(text)

    def open_sub_questions(self) -> list[SubQuestion]:
        return [sq for sq in self.sub_questions if sq.status == SubQuestionStatus.OPEN]

    def cover(self, sub_question_text: str) -> None:
        for sq in self.sub_questions:
            if sq.text == sub_question_text:
                sq.status = SubQuestionStatus.COVERED

    def budget_exhausted(self) -> bool:
        if self.iterations >= self.budget.max_iterations:
            return True
        if len(self.read_ids) >= self.budget.max_deep_reads:
            return True
        if self.budget.max_seconds is not None:
            elapsed = time.monotonic() - self._started_at
            if elapsed >= self.budget.max_seconds:
                return True
        return False
