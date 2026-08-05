"""Самопроверка ответа: citation coverage + faithfulness (Milestone 4).

- **Citation coverage** — доля предложений-утверждений в ответе, у которых
  есть хотя бы одна цитата `[n]`.
- **Faithfulness** — доля процитированных утверждений, которые реально
  подтверждаются текстом источника, на который они ссылаются. Проверяем не
  отдельной NLI-моделью, а уже готовым реранкером: `rerank.score_pairs`
  с парой (само утверждение, текст процитированного чанка) даёт калиброванный
  P(yes) — тот же порог `FUNNEL_MIN_RERANK_SCORE`, что и в gap-оценке
  agent/loop.py (не заводим отдельный произвольный порог для eval).

Эвристика деления на предложения простая (`.!?` + пробел) — не NLP-грейд,
но достаточно для оценки на уровне milestone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .. import config
from ..providers import rerank

_CITATION_RE = re.compile(r"\[(\d+)\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_SOURCES_HEADING_RE = re.compile(r"\n\s*Источники\s*:?", re.IGNORECASE)


@dataclass
class ClaimCheck:
    sentence: str
    citations: list[int]
    faithful: bool | None  # None — без цитаты, вне скоупа faithfulness


@dataclass
class EvaluationResult:
    citation_coverage: float
    faithfulness: float
    checks: list[ClaimCheck] = field(default_factory=list)

    @property
    def unsupported(self) -> list[ClaimCheck]:
        """Процитированные утверждения, которые источник НЕ подтвердил."""
        return [c for c in self.checks if c.faithful is False]


def _strip_sources_block(answer: str) -> str:
    """Модель нередко сама дописывает свой блок "Источники:" в конце ответа —
    это не утверждения, отрезаем перед разбором на предложения."""
    match = _SOURCES_HEADING_RE.search(answer)
    return answer[: match.start()] if match else answer


def _split_sentences(answer: str) -> list[str]:
    body = _strip_sources_block(answer)
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(body) if s.strip()]


def evaluate(answer: str, chunks: list[dict[str, Any]]) -> EvaluationResult:
    """`chunks` — те же контекстные чанки (в том же порядке), что были
    переданы в `synthesize()` — номер цитаты `[n]` соответствует `chunks[n-1]`.
    """
    sentences = _split_sentences(answer)
    checks: list[ClaimCheck] = []
    pairs_to_score: list[tuple[str, str]] = []
    pair_targets: list[int] = []  # индекс в checks, соответствующий каждой паре

    for sentence in sentences:
        citation_numbers = [int(n) for n in _CITATION_RE.findall(sentence)]
        check = ClaimCheck(sentence=sentence, citations=citation_numbers, faithful=None)
        checks.append(check)
        if not citation_numbers:
            continue
        source_text = next(
            (chunks[n - 1]["text"] for n in citation_numbers if 1 <= n <= len(chunks)),
            None,
        )
        if source_text is not None:
            pairs_to_score.append((sentence, source_text))
            pair_targets.append(len(checks) - 1)

    if pairs_to_score:
        scores = rerank.score_pairs(pairs_to_score)
        for idx, score in zip(pair_targets, scores, strict=True):
            checks[idx].faithful = score >= config.FUNNEL_MIN_RERANK_SCORE

    cited = [c for c in checks if c.citations]
    coverage = len(cited) / len(checks) if checks else 0.0
    checked = [c for c in cited if c.faithful is not None]
    faithfulness = sum(1 for c in checked if c.faithful) / len(checked) if checked else 0.0

    return EvaluationResult(citation_coverage=coverage, faithfulness=faithfulness, checks=checks)
