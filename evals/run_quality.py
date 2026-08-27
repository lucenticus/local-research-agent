"""Регресс-прогон качества: coverage, faithfulness, покрытие подтем, cost, latency.

В отличие от `run_discovery.py` (первый шаг воронки) прогоняет ПОЛНЫЙ
`run_research()` на каждом вопросе — со всеми итерациями, deep-read и
синтезом. Поэтому он дорогой: минуты на вопрос, и это не оптимизируется, а
принимается: качество ответа нельзя измерить, не сгенерировав ответ.

Метрики:

* **citation coverage** и **faithfulness** — из `agent/evaluate.py` как есть,
  без второй реализации. Это те же скореры, которыми агент проверяет себя в
  цикле, — расхождения между «внутренней» и «внешней» оценкой быть не должно.
* **покрытие подтем** — доля `expected_subtopics` из золотого набора, которые
  ответ реально закрыл. Считается тем же реранкером и тем же порогом
  (`FUNNEL_MIN_RERANK_SCORE`), что и остальные решения о релевантности:
  заводить отдельную модель-судью, чтобы судить свою же выдачу, — лишняя
  сущность и лишний порог.
* **честность на неотвечаемых** — для вопросов с `expected_gaps: true`
  проверяется, что агент СООБЩИЛ пробел, а не выдал уверенный ответ.
* **cost / latency** — из `providers/metrics.py`, счётчики сбрасываются
  перед каждым вопросом, так что видно и на вопрос, и суммарно.

Прогон пишется после КАЖДОГО вопроса: час работы не должен теряться из-за
падения на десятом.

    python -m evals.run_quality --limit 3           # быстрая проверка
    python -m evals.run_quality --tag adversarial   # только честность
    python -m evals.run_quality --baseline evals/runs/quality-<...>.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src import config
from src.agent.evaluate import evaluate
from src.agent.research_runner import run_research
from src.providers import metrics, rerank
from src.store.qdrant_store import QdrantStore

from .run_discovery import load_questions

_RUNS_DIR = Path(__file__).parent / "runs"


def subtopic_coverage(answer: str, subtopics: list[str]) -> tuple[float, list[str]]:
    """Доля подтем, подтверждённых ответом + список непокрытых.

    Реранкер отвечает на вопрос «подтверждает ли этот текст это утверждение» —
    ровно то же, что он делает в faithfulness-проверке, только там текст
    источника против утверждения ответа, а тут ответ против ожидаемой подтемы.
    """
    if not subtopics:
        return 1.0, []  # нечего покрывать (неотвечаемые вопросы) — не штраф
    scores = rerank.score_pairs([(topic, answer) for topic in subtopics])
    missed = [t for t, s in zip(subtopics, scores, strict=True) if s < config.FUNNEL_MIN_RERANK_SCORE]
    return (len(subtopics) - len(missed)) / len(subtopics), missed


def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    metrics.reset()
    # Своя коллекция, вычищенная перед КАЖДЫМ вопросом. Иначе замер врёт:
    # research-коллекция накопительная, второй прогон переиспользует
    # скачанное первым и выглядит быстрее и лучше просто потому, что он
    # второй (поймано на сравнении baseline↔after: 78с -> 35с на том же
    # вопросе без единого изменения в retrieval).
    store = QdrantStore(collection_name=config.QDRANT_EVAL_COLLECTION)
    store.rebuild([])  # удаляет коллекцию целиком

    started = time.perf_counter()
    error = None
    try:
        result = run_research(case["question"], store)
    except Exception as exc:  # прогон не должен падать целиком из-за одного вопроса
        # Ключи метрик присутствуют со значением None: строка с ошибкой должна
        # быть той же формы, что и обычная, иначе на ней падает любой
        # потребитель — дифф, отчёт, внешний анализ (поймано на своём же
        # A/B-скрипте: KeyError вместо результата после часа работы).
        return {"id": case["id"], "question": case["question"],
                "error": f"{type(exc).__name__}: {exc}",
                "wall_seconds": round(time.perf_counter() - started, 1),
                "citation_coverage": None, "faithfulness": None, "unsupported": [],
                "subtopic_coverage": None, "missed_subtopics": [],
                "expects_gap": bool(case.get("expected_gaps")), "reported_gap": None,
                "gap_honest": None, "iterations": None, "read_count": None,
                "candidates_count": None,
                "providers": metrics.snapshot()}
    wall = time.perf_counter() - started

    # Ровно тот контекст, что видел synthesize(): нумерация [n] в ответе
    # соответствует context[n-1]. Подставлять сюда заголовки из `sources`
    # нельзя — faithfulness тогда проверял бы утверждение против заголовка.
    quality = evaluate(result.answer, result.context) if result.context else None
    covered, missed = subtopic_coverage(result.answer, case.get("expected_subtopics", []))

    expects_gap = bool(case.get("expected_gaps"))
    reported_gap = bool(result.gaps)
    return {
        "id": case["id"],
        "question": case["question"],
        "error": error,
        "wall_seconds": round(wall, 1),
        "citation_coverage": round(quality.citation_coverage, 3) if quality else None,
        "faithfulness": round(quality.faithfulness, 3) if quality else None,
        "unsupported": [c.sentence for c in quality.unsupported] if quality else [],
        "subtopic_coverage": round(covered, 3),
        "missed_subtopics": missed,
        "expects_gap": expects_gap,
        "reported_gap": reported_gap,
        "gap_honest": reported_gap if expects_gap else None,
        "iterations": result.iterations,
        "read_count": result.read_count,
        "candidates_count": result.candidates_count,
        "providers": metrics.snapshot(),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if not r.get("error")]

    def mean(key: str) -> float | None:
        vals = [r[key] for r in ok if r.get(key) is not None]
        return round(statistics.mean(vals), 3) if vals else None

    honest = [r["gap_honest"] for r in ok if r.get("gap_honest") is not None]
    walls = sorted(r["wall_seconds"] for r in ok)
    totals: dict[str, dict[str, float]] = {}
    for r in ok:
        for name, m in r.get("providers", {}).items():
            acc = totals.setdefault(name, {"calls": 0, "seconds": 0.0, "tokens": 0})
            acc["calls"] += m["calls"]
            acc["seconds"] += m["seconds"]
            acc["tokens"] += m.get("prompt_tokens", 0) + m.get("completion_tokens", 0)
    return {
        "n_ok": len(ok), "n_error": len(rows) - len(ok),
        "citation_coverage": mean("citation_coverage"),
        "faithfulness": mean("faithfulness"),
        "subtopic_coverage": mean("subtopic_coverage"),
        "gap_honesty": round(sum(honest) / len(honest), 3) if honest else None,
        # По вопросам — min/median/max: p95 на десятке точек это максимум,
        # и называть его перцентилем нечестно. Перцентили есть у провайдеров,
        # где вызовов сотни (см. providers/metrics.py).
        "wall_seconds": {"min": walls[0], "median": statistics.median(walls), "max": walls[-1]} if walls else None,
        "providers_total": totals,
    }


def print_report(agg: dict[str, Any], baseline: dict[str, Any] | None) -> None:
    def delta(key: str) -> str:
        if not baseline or baseline.get(key) is None or agg.get(key) is None:
            return ""
        d = agg[key] - baseline[key]
        return f"   Δ {d:+.3f}" if abs(d) > 1e-9 else ""

    print(f"\n{'метрика':<24} {'значение':>10}")
    print("-" * 50)
    for key, label in [("citation_coverage", "citation coverage"), ("faithfulness", "faithfulness"),
                       ("subtopic_coverage", "покрытие подтем"), ("gap_honesty", "честность на gap")]:
        val = agg.get(key)
        print(f"{label:<24} {('—' if val is None else f'{val:.3f}'):>10}{delta(key)}")

    if agg.get("wall_seconds"):
        w = agg["wall_seconds"]
        print(f"\nвремя на вопрос, с:  min {w['min']:.0f} · median {w['median']:.0f} · max {w['max']:.0f}")
    if agg.get("providers_total"):
        print(f"\n{'провайдер':<24} {'вызовов':>9} {'секунд':>9} {'токенов':>10}")
        print("-" * 56)
        for name, m in sorted(agg["providers_total"].items()):
            print(f"{name:<24} {m['calls']:>9} {m['seconds']:>9.1f} {m['tokens']:>10}")
    if agg.get("n_error"):
        print(f"\n⚠ вопросов с ошибкой: {agg['n_error']}")


def main() -> None:
    p = argparse.ArgumentParser(description="Регресс-прогон качества research()")
    p.add_argument("--tag", default=None, help="только вопросы с этим тегом")
    p.add_argument("--limit", type=int, default=None, help="взять первые N вопросов (прогон дорогой)")
    p.add_argument("--baseline", metavar="FILE", default=None, help="прошлый прогон для диффа")
    args = p.parse_args()

    cases = load_questions(args.tag)[: args.limit]
    print(f"Вопросов: {len(cases)}  ·  полный research() на каждом, это минуты")

    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = _RUNS_DIR / f"quality-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    rows: list[dict[str, Any]] = []

    for i, case in enumerate(cases, start=1):
        print(f"\n[{i}/{len(cases)}] {case['id']} — {case['question'][:60]}", flush=True)
        row = evaluate_case(case)
        rows.append(row)
        if row.get("error"):
            print(f"    ОШИБКА: {row['error']}", flush=True)
        else:
            print(f"    coverage={row['citation_coverage']} faithful={row['faithfulness']} "
                  f"подтемы={row['subtopic_coverage']} за {row['wall_seconds']}с", flush=True)
        # пишем после каждого вопроса: час работы не должен пропасть из-за
        # падения на середине
        out.write_text(json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(),
                                   "aggregate": aggregate(rows), "rows": rows},
                                  ensure_ascii=False, indent=1), encoding="utf-8")

    baseline = None
    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))["aggregate"]
    print_report(aggregate(rows), baseline)
    print(f"\nПрогон сохранён: {out}")


if __name__ == "__main__":
    main()
