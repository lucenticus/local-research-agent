"""Регресс-прогон discovery: сколько кандидатов даёт каждый источник на золотом наборе.

Меряет самый первый шаг воронки — до триажа, deep-read и синтеза. Это
намеренно узкая метрика: если источник молча отдаёт ноль, всё, что дальше,
уже не спасти, а по итоговому ответу этого не видно (остальные источники
маскируют провал).

Идёт через настоящий путь (`planner.plan` → `funnel._discovery_query` →
`Source.discover`), а не мимо него: узел 5 (переформулировка) — ровно то,
что этот прогон должен уметь измерять.

    python -m evals.run_discovery --tag en                 # живой прогон
    python -m evals.run_discovery --record fixtures.json   # + записать фикстуры
    python -m evals.run_discovery --replay fixtures.json   # офлайн, из фикстур
    python -m evals.run_discovery --baseline evals/runs/<файл>.json

Фикстуры годятся для регресса «не сломалось ли остальное», но НЕ для оценки
изменений в формировании запроса к источнику — ключ кэша это подвопрос, а не
итоговый запрос к API (см. `sources/replay.py`). Такие изменения меряются
живым прогоном до и после.
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
from src.agent.funnel import _discovery_query
from src.agent.planner import plan
from src.agent.research_runner import default_sources
from src.sources.replay import MODE_OFF, MODE_RECORD, MODE_REPLAY, DiscoveryCache, wrap

_QUESTIONS = Path(__file__).parent / "questions.jsonl"
_RUNS_DIR = Path(__file__).parent / "runs"


def load_questions(tag: str | None = None) -> list[dict[str, Any]]:
    cases = [json.loads(line) for line in _QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [c for c in cases if tag is None or tag in c.get("tags", [])]


def run(cases: list[dict[str, Any]], cache: DiscoveryCache, delay: float) -> dict[str, Any]:
    sources = wrap(default_sources(), cache)
    limit = config.FUNNEL_DISCOVERY_LIMIT_PER_SOURCE
    rows: list[dict[str, Any]] = []

    for i, case in enumerate(cases, start=1):
        # Тот же путь, что в воронке: вопрос -> подвопросы -> поисковый запрос.
        sub_questions = plan(case["question"])
        queries = [_discovery_query(sq.text) for sq in sub_questions]
        per_source: dict[str, int] = {}
        per_errors: dict[str, list[str]] = {}

        for source in sources:
            found, errors = 0, []
            for query in queries:
                try:
                    found += len(source.discover(query, limit))
                except Exception as exc:
                    # «Источник упал» и «источник ничего не нашёл» — РАЗНЫЕ
                    # факты с разными причинами (наш запрос против чужого
                    # рейт-лимита), и смешивать их в один ноль нельзя.
                    errors.append(type(exc).__name__ + ": " + str(exc)[:60])
            per_source[source.name] = found
            if errors:
                per_errors[source.name] = errors

        rows.append({"id": case["id"], "question": case["question"], "queries": queries,
                     "per_source": per_source, "per_errors": per_errors})
        print(f"[{i}/{len(cases)}] {case['id']:<28} " +
              "  ".join(f"{name}={'ERR' if name in per_errors else n}"
                        for name, n in per_source.items()), flush=True)
        if delay and cache.mode != MODE_REPLAY:
            time.sleep(delay)

    return {"generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": cache.mode, "n_cases": len(cases),
            "aggregate": aggregate(rows), "rows": rows}


def aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    names = sorted({name for r in rows for name in r["per_source"]})
    out: dict[str, dict[str, float]] = {}
    for name in names:
        counts = [r["per_source"].get(name, 0) for r in rows]
        failed = sum(1 for r in rows if name in r.get("per_errors", {}))
        hits = sum(1 for c in counts if c > 0)
        out[name] = {
            "hit_rate": hits / len(counts) if counts else 0.0,  # доля вопросов с непустой выдачей
            "error_rate": failed / len(rows) if rows else 0.0,  # доля вопросов, где источник ОШИБСЯ
            "mean_items": statistics.mean(counts) if counts else 0.0,
            "total_items": sum(counts),
        }
    return out


def print_table(agg: dict[str, dict[str, float]], baseline: dict[str, dict[str, float]] | None) -> None:
    print(f"\n{'источник':<20} {'hit rate':>10} {'ошибок':>9} {'ср. кандидатов':>16} {'всего':>8}")
    print("-" * 68)
    for name, m in agg.items():
        line = (f"{name:<20} {m['hit_rate']:>9.0%} {m.get('error_rate', 0):>8.0%} "
                f"{m['mean_items']:>16.1f} {m['total_items']:>8}")
        if baseline and name in baseline:
            d_hit = m["hit_rate"] - baseline[name]["hit_rate"]
            d_mean = m["mean_items"] - baseline[name]["mean_items"]
            if abs(d_hit) > 1e-9 or abs(d_mean) > 1e-9:
                line += f"   Δ hit {d_hit:+.0%}, Δ ср. {d_mean:+.1f}"
        print(line)


def main() -> None:
    p = argparse.ArgumentParser(description="Регресс-прогон discovery на золотом наборе")
    p.add_argument("--tag", default=None, help="только вопросы с этим тегом (en / ru / recency / ...)")
    p.add_argument("--record", metavar="FILE", default=None, help="записать ответы источников в фикстуры")
    p.add_argument("--replay", metavar="FILE", default=None, help="офлайн-прогон по фикстурам")
    p.add_argument("--baseline", metavar="FILE", default=None, help="прошлый прогон для диффа")
    p.add_argument("--delay", type=float, default=3.0, help="пауза между вопросами, сек (arXiv троттлит)")
    p.add_argument("--no-save", action="store_true", help="не писать результат в evals/runs/")
    args = p.parse_args()

    if args.record and args.replay:
        p.error("--record и --replay взаимоисключающие")

    mode = MODE_RECORD if args.record else MODE_REPLAY if args.replay else MODE_OFF
    cache = DiscoveryCache(args.record or args.replay or "/dev/null", mode=mode)

    cases = load_questions(args.tag)
    print(f"Вопросов: {len(cases)}  ·  режим: {mode}")
    result = run(cases, cache, args.delay)

    baseline = None
    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))["aggregate"]
    print_table(result["aggregate"], baseline)

    if cache.misses:
        print(f"\n⚠ промахов кэша: {len(cache.misses)} — эти источники считались как пустые")
        for key in cache.misses[:5]:
            print(f"    {key}")
    if mode == MODE_RECORD:
        cache.save()
        print(f"\nФикстур записано: {len(cache)} → {cache.path}")

    if not args.no_save:
        _RUNS_DIR.mkdir(parents=True, exist_ok=True)
        out = _RUNS_DIR / f"discovery-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
        out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Прогон сохранён: {out}")


if __name__ == "__main__":
    main()
