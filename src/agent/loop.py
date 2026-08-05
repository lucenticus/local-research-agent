"""Итеративный контроллер: по открытым подвопросам retrieve -> gap-оценка ->
доуточнить/discover ещё -> до покрытия или бюджета (§3, §6 Milestone 3).

Gap-оценка — эвристика (§7: старт с порога покрытия, не LLM): подвопрос
закрыт, когда retrieval из LanceDB отдаёт чанки минимум от
`FUNNEL_MIN_SOURCES_TO_COVER` разных источников СО СКОРОМ РЕРАНКЕРА выше
`FUNNEL_MIN_RERANK_SCORE` (план §6: "порог score + покрытие подвопросов" —
обе части, не только счёт источников). Найдено реальным прогоном 2026-08-05:
без порога score тематически смежный, но нерелевантный локальный корпус
(например, про квантование LLM) ложно "закрывал" вопрос про KV-cache —
просто потому что в top-k гибридного поиска нашлось >=2 разных файлов, без
проверки, отвечают ли они на вопрос вообще. Bounded LLM yes/no gap-check —
опционально по плану, не реализован в Milestone 3 (реранкер уже даёт
калиброванный score, отдельный LLM-вызов избыточен).

Retrieve сначала смотрит в уже существующий индекс (в т.ч. прочитанное
прошлыми запросами) — это и есть кэш-хит: если предыдущий запрос уже привёл
нужную статью в LanceDB, discovery/deep-read на этот раз не понадобится.

Milestone 4: когда все подвопросы покрыты, черновой синтез прогоняется через
`agent/evaluate.py` — если faithfulness ниже `EVAL_FAITHFULNESS_THRESHOLD`,
подвопросы переоткрываются на ОДИН дополнительный проход (в пределах budget),
чтобы воронка собрала больше подтверждающих источников перед финалом. Больше
одного такого переоткрытия не делаем — иначе при системно слабом покрытии
можно застрять в переоткрытии до полного исчерпания budget без пользы.
"""

from __future__ import annotations

from .. import config
from ..providers import embed, rerank
from ..sources.base import Source
from ..store.lancedb_store import LanceDBStore
from . import evaluate, funnel, planner
from . import synthesize as synthesize_module
from .progress import ProgressCallback, emit as _emit
from .state import Budget, ResearchState, SubQuestion, SubQuestionStatus


def _distinct_sources(hits: list[dict]) -> set[str]:
    return {hit["source_id"] for hit in hits if hit.get("source_id")}


def _is_covered(store: LanceDBStore, sub_question: SubQuestion) -> bool:
    query_vector = embed.embed_texts([sub_question.text])[0]
    try:
        hits = store.search_hybrid(sub_question.text, query_vector, k=config.TOP_K_RETRIEVE)
    except RuntimeError:
        return False  # индекс ещё пуст — точно не закрыт
    if not hits:
        return False
    scored = rerank.score(sub_question.text, hits)
    relevant_hits = [
        hit for hit, score in scored if score >= config.FUNNEL_MIN_RERANK_SCORE
    ]
    return len(_distinct_sources(relevant_hits)) >= config.FUNNEL_MIN_SOURCES_TO_COVER


def _draft_is_faithful(question: str, store: LanceDBStore) -> bool:
    """Черновой синтез по текущему индексу + faithfulness-проверка.

    Пустой/отсутствующий индекс или пустая выдача — не считаем "нечестным",
    просто нечего проверять (возвращаем True, чтобы не зациклиться на
    заведомо пустом retrieval — этим уже занимается gap-оценка выше).
    """
    query_vector = embed.embed_texts([question])[0]
    try:
        hits = store.search_hybrid(question, query_vector, k=config.TOP_K_RETRIEVE)
    except RuntimeError:
        return True
    if not hits:
        return True
    draft = synthesize_module.synthesize(question, hits)
    result = evaluate.evaluate(draft, hits)
    return result.faithfulness >= config.EVAL_FAITHFULNESS_THRESHOLD


def run(
    question: str,
    sources: list[Source],
    store: LanceDBStore,
    budget: Budget | None = None,
    on_progress: ProgressCallback | None = None,
) -> ResearchState:
    state = ResearchState(question=question, budget=budget or Budget())
    state.sub_questions = planner.plan(question)
    low_faithfulness_retry_used = False

    _emit(on_progress, f"Подвопросов: {len(state.sub_questions)}.")

    while not state.budget_exhausted():
        open_sqs = state.open_sub_questions()
        if not open_sqs:
            _emit(on_progress, "Все подвопросы покрыты — проверяем обоснованность черновика…")
            if low_faithfulness_retry_used or not _draft_is_faithful(question, store):
                if low_faithfulness_retry_used:
                    break
                _emit(on_progress, "Обоснованность низкая — собираем больше источников.")
                low_faithfulness_retry_used = True
                for sq in state.sub_questions:
                    sq.status = SubQuestionStatus.OPEN
                continue
            break
        # Инкремент — ПОСЛЕ прохода (см. ниже), не здесь: если считать проход
        # начатым уже тут, budget_exhausted() внутри for-loop триггерится тем
        # же инкрементом и последний разрешённый проход всегда пропадает
        # вхолостую (было реальным багом, поймано тестом на моках).
        current_pass = state.iterations + 1
        _emit(on_progress, f"Итерация {current_pass}…")
        # С каждым проходом просим у источников больше кандидатов — иначе
        # повторный discover() с тем же запросом просто повторит уже
        # известную выдачу и не найдёт ничего нового (§ критерий "агент
        # находит статьи, которых не было предзагружено").
        discovery_limit = config.FUNNEL_DISCOVERY_LIMIT_PER_SOURCE * current_pass

        for sq in open_sqs:
            if state.budget_exhausted():
                break
            if not _is_covered(store, sq):
                funnel.run(sq, sources, state, store, discovery_limit=discovery_limit, on_progress=on_progress)
            if _is_covered(store, sq):
                state.cover(sq.text)

        state.iterations += 1

    # Итоговые пробелы — детерминированно из финального статуса подвопросов,
    # а не накоплены по ходу (иначе позже закрытый подвопрос остался бы в
    # gaps как стухший артефакт).
    state.gaps = [sq.text for sq in state.open_sub_questions()]
    _emit(on_progress, "Синтезируем ответ…")
    return state
