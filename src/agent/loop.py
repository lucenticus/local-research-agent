"""Итеративный контроллер: по открытым подвопросам retrieve -> gap-оценка ->
доуточнить/discover ещё -> до покрытия или бюджета.

Оркестрация переведена на LangGraph (эксперимент с фреймворком, см. запрос
пользователя) — сам контроллер собран как `StateGraph` с узлами `plan` /
`run_pass` / `check_faithfulness` / `finalize` и условными переходами между
ними (`_route_pass`, `_route_after_faithfulness`), 1:1 повторяющими прежний
`while`-цикл. Вся предметная логика (gap-оценка, faithfulness-проверка,
discovery/триаж) не изменилась и живёт в тех же местах — `_is_covered`,
`_draft_is_faithful` и `funnel.run` остаются обычными модульными функциями,
которые тесты подменяют через monkeypatch, граф лишь вызывает их по имени.

Gap-оценка — эвристика (§7: старт с порога покрытия, не LLM): подвопрос
закрыт, когда retrieval из LanceDB отдаёт чанки минимум от
`FUNNEL_MIN_SOURCES_TO_COVER` разных источников СО СКОРОМ РЕРАНКЕРА выше
`FUNNEL_MIN_RERANK_SCORE` ("порог score + покрытие подвопросов" —
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

Follow-up-вопросы: `run()` принимает опциональный уже существующий `state`
(см. docstring `state.py`) — тогда НЕ пересоздаёт его, а добавляет только
новые подвопросы follow-up-текста и выдаёт им свежий `budget`/таймер, оставляя
`candidates`/`findings`/`read_ids` прежних ходов как есть (кэш-хит). Чтобы
faithfulness-retry и подсчёт "все подвопросы покрыты" на этом проходе не
трогали ДАВНО решённые или ещё не решённые подвопросы прошлых ходов —
`_GraphState.active_texts` держит тексты только подвопросов ТЕКУЩЕГО вызова,
и весь граф (кроме `finalize`, который честно репортит gaps по ВСЕМ ходам
диалога сразу) фильтрует по этому множеству.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .. import config
from ..providers import embed, rerank
from ..sources.base import Source
from ..store.lancedb_store import LanceDBStore
from . import evaluate, funnel, planner
from . import synthesize as synthesize_module
from .progress import ProgressCallback, emit as _emit
from .state import Budget, ResearchState, SubQuestionStatus


def _distinct_sources(hits: list[dict]) -> set[str]:
    return {hit["source_id"] for hit in hits if hit.get("source_id")}


def _is_covered(store: LanceDBStore, sub_question) -> bool:
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


class _GraphState(TypedDict):
    research_state: ResearchState
    question: str
    sources: list[Source]
    store: LanceDBStore
    on_progress: ProgressCallback | None
    force_discovery: bool
    low_faithfulness_retry_used: bool
    active_texts: set[str]
    is_followup: bool


def _active_open(gs: _GraphState) -> list:
    return [sq for sq in gs["research_state"].open_sub_questions() if sq.text in gs["active_texts"]]


def _node_plan(gs: _GraphState) -> dict:
    """Вопрос -> подвопросы (`agent/planner.py`, детерминированный код, не
    LLM — см. его docstring) как явный узел графа. Follow-up (`is_followup`)
    добавляет подвопросы к уже накопленным в `research_state`, обычный вызов
    заменяет список с нуля."""
    rs = gs["research_state"]
    new_sub_questions = planner.plan(gs["question"])
    if gs["is_followup"]:
        rs.sub_questions.extend(new_sub_questions)
    else:
        rs.sub_questions = new_sub_questions
    active_texts = {sq.text for sq in new_sub_questions}
    _emit(gs["on_progress"], f"Подвопросов: {len(active_texts)}.")
    return {"active_texts": active_texts}


def _node_run_pass(gs: _GraphState) -> dict:
    rs = gs["research_state"]
    open_sqs = _active_open(gs)
    # Инкремент — ПОСЛЕ прохода (см. ниже), не до: если считать проход
    # начатым раньше, budget_exhausted() внутри for-loop триггерится тем же
    # инкрементом и последний разрешённый проход всегда пропадает вхолостую
    # (было реальным багом, поймано тестом на моках).
    current_pass = rs.iterations + 1
    _emit(gs["on_progress"], f"Итерация {current_pass}…")
    # С каждым проходом просим у источников больше кандидатов — иначе
    # повторный discover() с тем же запросом просто повторит уже известную
    # выдачу и не найдёт ничего нового.
    discovery_limit = config.FUNNEL_DISCOVERY_LIMIT_PER_SOURCE * current_pass
    force_discovery = gs["force_discovery"]

    for sq in open_sqs:
        if rs.budget_exhausted():
            break
        if force_discovery or not _is_covered(gs["store"], sq):
            funnel.run(
                sq, gs["sources"], rs, gs["store"],
                discovery_limit=discovery_limit, on_progress=gs["on_progress"],
            )
        if _is_covered(gs["store"], sq):
            rs.cover(sq.text)

    rs.iterations += 1
    return {"force_discovery": False}  # только на один проход сразу после retry


def _node_check_faithfulness(gs: _GraphState) -> dict:
    rs = gs["research_state"]
    _emit(gs["on_progress"], "Все подвопросы покрыты — проверяем обоснованность черновика…")
    if gs["low_faithfulness_retry_used"]:
        return {}  # retry уже использован — больше не переоткрываем
    if _draft_is_faithful(gs["question"], gs["store"]):
        return {}
    _emit(gs["on_progress"], "Обоснованность низкая — собираем больше источников.")
    for sq in rs.sub_questions:
        if sq.text in gs["active_texts"]:
            sq.status = SubQuestionStatus.OPEN
    return {"force_discovery": True, "low_faithfulness_retry_used": True}


def _node_finalize(gs: _GraphState) -> dict:
    rs = gs["research_state"]
    # Итоговые пробелы — детерминированно из финального статуса подвопросов,
    # а не накоплены по ходу (иначе позже закрытый подвопрос остался бы в
    # gaps как стухший артефакт).
    rs.gaps = [sq.text for sq in rs.open_sub_questions()]
    _emit(gs["on_progress"], "Синтезируем ответ…")
    return {}


def _route_pass(gs: _GraphState) -> str:
    if gs["research_state"].budget_exhausted():
        return "finalize"
    if _active_open(gs):
        return "run_pass"
    return "check_faithfulness"


def _route_after_faithfulness(gs: _GraphState) -> str:
    if not _active_open(gs):
        return "finalize"  # честный черновик, либо retry уже использован
    if gs["research_state"].budget_exhausted():
        return "finalize"
    return "run_pass"  # переоткрыто retry'ем — ещё один форсированный проход


def _build_graph():
    graph = StateGraph(_GraphState)
    graph.add_node("plan", _node_plan)
    graph.add_node("run_pass", _node_run_pass)
    graph.add_node("check_faithfulness", _node_check_faithfulness)
    graph.add_node("finalize", _node_finalize)

    graph.add_edge(START, "plan")
    graph.add_conditional_edges(
        "plan", _route_pass,
        {"run_pass": "run_pass", "check_faithfulness": "check_faithfulness", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "run_pass", _route_pass,
        {"run_pass": "run_pass", "check_faithfulness": "check_faithfulness", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "check_faithfulness", _route_after_faithfulness,
        {"run_pass": "run_pass", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)
    return graph.compile()


# Скомпилирован один раз на модуль — сам граф не хранит состояние запроса
# (оно целиком в `_GraphState`), пересобирать его на каждый `run()` не нужно.
_GRAPH = _build_graph()


def run(
    question: str,
    sources: list[Source],
    store: LanceDBStore,
    budget: Budget | None = None,
    on_progress: ProgressCallback | None = None,
    state: ResearchState | None = None,
) -> ResearchState:
    """`state=None` (обычный вызов) — свежий `ResearchState` с нуля.

    `state=<существующий>` — follow-up-ход того же диалога: подвопросы
    follow-up-текста добавляются к уже накопленным (не заменяют их, см.
    `_node_plan`), ход получает свой budget/таймер, `candidates`/`findings`/
    `read_ids` из прошлых ходов остаются доступны воронке как кэш."""
    is_followup = state is not None
    if state is None:
        state = ResearchState(question=question, budget=budget or Budget())
    else:
        state.start_new_turn(budget)

    initial: _GraphState = {
        "research_state": state,
        "question": question,
        "sources": sources,
        "store": store,
        "on_progress": on_progress,
        "force_discovery": False,
        "low_faithfulness_retry_used": False,
        "active_texts": set(),  # заполняется _node_plan первым делом
        "is_followup": is_followup,
    }
    # `recursion_limit` считает узлы графа, а не проходы воронки — потолок
    # даём с большим запасом относительно любого разумного budget.max_iterations,
    # реальную остановку делает budget_exhausted() внутри самого графа.
    _GRAPH.invoke(initial, config={"recursion_limit": 1000})
    return state
