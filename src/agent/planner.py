"""Вопрос -> подвопросы.

Деление по явным разделителям — детерминированный код (§7 CLAUDE.md).
Сверх него — bounded LLM-разбор вопроса на аспекты, с жёсткой проверкой
результата кодом и откатом на исходный вопрос при любом сомнении.

Почему добавлен LLM-шаг. Замер на золотом наборе (`evals/run_quality.py`)
показал **покрытие подтем 0.42**: многогранный вопрос вроде "What approaches
exist for KV-cache compression?" становился ОДНИМ подвопросом, воронка
находила одну грань (например, квантизацию), gap-оценка считала подвопрос
закрытым — и цикл останавливался, не тронув остальные. Детерминированной
эвристикой это не чинится: грани («квантизация», «вытеснение токенов»,
«шеринг между слоями») — предметное знание, а не синтаксис; в вопросе нет ни
одного разделителя, по которому его можно разрезать.

Это не отход от «управление живёт в коде»: у модели спрашивают одну короткую
вещь, ответ проверяется кодом (`_valid_facets`), и решение о том, что делать
дальше, принимает код. Тот же приём и та же граница, что у bounded-перевода
подвопроса в `funnel.py::_discovery_query`. Плоский список аспектов, не DAG:
зависимости между подвопросами здесь не нужны, а параллелить их всё равно
нечем — резидентная модель одна.
"""

from __future__ import annotations

import re

from .. import config
from ..providers import llm
from .state import SubQuestion

_SPLIT_RE = re.compile(r"\s*(?:\?|;|(?:,\s*)?\bа также\b)\s*", re.IGNORECASE)
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")

_DECOMPOSE_SYSTEM_PROMPT = (
    "You split a research question into its distinct aspects. "
    "Output 2-4 short sub-questions, one per line, no numbering, no preamble. "
    "Each sub-question must cover a DIFFERENT aspect of the original and be "
    "searchable on its own. If the question is already narrow and has only one "
    "aspect, output it unchanged as a single line."
)


def _deterministic_split(question: str) -> list[str]:
    return [p.strip() for p in _SPLIT_RE.split(question) if p.strip()]


def _valid_facets(raw: str) -> list[str]:
    """Ответ модели -> аспекты, или пусто, если доверять нечему.

    Проверки нарочно строгие: лишний подвопрос стоит целого прохода воронки
    (discovery по всем источникам + триаж), поэтому дешевле откатиться на
    исходный вопрос, чем гнать цикл по мусорной строке.
    """
    facets: list[str] = []
    seen: set[str] = set()
    for line in raw.splitlines():
        line = _BULLET_RE.sub("", line).strip().strip('"«»')
        if not line:
            continue
        words = line.split()
        # Слишком короткая строка не ищется, слишком длинная — это модель
        # ушла в рассуждения вместо списка.
        if not (config.PLANNER_MIN_FACET_WORDS <= len(words) <= config.PLANNER_MAX_FACET_WORDS):
            return []
        key = line.lower().rstrip("?.")
        if key in seen:
            continue
        seen.add(key)
        facets.append(line)

    # Одна строка — модель сочла вопрос неделимым (или просто вернула его);
    # разбивать нечего, работаем с оригиналом.
    if len(facets) < 2:
        return []
    return facets[: config.PLANNER_MAX_SUBQUESTIONS]


def _llm_facets(question: str) -> list[str]:
    """Bounded-вызов: одна короткая генерация, любой сбой -> пусто."""
    try:
        prompt = llm.build_chat_prompt(_DECOMPOSE_SYSTEM_PROMPT, question)
        raw = llm.generate(prompt, max_tokens=config.PLANNER_MAX_TOKENS)
    except Exception:
        # Планирование не должно ронять запрос: без декомпозиции агент
        # работает как раньше, просто уже.
        return []
    return _valid_facets(raw)


def plan(question: str) -> list[SubQuestion]:
    """Вопрос -> подвопросы. Всегда возвращает хотя бы один для непустого входа."""
    question = question.strip()
    if not question:
        return []

    parts = _deterministic_split(question)
    if len(parts) > 1:
        # Составной вопрос разрезан явными разделителями — предметное знание
        # не нужно, LLM тут ничего не добавит.
        return [SubQuestion(text=p if p.endswith("?") else f"{p}?") for p in parts]

    # Короткий вопрос делить нечего, а модель на таком входе не отказывается,
    # а ВЫДУМЫВАЕТ: на "question?" она выдала три подвопроса про глобальные
    # цепочки поставок (поймано тестом цикла). Порог отсекает это дёшево и
    # заодно экономит вызов там, где он бесполезен.
    if config.PLANNER_LLM_DECOMPOSE and len(question.split()) >= config.PLANNER_MIN_QUESTION_WORDS:
        facets = _llm_facets(question)
        if facets:
            return [SubQuestion(text=f) for f in facets]

    return [SubQuestion(text=question)]
