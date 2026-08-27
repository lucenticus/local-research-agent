"""Узел 2: куда отправить запрос — `clarify` / `ask` / `research`.

Решение принимается НЕ вопросом к модели, а двумя проверками по фактам:

1. **clarify** — в запросе слишком мало содержательных слов. Детерминированно:
   после отбрасывания стоп-слов не остаётся ничего, что можно искать. Это
   ровно тот случай, где модель не отказывается, а сочиняет — на входе
   "question?" декомпозер выдал три подвопроса про глобальные цепочки
   поставок (поймано тестом, см. `planner.py`).
2. **ask vs research** — пробный retrieval по локальному индексу. Если корпус
   уже отвечает на вопрос уверенно (реранкер выше того же порога
   `FUNNEL_MIN_RERANK_SCORE`, что и везде), незачем идти в интернет на
   минуты. Это не догадка о сложности вопроса, а измерение: у нас уже есть
   ровно тот инструмент, который умеет отвечать «релевантно ли это».

Классификатора на LLM здесь нет сознательно. Он добавил бы третью модель
(или лишний вызов резидентной) ради решения, которое дешевле проверить
фактами; а главное — ошибка классификатора асимметрична: неверный `ask`
отдаёт плохой ответ молча, тогда как неверный `research` всего лишь
медленнее. Поэтому по умолчанию — `research`, а `ask` требует доказательства.

Где это реально нужно: MCP-сервер и веб, где режим никто не выбирает руками.
В CLI человек, набравший `research`, решение уже принял — там роутер не
навязывается (см. `cli.py`).
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import config
from ..providers import embed, rerank
from ..store.qdrant_store import QdrantStore

ROUTE_CLARIFY = "clarify"
ROUTE_ASK = "ask"
ROUTE_RESEARCH = "research"

# Служебные слова, которые сами по себе не делают запрос искомым. Список
# намеренно узкий: цель — отсечь "?", "help", "что это", а не угадывать тему.
_FILLER = frozenset("""
a an the and or of for in on to with from by as at is are was were be been am
what which who whom whose when where why how does do did doing done can could
should would will shall may might must have has had it its this that these those
there here i we you they he she them us our your their my me
что такое чем чему как какой какая какие когда где почему зачем кто кого это этот эта эти
и или для на по из от до над под при так тоже ещё уже вот ну да нет
question вопрос help помощь hello привет
""".split())


@dataclass
class Route:
    route: str
    reason: str
    evidence_score: float | None = None  # скор реранкера для ask-решения


def content_words(query: str) -> list[str]:
    """Слова, по которым вообще можно искать."""
    words = (w.strip("?!.,:;()[]{}\"'«»").lower() for w in query.split())
    return [w for w in words if w and w not in _FILLER]


def _local_evidence(query: str, store: QdrantStore) -> float | None:
    """Лучший скор реранкера по локальному индексу. `None` — индекса нет.

    Пустой индекс, пустая выдача и недоступный Qdrant — все три означают
    «доказательств нет», то есть `research`; поднимать это как ошибку не надо,
    отсутствие локального корпуса штатно.
    """
    try:
        query_vector = embed.embed_texts([query])[0]
        hits = store.search_hybrid(query, query_vector, k=config.ROUTER_PROBE_K)
    except Exception:
        return None
    if not hits:
        return None
    scored = rerank.score(query, hits)
    return max((score for _, score in scored), default=None)


def route(query: str, store: QdrantStore | None = None) -> Route:
    """Куда отправить запрос. По умолчанию — `research`."""
    words = content_words(query)
    if len(words) < config.ROUTER_MIN_CONTENT_WORDS:
        return Route(
            ROUTE_CLARIFY,
            f"в запросе {len(words)} содержательных слов — искать не по чему",
        )

    if store is None or not config.ROUTER_PROBE_LOCAL_INDEX:
        return Route(ROUTE_RESEARCH, "локальный индекс не проверялся")

    best = _local_evidence(query, store)
    if best is None:
        return Route(ROUTE_RESEARCH, "локальный индекс пуст или недоступен")
    if best >= config.ROUTER_ASK_MIN_SCORE:
        return Route(ROUTE_ASK, "локальный корпус уже отвечает на вопрос", evidence_score=best)
    return Route(ROUTE_RESEARCH, "локальный корпус не покрывает вопрос", evidence_score=best)
