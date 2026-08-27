"""Постоянный кэш цитируемости на диске — узел 21.

Зачем. Воронка спрашивает цитируемость у каждого arXiv-кандидата на каждом
прогоне, а находятся снова и снова одни и те же популярные статьи: на замере
из 4 вопросов это 30 статей, и следующий прогон спрашивает почти те же
самые. Внешние лимиты при этом реальны и уже били по нам — OpenAlex перешёл
на платный тираж, Semantic Scholar троттлит безключевой доступ (см.
`citations.py` и `semantic_scholar.py`). Данные, которые у нас уже есть,
не должны стоить ещё одного запроса.

Где он стоит — важнее, чем как устроен. Кэш живёт ВНУТРИ
`semantic_scholar.lookup_citation_counts`, то есть **ниже** того шва, который
подменяют фикстуры (`evals/fixtures.py` патчит `funnel.lookup_citation_counts`).
Поэтому замер кэша не видит вовсе: replay отдаёт записанное, не доходя сюда.
Наоборот было бы нельзя — кэш вычитал бы часть id из батча, ключ фикстуры
(отсортированный список id) перестал бы совпадать с записанным, и прогон
пошёл бы в промахи.

Протухание обязательно: цитируемость растёт, а бессрочный кэш заморозил бы
свежую статью на нуле навсегда — ровно ту, ради которой в триаже есть
отдельный буст по свежести.

Кэшируются только НАЙДЕННЫЕ значения. Отсутствие статьи в ответе S2 — факт
временный (препринт проиндексируют через неделю), а стоит его перепроверка
недорого: неизвестные id едут в том же самом батче, что и остальные.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from .. import config

# Файл читается/пишется целиком: запись — id, число и метка времени, около
# 60 байт; даже десятки тысяч статей остаются файлом на пару мегабайт.
# Заводить ради этого sqlite значит платить зависимостью за задачу, которой
# нет.
_lock = threading.Lock()


def _path() -> Path:
    # В МОМЕНТ ВЫЗОВА, а не на импорте: значение по умолчанию, вычисленное
    # при загрузке модуля, молча игнорирует любую подмену конфига — тестами
    # или иначе (этот класс ошибок уже ловили в providers/llm.py).
    return Path(config.CITATION_CACHE_PATH)


def _load() -> dict[str, list]:
    path = _path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # Битый или недоступный кэш — не повод падать: это ускоритель, а не
        # источник истины. Худшее последствие — лишний запрос к API.
        return {}
    return data if isinstance(data, dict) else {}


def _fresh(entry: object, now: float) -> bool:
    if not isinstance(entry, list) or len(entry) != 2:
        return False
    count, stored_at = entry
    if not isinstance(count, int) or not isinstance(stored_at, (int, float)):
        return False
    return now - stored_at < config.CITATION_CACHE_TTL_DAYS * 86400


def get_many(arxiv_ids: list[str]) -> tuple[dict[str, int], list[str]]:
    """`(что нашлось в кэше, что надо спросить)`.

    Протухшие записи попадают во второй список: спросить заново дешевле, чем
    ранжировать по числу годичной давности.
    """
    if not config.CITATION_CACHE_ENABLED:
        return {}, list(arxiv_ids)

    with _lock:
        entries = _load()
    now = time.time()
    hits: dict[str, int] = {}
    misses: list[str] = []
    for arxiv_id in arxiv_ids:
        entry = entries.get(arxiv_id)
        if _fresh(entry, now):
            hits[arxiv_id] = entry[0]
        else:
            misses.append(arxiv_id)
    return hits, misses


def put_many(counts: dict[str, int]) -> None:
    """Сохранить свежеполученные значения. Ошибка записи не роняет прогон."""
    if not config.CITATION_CACHE_ENABLED or not counts:
        return

    now = time.time()
    with _lock:
        # Перечитываем под локом: между чтением и записью мог отработать
        # другой поток (веб гоняет джобы в потоках), и его значения терять
        # незачем.
        entries = _load()
        entries.update({arxiv_id: [count, now] for arxiv_id, count in counts.items()})
        path = _path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(entries, sort_keys=True), encoding="utf-8")
        except OSError:
            pass
