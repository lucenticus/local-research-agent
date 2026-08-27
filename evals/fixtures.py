"""Заморозка ВСЕХ внешних входов research-прогона — не только discovery.

Зачем именно всех. Замер показал разброс 0.333 против 0.167 по покрытию
подтем **между двумя прогонами одной конфигурации**: при нулевой температуре
разброс даёт не модель, а внешний мир. Пока хоть один вход живой, повторный
прогон видит другие данные, и любая дельта тонет в этом разбросе.

Входов пять, и заморозить надо каждый:

1. `discover()` каждого источника — `sources/replay.py`;
2. **полный текст PDF** (`funnel.fetch_pdf_sections`) — deep read качает статьи;
3. **цитируемость** (`funnel.lookup_citation_counts`, Semantic Scholar батчем)
   — влияет на скор триажа, то есть на то, какие кандидаты вообще доживут до
   чтения;
4. **текущее время** (`funnel._now`) — буст по свежести считается от «сейчас»,
   так что тот же прогон завтра ранжирует иначе. Это не сеть, но вход;
5. **лимит по времени** (`Budget.max_seconds`) — медленный прогон
   останавливается раньше, то есть результат зависит от загрузки машины.
   Снимается отдельно, см. `eval_budget()`.

Патчинг идёт по именам, импортированным в `funnel`, а не по исходным
модулям: воронка делает `from ..sources.pdf import fetch_pdf_sections`, и
подменять надо именно её ссылку.

**Насколько это сработало — замерено, не предположено.** Покрытие подтем на
4 вопросах:

    без фикстур:  0.333 / 0.167 / 0.5    (три прогона, разброс 0.333)
    с фикстурами: 0.416 / 0.417          (два прогона, разброс 0.001)

Разброс агрегата упал примерно на два порядка, и этого достаточно, чтобы
различать эффекты, которые раньше тонули. Но **полного детерминизма нет**:
повопросно два прогона по одним фикстурам всё ещё расходятся на части
вопросов (2 из 4 стали побитово одинаковыми, 2 плавают). Все входы,
перечисленные выше, заморожены, температура нулевая, id чанков
детерминированы (`qdrant_store.chunk_id_for`) — остаток лежит ниже уровня,
который мы контролируем: скорее всего порядок редукции в MPS-ядрах даёт
микроразличия в эмбеддингах, а те меняют порядок близких по скору
кандидатов. Гнаться за этим (эмбеддинги на CPU) значит мерить не ту систему,
которая работает.

Практический вывод: агрегат по набору вопросов сравнивать можно, отдельный
вопрос — нет, его надо смотреть как сигнал «куда копать», а не как число.

Ограничение наследуется от `replay.py`: ключ — подвопрос, а не итоговый
запрос к API источника, поэтому изменения в ФОРМИРОВАНИИ запроса такими
фикстурами не проверяются (при replay вернётся старая запись). Для них —
живой прогон до и после.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone

from src.agent import funnel
from src.agent.state import Budget
from src.ingest.extract import Section
from src.sources import replay
from src.sources.replay import MODE_OFF, MODE_RECORD, MODE_REPLAY, DiscoveryCache

# Момент, относительно которого считается свежесть при replay. Записывается в
# фикстуры при record, чтобы прогон через месяц ранжировал так же, как исходный.
_NOW_KEY = "recorded_at"


def eval_budget() -> Budget:
    """Бюджет замера: без лимита по времени.

    Иначе результат зависит от того, чем ещё занята машина: медленный прогон
    упирается в `max_seconds` и останавливается на другом шаге. Шаги и
    deep-read по-прежнему ограничены — цикл не может не завершиться.
    """
    return Budget(max_seconds=None)


def _frozen_now(cache: DiscoveryCache) -> datetime:
    raw = cache.get_call("meta", _NOW_KEY)
    if raw is replay._MISS or not isinstance(raw, str):
        return datetime.now(timezone.utc)
    return datetime.fromisoformat(raw)


@contextmanager
def frozen_world(cache: DiscoveryCache):
    """Подменяет PDF, цитируемость и время на записанные в `cache`.

    В `off` не делает ничего. В `record` — зовёт настоящие и запоминает.
    Источники оборачиваются отдельно (`sources_for`), потому что они
    передаются в `run_research` аргументом, а не патчатся.
    """
    if cache.mode == MODE_OFF:
        yield
        return

    real_pdf = funnel.fetch_pdf_sections
    real_citations = funnel.lookup_citation_counts
    real_now = funnel._now

    if cache.mode == MODE_RECORD:
        cache.put_call("meta", _NOW_KEY, datetime.now(timezone.utc).isoformat())
    frozen = _frozen_now(cache)

    def pdf(url: str) -> list[Section]:
        cached = cache.get_call("pdf", url)
        if cached is not replay._MISS:
            return [Section(**s) for s in cached]
        sections = real_pdf(url)
        if cache.writable:
            cache.put_call("pdf", url, [asdict(s) for s in sections])
        return sections

    def citations(arxiv_ids: list[str]) -> dict[str, int]:
        # Ключ — отсортированный список id, а не сам вызов: один и тот же
        # набор статей может прийти в разном порядке (источники отвечают не
        # по расписанию), и это тот же самый вопрос к API.
        key = ",".join(sorted(arxiv_ids))
        cached = cache.get_call("citations", key)
        if cached is not replay._MISS:
            return cached
        # Отсутствие статьи в ответе — валидный записываемый факт («S2 её не
        # знает»), в отличие от недоступности API: та поднимает
        # SourceUnavailable и до записи не доходит вовсе.
        counts = real_citations(arxiv_ids)
        if cache.writable:
            cache.put_call("citations", key, counts)
        return counts

    funnel.fetch_pdf_sections = pdf
    funnel.lookup_citation_counts = citations
    funnel._now = lambda: frozen
    try:
        yield
    finally:
        funnel.fetch_pdf_sections = real_pdf
        funnel.lookup_citation_counts = real_citations
        funnel._now = real_now


def sources_for(cache: DiscoveryCache):
    """Источники для `run_research(sources=...)`, обёрнутые фикстурами."""
    from src.agent.research_runner import default_sources

    return replay.wrap(default_sources(), cache) if cache.mode != MODE_OFF else None
