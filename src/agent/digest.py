"""Digest-режим: "что нового вышло" в заданных arXiv-категориях за последние
N дней — browse, не Q&A (§ пользовательский запрос: агент должен хорошо
следить за свежими статьями в области ИИ).

Сознательно отдельно от agent/funnel.py, а не поверх него: воронка всегда
организована вокруг конкретного подвопроса (`discover(query, limit)` +
триаж по релевантности этому подвопросу) — для дайджеста нет вопроса,
нужен весь свежий поток одной-нескольких категорий, отсортированный по
дате. Прогонять это через funnel/loop означало бы придумывать фиктивный
"вопрос" и потом резать по релевантности то, что и так уже отобрано по
свежести — бессмысленно и медленнее (funnel эмбеддит и реранкает каждый
подвопрос отдельно, тут это не нужно).

Опциональное bounded LLM-резюме тем (`config.DIGEST_SUMMARIZE`) — свободный
обзорный абзац по аннотациям, а не цитируемый ответ с проверкой
faithfulness/coverage, как у `research()`/`ask()`: явно помечен в выводе
как обзор, не факт с источником.

Опциональный "глубокий" анализ (`deep=True`, § пользовательский запрос) —
на каждую статью: русское саммари (bounded LLM по заголовку+аннотации, без
досочинения — см. `_ITEM_SUMMARY_SYSTEM_PROMPT`) + реальные метаданные из
OpenAlex (`sources/citations.py::lookup_paper_details`): цитируемость,
venue, институции и h-index авторов. Совсем свежие препринты у OpenAlex
почти никогда ещё не проиндексированы (задержка индексации) — в этом
случае честно `analysis.details is None`, а не выдумка через name-search
автора (см. docstring `citations.py` про тёзок с разным h-index). Не
включено по умолчанию: N дополнительных LLM-вызовов + N OpenAlex-lookup'ов
на дайджест из `limit` статей — заметно медленнее обычного browse-режима,
поэтому ограничено `config.DIGEST_DEEP_MAX_ITEMS` независимо от `limit`.

Сверх этого глубокий анализ читает сам PDF статьи (`_analyze_pdf`,
`config.DIGEST_PDF_ANALYSIS`): полный текст режется на секции
(`ingest/extract.py`), в контекст под жёстким лимитом
(`config.DIGEST_PDF_CONTEXT_CHARS`) набираются в первую очередь
результаты/обсуждение/выводы, и bounded-LLM сводит их в разбор «основные
результаты / сравнение с аналогами / ограничения». Ссылки на код и модели
(github, huggingface и т.п.) при этом берутся НЕ у LLM, а детерминированно
из link-аннотаций и текста PDF — выдуманный, но правдоподобный адрес
репозитория здесь недопустим ровно так же, как выдуманный h-index; для
github-ссылок дополнительно подтягивается число звёзд (`sources/github.py`).

Оттуда же, из PDF, берутся авторы с аффилиациями (`_authors_from_header`) —
данные, которых нет ни в arXiv API (только имена), ни в OpenAlex для свежих
препринтов (ещё не проиндексированы). Шапка первой страницы извлекается
детерминированно, а LLM работает по ней только как парсер (номера-сноски
при извлечении текста схлопываются в "Ivanov1", регулярками надёжно не
разобрать) — это не то же самое, что спрашивать у модели, где работает
автор.

Опциональный `query` — раньше AND'ился прямо в arXiv search_query
(`ArxivSource.recent(..., query=...)`), но составной булев запрос из
keyword'ов + всех категорий в OR на некоторых формулировках упирался в
таймаут на стороне arXiv (§ пользовательский запрос: "по некоторым запросам
находят слишком много статей и по таймауту отваливается"). Теперь
`ArxivSource.recent()` про `query` не знает вообще: тянем весь пул статей
за те же `days` только по категориям (дешёвый, предсказуемый по времени
запрос), а релевантность считаем локально, в два прохода
(`_rank_by_relevance`), как в `research_runner.retrieve()`: сначала
гибридный dense+sparse поиск по пулу (`QdrantStore.search_hybrid`) до
`config.DIGEST_QUERY_HYBRID_K` кандидатов, потом реранкер поверх них до
`config.DIGEST_QUERY_TOP_K` (топ-5). Реранкать весь пул напрямую нельзя:
реранкер последовательный, по forward-проходу на статью (~0.15с на реальном
абстракте — на тысячах статей это десятки минут), тогда как эмбеддинг того
же пула батчами заметно дешевле. Итог — не "что нового по теме X вообще", а
"топ-N самого релевантного X за последние дни".

Сам пул тоже кэшируется, не только его эмбеддинги (`_collect_pool`): окно за
неделю — это 2000+ статей и ~80с выгрузки, и выкачивать их из arXiv заново
на каждый запрос незачем. Пул поднимается из Qdrant, а из arXiv добираются
только статьи, появившиеся с прошлого прогона.
"""

from __future__ import annotations

import urllib.parse
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from .. import config
from ..ingest.extract import Section
from ..providers import embed, llm, rerank
from ..sources.arxiv import ArxivSource
from ..sources.base import DiscoveredItem
from ..sources.citations import PaperDetails, lookup_paper_details
from ..sources.github import lookup_stars
from ..sources.pdf import fetch_pdf
from ..store.qdrant_store import Chunk, QdrantStore
from .progress import ProgressCallback, emit as _emit


@dataclass
class CodeLink:
    url: str
    kind: str  # github / gitlab / bitbucket / huggingface / colab
    stars: int | None = None  # только github; None — не смогли узнать, не "ноль звёзд"


@dataclass
class PaperAuthor:
    """Автор с аффилиацией, вычитанной из шапки PDF.

    Отдельно от `citations.AuthorDetails` (OpenAlex): у свежих препринтов
    OpenAlex-записи ещё нет, а в PDF аффилиации есть всегда — это как раз
    тот случай, когда данные реально доступны, просто не там, где искали
    раньше (arXiv API их не отдаёт вовсе).
    """

    name: str
    affiliation: str | None = None


@dataclass
class PaperInsights:
    """Разбор полного текста статьи (PDF), а не только аннотации."""

    findings_ru: str  # основные результаты и сравнения с аналогами
    code_links: list[CodeLink] = field(default_factory=list)
    sections_used: list[str] = field(default_factory=list)  # какие секции реально разобрались
    authors: list[PaperAuthor] = field(default_factory=list)


@dataclass
class ItemAnalysis:
    summary_ru: str
    details: PaperDetails | None  # None — статья ещё не проиндексирована в OpenAlex
    insights: PaperInsights | None = None  # None — PDF не скачался/не разобрался


@dataclass
class DigestResult:
    items: list[DiscoveredItem]
    days: int
    categories: list[str]
    query: str | None = None
    summary: str | None = None
    analyses: dict[str, ItemAnalysis] = field(default_factory=dict)  # ключ — item.id


_SUMMARY_SYSTEM_PROMPT = (
    "Ты — обозреватель научных статей в области ИИ. Тебе дан список "
    "заголовков и аннотаций свежих статей. Напиши краткий (3-6 предложений) "
    "обзор основных тем и трендов, которые видны в этой подборке — не "
    "пересказывай каждую статью по отдельности, укажи на общие направления, "
    "повторяющиеся идеи и заметные результаты. Отвечай на русском языке."
)

_ITEM_SUMMARY_SYSTEM_PROMPT = (
    "Ты — обозреватель научных статей в области ИИ. Тебе дан заголовок и "
    "аннотация одной статьи на английском. Напиши краткое саммари на "
    "русском языке (3-5 предложений): какую проблему решает статья, какой "
    "подход предлагает и какие основные результаты. Пиши только на основе "
    "данного текста, не добавляй фактов, которых там нет."
)


def _summarize(items: list[DiscoveredItem]) -> str:
    listing = "\n\n".join(f"{item.title}\n{item.abstract[:400]}" for item in items)
    prompt = llm.build_chat_prompt(_SUMMARY_SYSTEM_PROMPT, listing)
    return llm.generate(prompt, max_tokens=400).strip()


def _summarize_item(item: DiscoveredItem) -> str:
    user_message = f"{item.title}\n\n{item.abstract}"
    prompt = llm.build_chat_prompt(_ITEM_SUMMARY_SYSTEM_PROMPT, user_message)
    return llm.generate(prompt, max_tokens=250).strip()


# Хосты, ссылки на которые для читателя статьи означают "здесь код/модель".
# Ссылки на arxiv.org/doi.org сознательно не считаются: их в тексте десятки
# (это цитаты), и как "исходники статьи" они бесполезны.
_CODE_LINK_HOSTS = {
    "github.com": "github",
    "gitlab.com": "gitlab",
    "bitbucket.org": "bitbucket",
    "huggingface.co": "huggingface",
    "colab.research.google.com": "colab",
}

# Хосты, где ссылка вида .../pull/123 — это указание на конкретное чужое
# изменение, а не "здесь код статьи".
_REPO_HOSTS = {"github.com", "gitlab.com", "bitbucket.org"}
# Поймано на реальной статье про бенчмарк: в её PDF два десятка ссылок на
# PR'ы и коммиты в чужих известных репозиториях — это её датасет, и
# показывать их как "исходники статьи" (да ещё со звёздами тех репозиториев)
# просто неверно.
_CHANGE_REF_SEGMENTS = {"pull", "pulls", "commit", "commits", "issues", "compare", "releases"}

# Порядок, в котором секции набираются в контекст LLM: сначала то, где
# лежат результаты и сравнения с аналогами (ровно то, что просили), потом
# всё остальное — до исчерпания бюджета контекста.
_INSIGHT_SECTION_PRIORITY = ("results", "discussion", "conclusion", "method", "abstract", "introduction")

_INSIGHTS_SYSTEM_PROMPT = (
    "Ты — обозреватель научных статей в области ИИ. Тебе даны фрагменты "
    "полного текста одной статьи (эксперименты, обсуждение, выводы). "
    "Напиши на русском языке разбор из трёх частей:\n"
    "1) Основные результаты — с конкретными числами и метриками, если они есть в тексте.\n"
    "2) Сравнение с аналогами — с какими методами/моделями сравниваются и в чём "
    "выигрывают или проигрывают.\n"
    "3) Ограничения — если авторы их указывают.\n"
    "Пиши только то, что есть в данном тексте. Если чего-то в тексте нет, так и "
    "напиши, что этого в тексте нет, и не придумывай."
)


# Промпт с примерами, а не только с инструкцией: 4B-модель на голой
# инструкции стабильно теряла общую для всех авторов аффилиацию, если та
# стояла отдельной строкой ниже списка имён (проверено на реальной статье —
# выдавала все прочерки при явном "Tel Aviv University" в шапке). Примеры
# закрывают оба реальных формата шапок: общая организация и номера-сноски.
_AUTHORS_SYSTEM_PROMPT = (
    "Ты извлекаешь авторов и их организации из шапки научной статьи. "
    "Выведи по одной строке на автора строго в формате «Имя — Организация». "
    "Ничего, кроме этого списка, не пиши.\n\n"
    "Пример 1.\n"
    "Шапка:\n"
    "Deep Nets Are Great\n"
    "Jane Doe∗, John Roe & Ann Poe\n"
    "Tel Aviv University\n"
    "{jane,john}@mail.tau.ac.il\n"
    "Ответ:\n"
    "Jane Doe — Tel Aviv University\n"
    "John Roe — Tel Aviv University\n"
    "Ann Poe — Tel Aviv University\n\n"
    "Пример 2.\n"
    "Шапка:\n"
    "A Benchmark\n"
    "Xin Zhou1, Kisub Kim2, David Lo1\n"
    "1Singapore Management University\n"
    "2DGIST, Republic of Korea\n"
    "Ответ:\n"
    "Xin Zhou — Singapore Management University\n"
    "Kisub Kim — DGIST, Republic of Korea\n"
    "David Lo — Singapore Management University\n\n"
    "Правила: организацию, указанную одну на всех, повторяй каждому автору; "
    "номера-сноски сопоставляй по номеру; имена сохраняй как в тексте, без "
    "звёздочек и цифр; если организации в шапке действительно нет, ставь "
    "после тире прочерк. Не придумывай организаций, которых нет в тексте."
)


def _parse_authors(raw: str) -> list[PaperAuthor]:
    """Ответ LLM ("Имя — Организация" построчно) -> структура.

    Разбор намеренно строгий: строки без разделителя игнорируются, а не
    трактуются как "автор без аффилиации" — так вводные фразы модели
    ("Вот список авторов:") не превращаются в несуществующих авторов.
    """
    authors: list[PaperAuthor] = []
    for line in raw.splitlines():
        line = line.strip().lstrip("-*•").strip()
        if not line:
            continue
        for dash in ("—", " – ", " - "):
            if dash in line:
                name, _, affiliation = line.partition(dash)
                name = name.strip()
                affiliation = affiliation.strip().strip("—-–").strip()
                if name:
                    authors.append(PaperAuthor(name=name, affiliation=affiliation or None))
                break
    return authors


def _authors_from_header(header: str) -> list[PaperAuthor]:
    """Авторы с аффилиациями из шапки PDF.

    Здесь LLM используется как парсер уже имеющегося текста, а не как
    источник знаний: аффилиации физически лежат в переданной шапке, задача
    — разложить их по авторам (номера-сноски при извлечении текста из PDF
    схлопываются в строку вида "Xin Zhou1", поэтому чисто регулярками
    надёжно не разобрать). Это принципиально не то же самое, что спросить у
    модели "где работает этот автор" — вот так делать нельзя, см. docstring
    `sources/citations.py` про тёзок.
    """
    if not header.strip():
        return []
    prompt = llm.build_chat_prompt(_AUTHORS_SYSTEM_PROMPT, header)
    return _parse_authors(llm.generate(prompt, max_tokens=400))


def _pdf_url(item: DiscoveredItem) -> str | None:
    """Ссылка на PDF статьи.

    `meta["pdf_url"]` есть только у статей, только что пришедших из
    `sources/arxiv.py`. Пул же теперь по большей части поднимается из
    Qdrant (`_item_from_payload`), где meta не хранится — поэтому для
    arXiv-статей ссылка при необходимости достраивается из их id, иначе
    глубокий анализ ломался бы ровно на закэшированных статьях, то есть
    почти всегда.
    """
    pdf_url = item.meta.get("pdf_url")
    if pdf_url:
        return pdf_url
    if item.id.startswith("arxiv:"):
        return f"https://arxiv.org/pdf/{item.id.removeprefix('arxiv:')}"
    return None


def _code_links(urls: list[str]) -> list[CodeLink]:
    """URL'ы из PDF -> только ссылки на код/модели, дедуплицированные.

    Чисто детерминированный отбор по хосту, без участия LLM: правдоподобный,
    но выдуманный адрес репозитория — ровно то, чего в этом проекте быть не
    должно (та же логика, что с h-index у тёзок в `sources/citations.py`).
    """
    links: list[CodeLink] = []
    seen: set[str] = set()
    for url in urls:
        parsed = urllib.parse.urlparse(url)
        host = parsed.netloc.lower().removeprefix("www.")
        kind = _CODE_LINK_HOSTS.get(host)
        segments = [s for s in parsed.path.split("/") if s]
        # Голый хост без пути ("https://github.com/") — не ссылка на код, а
        # чаще всего разорванный переносом URL; поймано на реальной статье.
        if not kind or not segments:
            continue
        if host in _REPO_HOSTS:
            if len(segments) > 2 and segments[2].lower() in _CHANGE_REF_SEGMENTS:
                continue  # ссылка на чужой PR/коммит — это цитата, не код статьи
            # Нормализуем к корню репозитория: .../tree/main/src и сам репозиторий
            # — одно и то же место, показывать оба незачем.
            url = f"https://{host}/{segments[0]}/{segments[1]}" if len(segments) >= 2 else url
        if url not in seen:
            seen.add(url)
            links.append(CodeLink(url=url, kind=kind))

    # Выкидываем ссылки, являющиеся префиксом другой: в реальном PDF
    # рядом с "…/acme/DistMoE" попадался огрызок "…/acme/" — тот же URL,
    # разорванный вёрсткой. Более длинная ссылка всегда информативнее.
    # Префикс считается по границе пути, а не по символам: иначе ссылка на
    # ".../acme/repo" выкидывалась бы как "префикс" ".../acme/repo2", хотя
    # это разные репозитории.
    return [
        link
        for link in links
        if not any(
            other.url != link.url and other.url.startswith(link.url.rstrip("/") + "/")
            for other in links
        )
    ]


def _insight_context(sections: list[Section]) -> tuple[str, list[str]]:
    """Секции -> текст под жёстким лимитом контекста + какие секции вошли.

    Лимит обязателен (§1 CLAUDE.md): полный текст статьи — это десятки тысяч
    символов, а раздутый контекст надувает KV-кэш, это главный риск OOM на
    16ГБ.
    """
    budget = config.DIGEST_PDF_CONTEXT_CHARS
    parts: list[str] = []
    used: list[str] = []
    by_category: dict[str, list[Section]] = {}
    for section in sections:
        by_category.setdefault(section.category, []).append(section)

    for category in _INSIGHT_SECTION_PRIORITY:
        for section in by_category.get(category, []):
            if budget <= 0:
                break
            text = section.text.strip()
            if not text:
                continue
            parts.append(f"## {section.name}\n{text[:budget]}")
            used.append(section.name)
            budget -= len(text[:budget])
    return "\n\n".join(parts), used


def _analyze_pdf(item: DiscoveredItem) -> PaperInsights | None:
    """Разбор полного текста статьи. `None` — PDF недоступен или не разобрался.

    Best effort, как и `lookup_paper_details`: у части статей PDF закрыт,
    свёрстан так, что эвристика заголовков не находит секций, или просто не
    отдаётся. Тогда честнее показать "не разобрали", чем выдать разбор
    неизвестно чего.
    """
    pdf_url = _pdf_url(item)
    if not pdf_url:
        return None
    try:
        fetched = fetch_pdf(pdf_url)
    except Exception:
        return None

    context, used = _insight_context(fetched.sections)
    if not context:
        return None

    prompt = llm.build_chat_prompt(_INSIGHTS_SYSTEM_PROMPT, f"{item.title}\n\n{context}")
    findings = llm.generate(prompt, max_tokens=config.DIGEST_PDF_MAX_TOKENS).strip()
    return PaperInsights(
        findings_ru=findings,
        code_links=_with_stars(_code_links(fetched.links)),
        sections_used=used,
        authors=_authors_from_header(fetched.header),
    )


def _with_stars(links: list[CodeLink]) -> list[CodeLink]:
    """Проставляет звёзды GitHub-ссылкам. Остальные хосты не трогаются."""
    for link in links:
        if link.kind == "github":
            link.stars = lookup_stars(link.url)
    return links


def analyze_item(item: DiscoveredItem, on_progress: ProgressCallback | None = None) -> ItemAnalysis:
    """Глубокий анализ одной статьи — русское саммари + OpenAlex-метаданные.

    Вынесено отдельной публичной функцией (а не только внутренним шагом
    `_analyze_items`), т.к. веб-UI теперь запускает анализ статьи по клику
    на конкретную карточку, а не на весь дайджест разом — см.
    `web/app.py::_run_item_analysis_job`."""
    _emit(on_progress, "Writing the summary…")
    summary_ru = _summarize_item(item)
    _emit(on_progress, "Looking up OpenAlex metadata (citations, venue, authors)…")
    details = lookup_paper_details(item.title)

    insights = None
    if config.DIGEST_PDF_ANALYSIS:
        _emit(on_progress, "Fetching and parsing the PDF (results, comparisons, code links)…")
        insights = _analyze_pdf(item)
        if insights is None:
            _emit(on_progress, "Could not parse the PDF — abstract and metadata only.")

    _emit(on_progress, "Done.")
    return ItemAnalysis(summary_ru=summary_ru, details=details, insights=insights)


def _analyze_items(
    items: list[DiscoveredItem], on_progress: ProgressCallback | None
) -> dict[str, ItemAnalysis]:
    analyses: dict[str, ItemAnalysis] = {}
    for i, item in enumerate(items, start=1):
        _emit(on_progress, f"Analysing paper {i}/{len(items)}: {item.title[:70]}…")
        analyses[item.id] = analyze_item(item)
    return analyses


# Во сколько раз шире DIGEST_QUERY_HYBRID_K забирать из поиска до отсева по
# текущему пулу — коллекция накопительная, часть верхних хитов может быть из
# прошлых прогонов и не попасть в окно «за последние N дней».
_POOL_OVERFETCH = 5


def _pool_chunks(items: list[DiscoveredItem]) -> list[Chunk]:
    texts = [f"{item.title}\n{item.abstract}" for item in items]
    dense, sparse = embed.embed_texts_hybrid(texts)
    return [
        Chunk(
            id=str(uuid.uuid4()),
            text=text,
            source_id=item.id,
            source_title=item.title,
            section="abstract",
            vector=vector,
            sparse=sparse_vector,
            url=item.url or "",
            published_date=item.published_date or "",
            authors=item.meta.get("authors") or [],
        )
        for item, text, vector, sparse_vector in zip(items, texts, dense, sparse, strict=True)
    ]


def _item_from_payload(payload: dict) -> DiscoveredItem:
    """Обратное к `_pool_chunks`: статья пула, восстановленная из индекса.

    `text` собирался как "заголовок\\nаннотация", а заголовок arXiv-парсер
    схлопывает в одну строку (`_parse` джойнит пробелами) — значит первый
    перевод строки и есть граница, и аннотация восстанавливается точно.
    """
    text = payload.get("text") or ""
    _, _, abstract = text.partition("\n")
    return DiscoveredItem(
        id=payload["source_id"],
        source="arxiv",
        title=payload.get("source_title") or "",
        abstract=abstract,
        url=payload.get("url") or "",
        published_date=payload.get("published_date") or None,
        meta={"authors": payload.get("authors") or []},
    )


def _load_cached_pool(
    store: QdrantStore, days: int
) -> tuple[list[DiscoveredItem], bool]:
    """Пул из индекса + покрывает ли кэш всё окно.

    Второе значение — можно ли останавливать выгрузку из arXiv досрочно:
    только если в индексе есть статья старше начала окна, нижняя граница
    окна точно уже закэширована (см. `ArxivSource.recent(known_ids=...)`).
    Иначе (первый прогон, либо пользователь расширил `--days`) кэша на дно
    окна не хватает и окно надо выкачать целиком.
    """
    cutoff_ts = (datetime.now(timezone.utc) - timedelta(days=days)).timestamp()
    cached = [_item_from_payload(p) for p in store.load_pool(cutoff_ts)]
    oldest = store.oldest_published_ts()
    return cached, oldest is not None and oldest < cutoff_ts


def _collect_pool(
    store: QdrantStore,
    categories: list[str],
    days: int,
    on_progress: ProgressCallback | None,
) -> list[DiscoveredItem]:
    """Все статьи окна: закэшированные из индекса + доехавшие с прошлого раза.

    Раньше окно выкачивалось из arXiv целиком на каждый запрос (~80с на
    2000+ статей за неделю), даже когда все они уже лежали в индексе —
    кэшировались только эмбеддинги, но не сама выдача. Теперь пул сначала
    поднимается из индекса (`load_pool`), а из arXiv добираются только
    статьи, появившиеся с прошлого прогона: выдача идёт от свежих к старым,
    поэтому первая полностью известная страница означает "догнали кэш"
    (см. `ArxivSource.recent(known_ids=...)`).

    Досрочная остановка разрешена только когда кэш заведомо покрывает дно
    окна, иначе (первый прогон, либо расширили `--days`) окно выкачивается
    целиком.
    """
    cached, covers_window = _load_cached_pool(store, days)
    if cached:
        _emit(on_progress, f"Restored {len(cached)} papers of the window from the index.")

    known_ids = {item.id for item in cached} if (cached and covers_window) else None
    if known_ids:
        _emit(on_progress, "Fetching only new papers from arXiv…")
    fetched = ArxivSource(categories=categories).recent(
        days=days,
        limit=None,
        on_progress=lambda n: _emit(on_progress, f"New papers fetched: {n}…"),
        known_ids=known_ids,
    )
    _emit(on_progress, f"New papers from arXiv: {len(fetched)}.")

    # Кэш мог собираться по другому набору категорий — дедуп по id и
    # сортировка по дате (свежие первыми), как и отдаёт arXiv.
    by_id = {item.id: item for item in cached}
    by_id.update({item.id: item for item in fetched})
    return sorted(by_id.values(), key=lambda i: i.published_date or "", reverse=True)


def _rank_by_relevance(
    store: QdrantStore,
    query: str,
    items: list[DiscoveredItem],
    top_k: int,
    on_progress: ProgressCallback | None,
) -> list[DiscoveredItem]:
    """Пул статей -> top_k самых релевантных `query`, в два прохода.

    Реранкер идёт последовательно, по forward-проходу на статью, поэтому
    прогонять через него весь пул (тысячи статей за неделю) — минуты. Вместо
    этого пул сначала эмбеддится батчами и сужается тем же гибридным
    dense+sparse поиском, что и `research`/`ask`
    (`QdrantStore.search_hybrid`), до `config.DIGEST_QUERY_HYBRID_K`, и только
    эти кандидаты идут в реранкер — та же двухступенчатая схема
    (retrieval -> rerank), что в `research_runner.retrieve()`.

    Эмбеддится только то, чего в коллекции ещё нет (`has_source`), — сам пул
    к этому моменту уже собран `_collect_pool`, которая тоже переиспользует
    эту коллекцию как кэш.
    """
    fresh = [item for item in items if not store.has_source(item.id)]
    if fresh:
        _emit(on_progress, f"Indexing {len(fresh)} new papers (of {len(items)})…")
        store.add_chunks(_pool_chunks(fresh))

    _emit(on_progress, f"Hybrid search for: {query}…")
    query_vector = embed.embed_texts([query])[0]
    by_source_id = {item.id: item for item in items}
    # Берём с запасом и отсекаем всё, чего нет в текущем пуле: коллекция
    # накопительная и держит статьи прошлых прогонов (другие дни/категории),
    # а дайджест обязан остаться в своём окне «за последние N дней».
    hits = store.search_hybrid(
        query, query_vector, k=config.DIGEST_QUERY_HYBRID_K * _POOL_OVERFETCH
    )
    candidates = [
        {"text": hit["text"], "item": by_source_id[hit["source_id"]]}
        for hit in hits
        if hit["source_id"] in by_source_id
    ][: config.DIGEST_QUERY_HYBRID_K]
    if not candidates:
        return []

    _emit(on_progress, f"Reranking the top {len(candidates)} candidates…")
    ranked = rerank.rerank(query, candidates, top_n=top_k)
    return [c["item"] for c in ranked]


def run_digest(
    days: int | None = None,
    categories: list[str] | None = None,
    limit: int | None = None,
    summarize: bool | None = None,
    query: str | None = None,
    deep: bool = False,
    on_progress: ProgressCallback | None = None,
) -> DigestResult:
    days = config.DIGEST_DEFAULT_DAYS if days is None else days
    categories = categories or config.ARXIV_AI_CATEGORIES
    limit = config.DIGEST_DEFAULT_LIMIT if limit is None else limit
    summarize = config.DIGEST_SUMMARIZE if summarize is None else summarize
    query = query.strip() if query and query.strip() else None

    scope = f"on '{query}' " if query else ""
    _emit(on_progress, f"Looking for papers {scope}from the last {days} days in {', '.join(categories)}…")

    if query:
        # Режим с темой: нужен весь пул окна, чтобы было из чего выбирать
        # релевантное — он же и кэшируется между запросами.
        store = QdrantStore(collection_name=config.QDRANT_DIGEST_COLLECTION)
        items = _collect_pool(store, categories, days, on_progress)
        _emit(on_progress, f"Papers in the window: {len(items)}.")
        if items:
            items = _rank_by_relevance(
                store, query, items, config.DIGEST_QUERY_TOP_K, on_progress
            )
            _emit(on_progress, f"Showing the top {len(items)} by relevance.")
    else:
        # Без темы дайджест — это просто «последние N статей»: тянуть и
        # индексировать всё окно незачем.
        items = ArxivSource(categories=categories).recent(days=days, limit=limit)
        _emit(on_progress, f"Found {len(items)}.")

    summary = None
    if summarize and items:
        _emit(on_progress, "Building the topic overview…")
        summary = _summarize(items)

    analyses: dict[str, ItemAnalysis] = {}
    if deep and items:
        analyses = _analyze_items(items[: config.DIGEST_DEEP_MAX_ITEMS], on_progress)

    return DigestResult(
        items=items, days=days, categories=categories, query=query, summary=summary, analyses=analyses
    )
