"""Прогрессивная воронка: discovery -> триаж -> deep read (§3 DEVELOPMENT_PLAN.md).

Discovery дёшево (только метаданные, все источники сразу). Триаж скорит
abstract кандидатов против подвопроса эмбеддингом и оставляет top-N — большая
часть кандидатов умирает здесь, полный текст не качается. Deep read — только
для выживших и ещё не прочитанных (`state.read_ids`): для arXiv скачивается
PDF и извлекается секция-осознанно, для остальных источников (без открытого
полного текста) используется сам abstract как единственный chunk — честный
fallback, а не притворство, что full-text недоступен.

Найдено реальным прогоном 2026-08-05: arXiv/Semantic Scholar — англоязычные
корпуса, подвопрос на русском (а агент по умолчанию русскоязычный) даёт 0
результатов дословным запросом. Добавлен bounded LLM-перевод подвопроса в
короткий английский поисковый запрос перед discovery — это расширение
"опц. bounded LLM" из плана (там разрешён для gap-check) на реальную
необходимость, без которой источники не работают для целевого пользователя.

Триаж учитывает цитируемость (§ пользовательский запрос 2026-08-06):
Semantic Scholar отдаёт `citationCount` прямо при discovery, у arXiv своей
цитируемости нет — обогащаем через `sources/citations.py` (OpenAlex, best
effort). Итоговый score триажа = косинус (семантическая релевантность
подвопросу) + небольшой логарифмический буст по цитируемости — см.
`_combined_score` и `CITATION_BOOST_SCALE` в config.py: буст — тайбрейкер
между близкими по смыслу кандидатами, а не замена семантике (иначе
популярная, но нерелевантная статья обходила бы точный, но малоцитируемый
ответ).

Кросс-источниковый дедуп по arXiv id (найдено реальным прогоном 2026-08-06):
одна и та же статья находится и через `arxiv.py` (id вида `arxiv:XXXXvN`), и
через общий веб-поиск (id вида `web:<url>` — например, ссылка на HTML-версию
той же статьи на arxiv.org). Разные id -> `state.add_candidates` не видит,
что это один и тот же кандидат, и статья дублируется в списке источников.
`_canonical_candidate_id` нормализует любой arXiv-подобный id/URL к номеру
статьи без версии — все варианты (abs/html/pdf, разные версии, найдено через
любой источник) схлопываются в один кандидат.
"""

from __future__ import annotations

import math
import re
import tempfile
import urllib.request
import uuid
from pathlib import Path
from .. import config
from ..ingest.chunk import chunk_sections, chunk_text
from ..ingest.extract import Section, extract_pdf_sections
from ..providers import embed, llm
from ..sources.base import DiscoveredItem, Source
from ..sources.citations import lookup_citation_count
from ..store.lancedb_store import Chunk, LanceDBStore
from .progress import ProgressCallback, emit as _emit
from .state import Candidate, Finding, ResearchState, SubQuestion

_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(?:v\d+)?")


def _canonical_candidate_id(item: DiscoveredItem) -> str:
    """arXiv-статьи — единый id независимо от источника обнаружения и URL-варианта."""
    if item.source == "arxiv":
        haystack = item.id
    elif item.source == "semantic_scholar":
        haystack = (item.meta.get("external_ids") or {}).get("ArXiv", "") or (item.url or "")
    else:
        haystack = item.url or ""
    match = _ARXIV_ID_RE.search(haystack)
    return f"arxiv:{match.group(1)}" if match else item.id

_TRANSLATE_SYSTEM_PROMPT = (
    "Translate the user's question into a short English web-search query "
    "(3-8 keywords, no punctuation, no explanations, no quotes). "
    "Output ONLY the query, nothing else."
)


def _looks_non_english(text: str) -> bool:
    cyrillic = sum(1 for ch in text if "а" <= ch.lower() <= "я" or ch.lower() == "ё")
    return cyrillic > len(text) * 0.3


def _discovery_query(text: str) -> str:
    """Английский поисковый запрос для discovery — переводим, только если
    подвопрос явно не на английском (bounded LLM-вызов, см. docstring модуля)."""
    if not _looks_non_english(text):
        return text
    prompt = llm.build_chat_prompt(_TRANSLATE_SYSTEM_PROMPT, text)
    translated = llm.generate(prompt, max_tokens=32).strip()
    return translated or text


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _discover(
    sub_question: SubQuestion, sources: list[Source], discovery_limit: int
) -> list[Candidate]:
    query = _discovery_query(sub_question.text)
    candidates: list[Candidate] = []
    for source in sources:
        try:
            items: list[DiscoveredItem] = source.discover(query, limit=discovery_limit)
        except Exception:
            # Внешний источник недоступен/троттлит — воронка продолжает с тем,
            # что нашли остальные источники, а не падает целиком.
            continue
        for item in items:
            citation_count = item.citation_count
            if citation_count is None and item.source == "arxiv":
                # arXiv не отдаёт цитируемость сам — обогащаем через OpenAlex
                # (best effort: не найдено/недоступно -> остаётся None).
                citation_count = lookup_citation_count(item.title)
            candidates.append(
                Candidate(
                    id=_canonical_candidate_id(item),
                    source=item.source,
                    title=item.title,
                    abstract=item.abstract,
                    meta={**item.meta, "url": item.url, "year": item.year,
                          "citation_count": citation_count},
                )
            )
    return candidates


def _combined_score(cosine_similarity: float, citation_count: int | None) -> float:
    """Семантическая релевантность + небольшой буст по цитируемости.

    log1p, не сырое число — иначе статья с 10000 цитирований задавила бы
    любую семантику. `CITATION_BOOST_SCALE` откалиброван так, чтобы буст был
    тайбрейкером (доли от типичного разброса косинуса), а не доминирующим
    фактором — см. docstring модуля.
    """
    if not citation_count or citation_count <= 0:
        return cosine_similarity
    return cosine_similarity + config.CITATION_BOOST_SCALE * math.log1p(citation_count)


def _triage(sub_question: SubQuestion, candidates: list[Candidate]) -> list[Candidate]:
    scoreable = [c for c in candidates if c.abstract.strip()]
    if not scoreable:
        return []
    texts = [sub_question.text] + [c.abstract for c in scoreable]
    vectors = embed.embed_texts(texts)
    query_vec, candidate_vecs = vectors[0], vectors[1:]
    for candidate, vec in zip(scoreable, candidate_vecs, strict=True):
        cosine_similarity = _cosine(query_vec, vec)
        candidate.triage_score = _combined_score(
            cosine_similarity, candidate.meta.get("citation_count")
        )
    scoreable.sort(key=lambda c: c.triage_score or 0.0, reverse=True)
    return scoreable[: config.FUNNEL_TRIAGE_TOP_N]


def _fetch_pdf_sections(pdf_url: str) -> list[Section]:
    request = urllib.request.Request(pdf_url, headers={"User-Agent": "local-research-agent/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read(config.FUNNEL_MAX_PDF_BYTES)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(body)
        tmp_path = Path(tmp.name)
    try:
        return extract_pdf_sections(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _deep_read_sections(candidate: Candidate) -> list[Section]:
    pdf_url = candidate.meta.get("pdf_url")
    if pdf_url:
        try:
            sections = _fetch_pdf_sections(pdf_url)
            if sections:
                return sections
        except Exception:
            pass  # источник недоступен -> fallback на abstract ниже
    # Нет полного текста (Semantic Scholar/web без PDF, или скачивание не
    # удалось) — честно используем сам abstract как единственный chunk, а не
    # притворяемся, что deep read сделан на полном тексте.
    return [Section(name=candidate.title, category="abstract", text=candidate.abstract)]


def run(
    sub_question: SubQuestion,
    sources: list[Source],
    state: ResearchState,
    store: LanceDBStore,
    discovery_limit: int | None = None,
    on_progress: ProgressCallback | None = None,
) -> None:
    """Одна итерация воронки для одного подвопроса — расширяет state и LanceDB.

    `discovery_limit` растёт с каждой повторной попыткой (см. agent/loop.py) —
    так повторный проход реально достаёт статьи, которых не было в первой
    (более узкой) выдаче, а не просто повторяет тот же самый запрос.
    """
    _emit(on_progress, f"Ищем источники: «{sub_question.text}»…")
    discovered = _discover(sub_question, sources, discovery_limit or config.FUNNEL_DISCOVERY_LIMIT_PER_SOURCE)
    new_candidates = state.add_candidates(discovered)
    survivors = _triage(sub_question, new_candidates)
    _emit(
        on_progress,
        f"Найдено {len(discovered)} кандидатов, {len(survivors)} прошли триаж.",
    )

    for candidate in survivors:
        if state.budget_exhausted():
            break
        if state.is_read(candidate.id):
            continue

        already_indexed = store.has_source(candidate.id)
        if not already_indexed:
            _emit(on_progress, f"Читаем: {candidate.title[:80]}…")
            sections = _deep_read_sections(candidate)
            raw_chunks = chunk_sections(sections) or chunk_text(candidate.abstract)
            if raw_chunks:
                vectors = embed.embed_texts([c.text for c in raw_chunks])
                citation_count = candidate.meta.get("citation_count")
                chunks = [
                    Chunk(
                        id=str(uuid.uuid4()),
                        text=raw.text,
                        source_id=candidate.id,
                        source_title=candidate.title,
                        section=raw.section,
                        vector=vector,
                        url=candidate.meta.get("url") or "",
                        citation_count=citation_count if citation_count is not None else -1,
                    )
                    for raw, vector in zip(raw_chunks, vectors, strict=True)
                ]
                store.add_chunks(chunks)
                state.add_findings(
                    [
                        Finding(text=c.text, source_id=candidate.id, sub_question=sub_question.text)
                        for c in raw_chunks
                    ]
                )
        else:
            _emit(on_progress, f"Уже в индексе (кэш): {candidate.title[:80]}")

        state.mark_read(candidate.id)
