"""Секция-осознанное извлечение: markdown / HTML / PDF -> список секций.

Отбрасывает References/Bibliography/Acknowledgments/Appendix — они не несут
фактов для синтеза и только раздувают контекст (§1: контекст под лимитом).
PDF не хранит семантическую структуру (в отличие от HTML/markdown), поэтому
заголовки в PDF распознаются эвристикой по словарю известных названий секций
— это может ошибаться на нестандартных макетах (двухколоночная вёрстка,
нестандартные названия разделов).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_SECTION_ALIASES: dict[str, str] = {
    "abstract": "abstract",
    "аннотация": "abstract",
    "резюме": "abstract",
    "introduction": "introduction",
    "введение": "introduction",
    "related work": "related_work",
    "background": "related_work",
    "обзор литературы": "related_work",
    "смежные работы": "related_work",
    "method": "method",
    "methods": "method",
    "methodology": "method",
    "approach": "method",
    "метод": "method",
    "методология": "method",
    "experiment": "results",
    "experiments": "results",
    "experimental setup": "results",
    "results": "results",
    "результаты": "results",
    "эксперименты": "results",
    "discussion": "discussion",
    "обсуждение": "discussion",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
    "заключение": "conclusion",
    "выводы": "conclusion",
    "references": "references",
    "bibliography": "references",
    "список литературы": "references",
    "литература": "references",
    "acknowledgments": "acknowledgments",
    "acknowledgements": "acknowledgments",
    "благодарности": "acknowledgments",
    "appendix": "appendix",
    "приложение": "appendix",
}

DROP_CATEGORIES = {"references", "acknowledgments", "appendix"}

_MD_HEADER_RE = re.compile(r"^(#{1,3})\s+(.*\S)\s*$")
_URL_RE = re.compile(r"https?://[^\s<>\"{}|\\^`\[\]]+")
_ABSTRACT_MARKER_RE = re.compile(r"^\s*(abstract|аннотация|резюме)\b", re.IGNORECASE | re.MULTILINE)
_PDF_HEADER_RE = re.compile(
    r"^\s*(?:[0-9]+[.)]?\s*)?(" + "|".join(sorted(_SECTION_ALIASES, key=len, reverse=True)) + r")\s*$",
    re.IGNORECASE,
)


@dataclass
class Section:
    name: str
    category: str
    text: str


def _normalize_header(header: str) -> str:
    return re.sub(r"^\s*\d+[.):]?\s*", "", header.strip().lower())


def _categorize(header: str) -> str:
    return _SECTION_ALIASES.get(_normalize_header(header), "other")


def split_markdown_sections(text: str) -> list[Section]:
    """Делит markdown-текст на секции по заголовкам `#`/`##`/`###`.

    Текст до первого заголовка попадает в секцию с именем "" (category="other").
    """
    current_name = ""
    current_category = "other"
    buffer: list[str] = []
    sections: list[Section] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append(Section(name=current_name, category=current_category, text=body))

    for line in text.splitlines():
        match = _MD_HEADER_RE.match(line)
        if match:
            flush()
            current_name = match.group(2)
            current_category = _categorize(current_name)
            buffer = []
        else:
            buffer.append(line)
    flush()
    return sections


def extract_sections(text: str) -> list[Section]:
    """Секции markdown-текста без References/Acknowledgments/Appendix."""
    return [s for s in split_markdown_sections(text) if s.category not in DROP_CATEGORIES]


def extract_html_sections(html: str) -> list[Section]:
    """HTML-статья -> секции. h1-h3 становятся заголовками, p/li — телом."""
    from bs4 import BeautifulSoup  # lazy import: опциональная зависимость

    soup = BeautifulSoup(html, "html.parser")
    lines: list[str] = []
    for el in soup.find_all(["h1", "h2", "h3", "p", "li"]):
        text = el.get_text(" ", strip=True)
        if not text:
            continue
        lines.append(f"## {text}" if el.name in ("h1", "h2", "h3") else text)
    return extract_sections("\n".join(lines))


def extract_pdf_sections(path: str | Path) -> list[Section]:
    """PDF -> секции. Заголовки распознаются эвристикой (см. docstring модуля)."""
    import fitz  # PyMuPDF, lazy import: опциональная зависимость

    lines: list[str] = []
    with fitz.open(path) as doc:
        for page in doc:
            for raw_line in page.get_text().splitlines():
                stripped = raw_line.strip()
                lines.append(f"## {stripped}" if _PDF_HEADER_RE.match(stripped) else raw_line)
    return extract_sections("\n".join(lines))


def extract_pdf_header(path: str | Path, max_chars: int = 2000) -> str:
    """Шапка первой страницы: всё до аннотации.

    Именно здесь у статьи лежат авторы с аффилиациями и почтами — данных,
    которых нет ни в arXiv API (там только имена), ни в OpenAlex для свежих
    препринтов (не проиндексированы). Возвращается сырой текст, разбор
    оставлен вызывающему (`agent/digest.py`).

    Обрезается по первому вхождению "Abstract"/"Аннотация" и по `max_chars`:
    если разметка нестандартная и маркер не нашёлся, лучше отдать первые
    пару тысяч символов, чем всю страницу.
    """
    import fitz  # PyMuPDF, lazy import: опциональная зависимость

    with fitz.open(path) as doc:
        if doc.page_count == 0:
            return ""
        text = doc[0].get_text()

    match = _ABSTRACT_MARKER_RE.search(text)
    if match:
        text = text[: match.start()]
    return text[:max_chars].strip()


def extract_pdf_links(path: str | Path) -> list[str]:
    """Все URL статьи: сначала из link-аннотаций PDF, потом из текста.

    Аннотации — основной источник и они надёжнее регулярки по тексту: при
    извлечении текста PDF длинный URL сплошь и рядом разрывается переносом
    строки, и склеить его обратно без догадок нельзя. LaTeX'овский \\url{} как
    раз создаёт аннотацию, поэтому ссылки на репозитории из статей обычно
    вытаскиваются точно. Регулярка добирает URL, набранные простым текстом,
    без гиперссылки.

    Порядок сохраняется (аннотации первыми), дубликаты убираются — вызывающий
    код (agent/digest.py) фильтрует это до github/huggingface и показывает
    пользователю как есть. Ссылки принципиально не отдаются на откуп LLM:
    правдоподобный, но выдуманный URL репозитория — ровно тот вид
    галлюцинации, который тут недопустим.
    """
    import fitz  # PyMuPDF, lazy import: опциональная зависимость

    urls: list[str] = []
    seen: set[str] = set()

    def add(url: str) -> None:
        cleaned = url.strip().rstrip(".,;)]}>'\"")
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            urls.append(cleaned)

    with fitz.open(path) as doc:
        for page in doc:
            for link in page.get_links():
                if link.get("uri"):
                    add(link["uri"])
        for page in doc:
            for match in _URL_RE.finditer(page.get_text()):
                add(match.group(0))
    return urls
