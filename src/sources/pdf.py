"""Скачивание PDF статьи и разбор его на секции + ссылки.

Сетевая часть отделена от разбора (`ingest/extract.py` умеет только читать
уже лежащий на диске файл) — тот же шов «внешняя зависимость за тонким
интерфейсом», что и у остальных `sources/`.

Живёт здесь, а не в `agent/funnel.py`, потому что потребителей стало двое:
воронка (глубокое чтение кандидата) и дайджест (разбор конкретной статьи по
кнопке, см. `agent/digest.py::analyze_item`). Раньше скачивание было
приватной функцией внутри funnel, и дайджесту пришлось бы либо тащить за
собой весь funnel (embed, MCP, langchain-инструменты), либо завести вторую
копию.
"""

from __future__ import annotations

import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .. import config
from ..ingest.extract import (
    Section,
    extract_pdf_header,
    extract_pdf_links,
    extract_pdf_sections,
)

_TIMEOUT_SECONDS = 30


@dataclass
class FetchedPdf:
    sections: list[Section] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    header: str = ""  # шапка первой страницы: авторы, аффилиации, почты


def fetch_pdf(pdf_url: str) -> FetchedPdf:
    """PDF по ссылке -> его секции, все встреченные URL и шапка первой страницы.

    Читается не больше `config.FUNNEL_MAX_PDF_BYTES` — не тянуть произвольно
    большой файл в память. Файл кладётся во временный, потому что PyMuPDF
    работает с путём, и удаляется в `finally` независимо от исхода разбора.
    """
    request = urllib.request.Request(pdf_url, headers={"User-Agent": "local-research-agent/0.1"})
    with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
        body = response.read(config.FUNNEL_MAX_PDF_BYTES)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(body)
        tmp_path = Path(tmp.name)
    try:
        return FetchedPdf(
            sections=extract_pdf_sections(tmp_path),
            links=extract_pdf_links(tmp_path),
            header=extract_pdf_header(tmp_path),
        )
    finally:
        tmp_path.unlink(missing_ok=True)


def fetch_pdf_sections(pdf_url: str) -> list[Section]:
    """Только секции — то, что нужно воронке (`agent/funnel.py`)."""
    return fetch_pdf(pdf_url).sections
