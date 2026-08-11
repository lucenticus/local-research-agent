from src.ingest.extract import (
    extract_html_sections,
    extract_pdf_header,
    extract_pdf_links,
    extract_pdf_sections,
    extract_sections,
)

_PAPER_MD = """# Заметка

## Abstract

Короткая аннотация.

## Introduction

Основной текст введения.

## References

[1] Автор. Название. Год.

## Acknowledgments

Спасибо рецензентам.
"""


def test_markdown_drops_references_and_acknowledgments():
    sections = extract_sections(_PAPER_MD)
    categories = [s.category for s in sections]
    assert "references" not in categories
    assert "acknowledgments" not in categories
    assert {"abstract", "introduction"} <= set(categories)


def test_markdown_keeps_section_text():
    sections = extract_sections(_PAPER_MD)
    intro = next(s for s in sections if s.category == "introduction")
    assert "Основной текст введения." in intro.text


_PAPER_HTML = """
<html><body>
<h1>Заметка</h1>
<h2>Abstract</h2>
<p>Короткая аннотация.</p>
<h2>Introduction</h2>
<p>Основной текст введения.</p>
<h2>References</h2>
<p>[1] Автор. Название. Год.</p>
<h2>Acknowledgments</h2>
<p>Спасибо рецензентам.</p>
</body></html>
"""


def test_html_drops_references_and_acknowledgments():
    sections = extract_html_sections(_PAPER_HTML)
    categories = [s.category for s in sections]
    assert "references" not in categories
    assert "acknowledgments" not in categories
    assert {"abstract", "introduction"} <= set(categories)


def test_pdf_drops_references_and_acknowledgments(tmp_path):
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    text = (
        "Abstract\nShort abstract text.\n"
        "Introduction\nMain introduction body.\n"
        "References\n[1] Author. Title. Year.\n"
        "Acknowledgments\nThanks to reviewers.\n"
    )
    page.insert_text((50, 50), text)
    pdf_path = tmp_path / "paper.pdf"
    doc.save(pdf_path)
    doc.close()

    sections = extract_pdf_sections(pdf_path)
    categories = [s.category for s in sections]
    assert "references" not in categories
    assert "acknowledgments" not in categories
    assert "abstract" in categories
    assert "introduction" in categories


def test_extract_pdf_links_prefers_annotations_and_adds_plain_text_urls(tmp_path):
    """Аннотации надёжнее регулярки: при извлечении текста длинный URL
    сплошь и рядом рвётся переносом строки и склеить его нельзя."""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Code: see our repository. Also https://huggingface.co/acme/model")
    page.insert_link({
        "kind": fitz.LINK_URI, "from": fitz.Rect(72, 60, 200, 80),
        "uri": "https://github.com/acme/repo",
    })
    path = tmp_path / "paper.pdf"
    doc.save(path)
    doc.close()

    links = extract_pdf_links(path)
    assert links[0] == "https://github.com/acme/repo"          # аннотация первой
    assert "https://huggingface.co/acme/model" in links        # текстовый URL добран


def test_extract_pdf_links_dedupes_and_strips_trailing_punctuation(tmp_path):
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "See https://github.com/acme/repo.")
    page.insert_link({
        "kind": fitz.LINK_URI, "from": fitz.Rect(72, 60, 200, 80),
        "uri": "https://github.com/acme/repo",
    })
    path = tmp_path / "paper.pdf"
    doc.save(path)
    doc.close()

    assert extract_pdf_links(path) == ["https://github.com/acme/repo"]


def test_extract_pdf_links_empty_for_pdf_without_links(tmp_path):
    import fitz

    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "No links here at all.")
    path = tmp_path / "paper.pdf"
    doc.save(path)
    doc.close()

    assert extract_pdf_links(path) == []


def test_extract_pdf_header_stops_at_the_abstract(tmp_path):
    """В шапке лежат авторы с аффилиациями — то, чего нет ни в arXiv API,
    ни в OpenAlex для свежих препринтов."""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Attention Is All You Need")
    page.insert_text((72, 92), "Ashish Vaswani")
    page.insert_text((72, 112), "Google Brain")
    page.insert_text((72, 132), "Abstract")
    page.insert_text((72, 152), "We propose a new architecture.")
    path = tmp_path / "paper.pdf"
    doc.save(path)
    doc.close()

    header = extract_pdf_header(path)
    assert "Ashish Vaswani" in header
    assert "Google Brain" in header
    assert "We propose a new architecture" not in header  # тело статьи отрезано


def test_extract_pdf_header_respects_max_chars(tmp_path):
    """Маркер аннотации не нашёлся (нестандартная вёрстка) — отдаём начало
    страницы, а не всю её."""
    import fitz

    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "A" * 500)
    path = tmp_path / "paper.pdf"
    doc.save(path)
    doc.close()

    assert len(extract_pdf_header(path, max_chars=100)) <= 100
