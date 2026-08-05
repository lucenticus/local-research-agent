from src.ingest.extract import extract_html_sections, extract_pdf_sections, extract_sections

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
