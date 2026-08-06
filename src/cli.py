"""Entrypoints: `python -m src.cli index` / `python -m src.cli ask "вопрос"`."""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

from . import config
from .agent.research_runner import retrieve, run_research
from .agent.synthesize import synthesize
from .ingest.chunk import chunk_sections
from .ingest.extract import extract_html_sections, extract_pdf_sections, extract_sections
from .providers import embed
from .store.lancedb_store import Chunk, LanceDBStore

_SECTION_EXTRACTORS = {
    ".txt": lambda path: extract_sections(path.read_text(encoding="utf-8")),
    ".md": lambda path: extract_sections(path.read_text(encoding="utf-8")),
    ".html": lambda path: extract_html_sections(path.read_text(encoding="utf-8")),
    ".htm": lambda path: extract_html_sections(path.read_text(encoding="utf-8")),
    ".pdf": lambda path: extract_pdf_sections(path),
}


def _iter_corpus_files(corpus_dir: Path):
    """Генератор путей — не держим список файлов/содержимое корпуса разом (§1)."""
    for path in sorted(corpus_dir.glob("*")):
        if path.is_file() and path.suffix.lower() in _SECTION_EXTRACTORS:
            yield path


def cmd_index(args: argparse.Namespace) -> None:
    corpus_dir = Path(args.corpus_dir)
    store = LanceDBStore()
    all_chunks: list[Chunk] = []
    for path in _iter_corpus_files(corpus_dir):
        sections = _SECTION_EXTRACTORS[path.suffix.lower()](path)
        raw_chunks = chunk_sections(sections)
        if not raw_chunks:
            continue
        vectors = embed.embed_texts([c.text for c in raw_chunks])
        for raw, vector in zip(raw_chunks, vectors, strict=True):
            all_chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    text=raw.text,
                    source_id=path.name,
                    source_title=path.stem,
                    section=raw.section,
                    vector=vector,
                )
            )
    store.rebuild(all_chunks)
    print(f"Индекс построен: {len(all_chunks)} чанков из {corpus_dir}")


def _format_source_line(index: int, title: str, url: str = "", citation_count: int | None = None) -> str:
    line = f"[{index}] {title}"
    if citation_count is not None:
        line += f" (цитирований: {citation_count})"
    if url:
        # Голый URL на отдельной строке — большинство терминалов сами
        # превращают его в кликабельную ссылку, без доп. разметки.
        line += f"\n    {url}"
    return line


def _print_gaps(gaps: list[str] | None) -> None:
    if gaps:
        print("\nНепокрытые вопросы (бюджет исследования исчерпан):")
        for gap in gaps:
            print(f"  - {gap}")


def _print_answer(question: str, hits: list[dict], gaps: list[str] | None = None) -> None:
    answer = synthesize(question, hits, gaps=gaps)
    print(answer)
    print("\nИсточники:")
    seen: set[str] = set()
    i = 0
    for hit in hits:
        title = hit.get("source_title") or hit.get("source_id") or "?"
        if title in seen:
            continue
        seen.add(title)
        i += 1
        citation_count = hit.get("citation_count")
        print(
            _format_source_line(
                i, title, url=hit.get("url") or "",
                citation_count=citation_count if citation_count is not None and citation_count >= 0 else None,
            )
        )
    _print_gaps(gaps)


def cmd_ask(args: argparse.Namespace) -> None:
    store = LanceDBStore()
    hits = retrieve(store, args.question)
    _print_answer(args.question, hits)


def cmd_research(args: argparse.Namespace) -> None:
    """Deep-research режим (Milestone 3): воронка + итеративный цикл поверх
    внешних источников (arXiv, Semantic Scholar, web), в отличие от `ask`,
    который только ищет по уже построенному индексу."""
    store = LanceDBStore()
    result = run_research(args.question, store, on_progress=print)
    print()
    print(result.answer)
    print("\nИсточники:")
    for i, source in enumerate(result.sources, start=1):
        print(_format_source_line(i, source.title, url=source.url, citation_count=source.citation_count))
    _print_gaps(result.gaps)
    print(
        f"\n[итераций: {result.iterations}, прочитано источников: {result.read_count}, "
        f"найдено кандидатов: {result.candidates_count}]"
    )


def cmd_serve(args: argparse.Namespace) -> None:
    """Веб-интерфейс (research) — FastAPI + одна HTML-страница, без сборки."""
    import uvicorn

    uvicorn.run("src.web.app:app", host=args.host, port=args.port, reload=False)


def main() -> None:
    parser = argparse.ArgumentParser(prog="research-agent")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Построить индекс из corpus/")
    p_index.add_argument(
        "--corpus-dir", default=str(config.CORPUS_DIR), dest="corpus_dir"
    )
    p_index.set_defaults(func=cmd_index)

    p_ask = sub.add_parser("ask", help="Задать вопрос по индексу")
    p_ask.add_argument("question")
    p_ask.set_defaults(func=cmd_ask)

    p_research = sub.add_parser(
        "research", help="Deep-research: воронка + итеративный цикл поверх внешних источников"
    )
    p_research.add_argument("question")
    p_research.set_defaults(func=cmd_research)

    p_serve = sub.add_parser("serve", help="Запустить веб-интерфейс (research)")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
