"""Entrypoints: `python -m src.cli index` / `python -m src.cli ask "вопрос"`."""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

from . import config
from .agent import loop
from .agent.synthesize import synthesize
from .ingest.chunk import chunk_sections
from .ingest.extract import extract_html_sections, extract_pdf_sections, extract_sections
from .providers import embed, rerank
from .sources.arxiv import ArxivSource
from .sources.semantic_scholar import SemanticScholarSource
from .sources.web import WebSource
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


def _retrieve(store: LanceDBStore, question: str) -> list[dict]:
    query_vector = embed.embed_texts([question])[0]
    candidate_k = config.RERANK_CANDIDATES_K if config.RERANK_ENABLED else config.TOP_K_RETRIEVE
    hits = store.search_hybrid(question, query_vector, k=candidate_k)
    if config.RERANK_ENABLED:
        hits = rerank.rerank(question, hits, top_n=config.TOP_K_RETRIEVE)
    return hits


def _print_answer(question: str, hits: list[dict], gaps: list[str] | None = None) -> None:
    answer = synthesize(question, hits, gaps=gaps)
    print(answer)
    print("\nИсточники:")
    seen: set[str] = set()
    for i, hit in enumerate(hits, start=1):
        title = hit.get("source_title") or hit.get("source_id") or "?"
        if title not in seen:
            print(f"[{i}] {title}")
            seen.add(title)
    if gaps:
        print("\nНепокрытые вопросы (бюджет исследования исчерпан):")
        for gap in gaps:
            print(f"  - {gap}")


def cmd_ask(args: argparse.Namespace) -> None:
    store = LanceDBStore()
    hits = _retrieve(store, args.question)
    _print_answer(args.question, hits)


def cmd_research(args: argparse.Namespace) -> None:
    """Deep-research режим (Milestone 3): воронка + итеративный цикл поверх
    внешних источников (arXiv, Semantic Scholar, web), в отличие от `ask`,
    который только ищет по уже построенному индексу."""
    store = LanceDBStore()
    sources = [ArxivSource(), SemanticScholarSource(), WebSource()]
    state = loop.run(args.question, sources, store)
    hits = _retrieve(store, args.question)
    _print_answer(args.question, hits, gaps=state.gaps)
    print(
        f"\n[итераций: {state.iterations}, прочитано источников: {len(state.read_ids)}, "
        f"найдено кандидатов: {len(state.candidates)}]"
    )


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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
