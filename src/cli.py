"""Entrypoints: `python -m src.cli index` / `python -m src.cli ask "вопрос"`."""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

from . import config
from .agent.synthesize import synthesize
from .ingest.chunk import chunk_text
from .providers import embed
from .store.lancedb_store import Chunk, LanceDBStore


def _iter_corpus_files(corpus_dir: Path):
    """Генератор путей — не держим список файлов/содержимое корпуса разом (§1)."""
    for path in sorted(corpus_dir.glob("*")):
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}:
            yield path


def cmd_index(args: argparse.Namespace) -> None:
    corpus_dir = Path(args.corpus_dir)
    store = LanceDBStore()
    all_chunks: list[Chunk] = []
    for path in _iter_corpus_files(corpus_dir):
        text = path.read_text(encoding="utf-8")
        raw_chunks = chunk_text(text)
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
                    vector=vector,
                )
            )
    store.rebuild(all_chunks)
    print(f"Индекс построен: {len(all_chunks)} чанков из {corpus_dir}")


def cmd_ask(args: argparse.Namespace) -> None:
    store = LanceDBStore()
    query_vector = embed.embed_texts([args.question])[0]
    hits = store.search(query_vector, k=config.TOP_K_RETRIEVE)
    answer = synthesize(args.question, hits)
    print(answer)
    print("\nИсточники:")
    seen: set[str] = set()
    for i, hit in enumerate(hits, start=1):
        title = hit.get("source_title") or hit.get("source_id") or "?"
        if title not in seen:
            print(f"[{i}] {title}")
            seen.add(title)


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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
