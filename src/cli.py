"""Entrypoints: `python -m src.cli index` / `python -m src.cli ask "вопрос"`."""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

from . import config
from .agent.research_runner import ResearchResult, retrieve, run_followup, run_research
from .agent.synthesize import synthesize
from .ingest.chunk import chunk_sections
from .ingest.extract import Section, extract_html_sections, extract_pdf_sections, extract_sections
from .providers import embed, tracing
from .providers.mcp_client import content_to_text, get_mcp_tools
from .store.qdrant_store import Chunk, QdrantStore

_SECTION_EXTRACTORS = {
    ".txt": lambda path: extract_sections(path.read_text(encoding="utf-8")),
    ".md": lambda path: extract_sections(path.read_text(encoding="utf-8")),
    ".html": lambda path: extract_html_sections(path.read_text(encoding="utf-8")),
    ".htm": lambda path: extract_html_sections(path.read_text(encoding="utf-8")),
    ".pdf": lambda path: extract_pdf_sections(path),
}

# Текстовые (не PDF) расширения, доступные через MCP filesystem-сервер
# (см. _iter_mcp_files) — extract_pdf_sections открывает PDF через PyMuPDF
# по реальному пути на диске, а MCP filesystem-сервер отдаёт содержимое
# файла текстом/base64 по сети MCP-протокола, не пишет временный файл;
# делать этот temp-file round-trip для PDF, как funnel.py делает для
# скачанных статей, здесь пока не стали — не тот случай использования
# (PDF из MCP-директории проще прочитать напрямую с диска, чем через MCP).
_MCP_TEXT_EXTRACTORS = {
    ".txt": extract_sections,
    ".md": extract_sections,
    ".html": extract_html_sections,
    ".htm": extract_html_sections,
}


def _iter_corpus_files(corpus_dir: Path):
    """Генератор путей — не держим список файлов/содержимое корпуса разом (§1)."""
    for path in sorted(corpus_dir.glob("*")):
        if path.is_file() and path.suffix.lower() in _SECTION_EXTRACTORS:
            yield path


def _iter_mcp_files(root_dir: str):
    """Находит и читает `.txt`/`.md`/`.html`/`.htm`-файлы под `root_dir` через
    MCP filesystem-сервер (`@modelcontextprotocol/server-filesystem`,
    запускается на лету через `npx`) — даёт `index` забирать документы из
    ЛЮБОЙ директории на диске, а не только из `corpus/`, без правки
    config.py под каждую новую директорию. Рекурсивно (search_files сам
    обходит поддиректории)."""
    connections = {
        "fs": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", root_dir],
        }
    }
    tools = {t.name: t for t in get_mcp_tools(connections)}
    search_tool, read_tool = tools["search_files"], tools["read_text_file"]

    seen: set[str] = set()
    for ext in _MCP_TEXT_EXTRACTORS:
        matches = content_to_text(search_tool.invoke({"path": root_dir, "pattern": f"*{ext}"})).strip()
        if not matches or matches.lower() == "no matches found":  # confirmed real server response
            continue
        for file_path in (line.strip() for line in matches.splitlines()):
            if not file_path or file_path in seen:
                continue
            seen.add(file_path)
            text = content_to_text(read_tool.invoke({"path": file_path}))
            yield file_path, ext, text


def _chunks_from_sections(sections: list[Section], source_id: str, source_title: str) -> list[Chunk]:
    raw_chunks = chunk_sections(sections)
    if not raw_chunks:
        return []
    # dense+sparse одним forward pass'ом (providers/embed.py) — оба нужны
    # для гибридного поиска в Qdrant, дешевле посчитать вместе, чем отдельно.
    vectors, sparse_vectors = embed.embed_texts_hybrid([c.text for c in raw_chunks])
    return [
        Chunk(
            id=str(uuid.uuid4()), text=raw.text, source_id=source_id,
            source_title=source_title, section=raw.section, vector=vector, sparse=sparse,
        )
        for raw, vector, sparse in zip(raw_chunks, vectors, sparse_vectors, strict=True)
    ]


def cmd_index(args: argparse.Namespace) -> None:
    corpus_dir = Path(args.corpus_dir)
    store = QdrantStore()
    all_chunks: list[Chunk] = []
    for path in _iter_corpus_files(corpus_dir):
        sections = _SECTION_EXTRACTORS[path.suffix.lower()](path)
        all_chunks.extend(_chunks_from_sections(sections, source_id=path.name, source_title=path.stem))

    for mcp_dir in args.mcp_dirs:
        for file_path, ext, text in _iter_mcp_files(mcp_dir):
            sections = _MCP_TEXT_EXTRACTORS[ext](text)
            title = Path(file_path).stem
            all_chunks.extend(_chunks_from_sections(sections, source_id=file_path, source_title=title))

    store.rebuild(all_chunks)
    print(f"Индекс построен: {len(all_chunks)} чанков из {corpus_dir}"
          + (f" + {len(args.mcp_dirs)} MCP-директори{'й' if len(args.mcp_dirs) != 1 else 'и'}" if args.mcp_dirs else ""))


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
    store = QdrantStore()
    hits = retrieve(store, args.question)
    _print_answer(args.question, hits)


def _print_research_result(result: ResearchResult) -> None:
    print()
    print(result.answer)
    print("\nИсточники:")
    for i, source in enumerate(result.sources, start=1):
        print(_format_source_line(i, source.title, url=source.url, citation_count=source.citation_count))
    _print_gaps(result.gaps)

    if result.candidates:
        # Номер в списке — то, что пользователь вводит в "подробнее N", чтобы
        # раскрыть подробнее конкретную найденную тему (см. cmd_research).
        print("\nВсе найденные кандидаты (как агент сузил поиск):")
        for i, c in enumerate(result.candidates, start=1):
            mark = "✓ прочитан" if c.read else "  найден"
            score = f"score={c.triage_score:.3f}" if c.triage_score is not None else "score=—"
            citation = f", цитирований={c.citation_count}" if c.citation_count is not None else ""
            print(f"  [{i}] [{mark}] {score}{citation} ({c.source}) {c.title}")

    print(
        f"\n[итераций: {result.iterations}, прочитано источников: {result.read_count}, "
        f"найдено кандидатов: {result.candidates_count}]"
    )


def cmd_research(args: argparse.Namespace) -> None:
    """Deep-research режим (Milestone 3): воронка + итеративный цикл поверх
    внешних источников (arXiv, Semantic Scholar, web), в отличие от `ask`,
    который только ищет по уже построенному индексу.

    Отдельная коллекция от `ask`/`index` (config.QDRANT_RESEARCH_COLLECTION) —
    иначе demo-корпус из corpus/ просачивается в источники реальных ответов
    (найдено реальным прогоном).

    После первого ответа переходит в интерактивный follow-up-режим (тот же
    диалог, `agent/research_runner.run_followup` — переиспользует уже
    накопленный `ResearchState`, а не начинает с нуля): можно задать
    уточняющий вопрос или написать `подробнее N`, чтобы форсировать
    deep-read N-го кандидата из списка выше и раскрыть эту тему подробнее.
    Пустая строка / Ctrl-D — выход."""
    store = QdrantStore(collection_name=config.QDRANT_RESEARCH_COLLECTION)
    result = run_research(args.question, store, on_progress=print)
    _print_research_result(result)

    print(
        "\nМожно задать уточняющий вопрос, написать «подробнее N» — раскрыть "
        "тему N из списка кандидатов, или просто нажать Enter, чтобы выйти."
    )
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break

        message = line
        focus_candidate_id = None
        lowered = line.lower()
        if lowered.startswith("подробнее ") or lowered.startswith("more "):
            _, _, arg = line.partition(" ")
            try:
                candidate = result.candidates[int(arg.strip()) - 1]
            except (ValueError, IndexError):
                print(f"Не понял номер — используйте «подробнее N» с номером из списка (1-{len(result.candidates)}).")
                continue
            message = f"Расскажи подробнее об источнике: {candidate.title}"
            focus_candidate_id = candidate.id

        result = run_followup(
            message, result.state, store, on_progress=print, focus_candidate_id=focus_candidate_id
        )
        _print_research_result(result)


def cmd_serve(args: argparse.Namespace) -> None:
    """Веб-интерфейс (research) — FastAPI + одна HTML-страница, без сборки."""
    import uvicorn

    uvicorn.run("src.web.app:app", host=args.host, port=args.port, reload=False)


def cmd_digest(args: argparse.Namespace) -> None:
    """Дайджест свежих статей по ИИ — browse по arXiv-категориям за
    последние N дней, не Q&A. См. src/agent/digest.py."""
    from .agent.digest import run_digest

    result = run_digest(
        days=args.days, categories=args.categories or None, limit=args.limit,
        summarize=not args.no_summary, query=args.query, deep=args.deep,
        on_progress=print if args.deep else None,  # --deep не быстрый - видно, что не зависло
    )
    scope = f" по «{result.query}»" if result.query else ""
    print(
        f"\nДайджест{scope}: {len(result.items)} статей за последние {result.days} дн. "
        f"в категориях {', '.join(result.categories)}\n"
    )
    if result.summary:
        print("Обзор тем (сгенерирован моделью, не факт с источником):")
        print(result.summary)
        print()
    for i, item in enumerate(result.items, start=1):
        authors = item.meta.get("authors") or []
        authors_line = ", ".join(authors[:3]) + (" и др." if len(authors) > 3 else "")
        print(f"[{i}] {item.title}")
        if authors_line:
            print(f"    {authors_line}")
        published = (item.published_date or "")[:10]
        print(f"    {published}  {item.url}")

        analysis = result.analyses.get(item.id)
        if analysis is not None:
            print(f"    Саммари: {analysis.summary_ru}")
            details = analysis.details
            if details is None:
                print("    Метаданные: — (статья ещё не проиндексирована в OpenAlex)")
            else:
                cited = details.citation_count if details.citation_count is not None else "—"
                venue = details.venue or "—"
                print(f"    Цитирований: {cited}  ·  Venue: {venue}")
                for author in details.authors:
                    inst = author.institution or "—"
                    h = author.h_index if author.h_index is not None else "—"
                    print(f"      {author.name} — {inst}, h-index: {h}")

            insights = analysis.insights
            if insights is None:
                print("    Разбор PDF: — (не удалось скачать или разобрать)")
            else:
                print(f"    Разбор PDF (секции: {', '.join(insights.sections_used) or '—'}):")
                for line in insights.findings_ru.splitlines():
                    print(f"      {line}" if line.strip() else "")
                if insights.authors:
                    print("    Авторы и аффилиации (из PDF):")
                    for author in insights.authors:
                        print(f"      {author.name} — {author.affiliation or '—'}")
                if insights.code_links:
                    print("    Код и модели:")
                    for link in insights.code_links:
                        # stars=0 — реальный ноль, None — не смогли узнать
                        stars = f"  ★ {link.stars}" if link.stars is not None else ""
                        print(f"      [{link.kind}] {link.url}{stars}")
                else:
                    print("    Код и модели: — (ссылок в PDF не нашлось)")
        print()


def cmd_mcp_serve(args: argparse.Namespace) -> None:
    """MCP-сервер (stdio) — ask()/research() как MCP-инструменты для
    любого MCP-клиента (Claude Code, Claude Desktop и т.п.), см.
    src/mcp_server.py."""
    from .mcp_server import serve

    serve()


def main() -> None:
    tracing.enable_if_configured()
    parser = argparse.ArgumentParser(prog="research-agent")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Построить индекс из corpus/")
    p_index.add_argument(
        "--corpus-dir", default=str(config.CORPUS_DIR), dest="corpus_dir"
    )
    p_index.add_argument(
        "--mcp-dir", action="append", default=[], dest="mcp_dirs",
        help="Дополнительная директория (любая на диске) — читается через MCP "
             "filesystem-сервер (npx), не только corpus/. Можно указать несколько раз.",
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

    p_digest = sub.add_parser(
        "digest", help="Дайджест свежих статей по ИИ за последние N дней (browse по arXiv, не Q&A)"
    )
    p_digest.add_argument("--days", type=int, default=None, help=f"По умолчанию {config.DIGEST_DEFAULT_DAYS}")
    p_digest.add_argument(
        "--category", action="append", dest="categories", default=[],
        help=f"arXiv-категория, можно несколько раз (по умолчанию {config.ARXIV_AI_CATEGORIES})",
    )
    p_digest.add_argument(
        "--limit", type=int, default=None,
        help=f"По умолчанию {config.DIGEST_DEFAULT_LIMIT}. Игнорируется при --query — "
        f"там показывается топ {config.DIGEST_QUERY_TOP_K} самых релевантных",
    )
    p_digest.add_argument(
        "--query", default=None,
        help=f"Топ-{config.DIGEST_QUERY_TOP_K} самых релевантных теме статей за период "
        f"(ранжируется локально по расширенному пулу, не через arXiv-запрос)",
    )
    p_digest.add_argument("--no-summary", action="store_true", help="Не генерировать LLM-обзор тем")
    p_digest.add_argument(
        "--deep", action="store_true",
        help=f"Глубокий анализ каждой статьи: русское саммари, разбор PDF (результаты, "
             f"сравнения, авторы с аффилиациями, ссылки на код со звёздами GitHub) и "
             f"цитируемость/venue/h-index через OpenAlex (медленнее, лимит "
             f"{config.DIGEST_DEEP_MAX_ITEMS} статей)",
    )
    p_digest.set_defaults(func=cmd_digest)

    p_mcp_serve = sub.add_parser(
        "mcp-serve", help="Запустить MCP-сервер (stdio) — ask/research как MCP-инструменты"
    )
    p_mcp_serve.set_defaults(func=cmd_mcp_serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
