"""Интеграционный smoke-тест end-to-end на крошечном корпусе.

LLM и эмбеддинги мокаются (§7 CLAUDE.md: тесты офлайн и быстрые) — проверяем
реальную склейку index -> Qdrant -> search -> synthesize -> cli.

`QdrantStore` подменяется на embedded (`path=tmp_path`) — тот же трюк, что
раньше был через `config.INDEX_DIR` для LanceDB: реальный векторный поиск
(не мок), но без Docker и сети (см. `QdrantStore.__init__`'s `path` — только
для тестов, продовый код всегда ходит в Docker-сервер по `config.QDRANT_URL`).
`embed.embed_sparse` тоже мокается — `add_texts`/`_chunks_from_sections`
теперь считают sparse-вектор при индексации, а `search_hybrid` — при поиске,
оба реального bge-m3-вызова здесь не нужны для теста склейки.
"""

from __future__ import annotations

import argparse

from src import cli
from src.providers import embed, llm, rerank
from src.store.qdrant_store import QdrantStore


def test_index_and_ask_smoke(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "QdrantStore", lambda *a, **kw: QdrantStore(path=str(tmp_path / "qdrant")))
    monkeypatch.setattr(embed, "embed_sparse", lambda texts: [{} for _ in texts])
    monkeypatch.setattr(embed, "embed_texts_hybrid", lambda texts: ([[float(len(t))] for t in texts], [{} for _ in texts]))

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "doc1.txt").write_text(
        "Кит — крупное морское млекопитающее, живущее в океане.", encoding="utf-8"
    )

    # Единственный чанк в индексе -> любой вектор запроса найдёт именно его,
    # реальная семантика эмбеддингов здесь не нужна для smoke-теста склейки.
    monkeypatch.setattr(embed, "embed_texts", lambda texts: [[float(len(t))] for t in texts])
    monkeypatch.setattr(llm, "build_chat_prompt", lambda system, user: user)
    monkeypatch.setattr(llm, "generate", lambda prompt, **kw: "Кит — млекопитающее [1].")
    # rerank() сам по себе покрыт tests/test_rerank.py, здесь только проверяем
    # склейку cli -> rerank -> synthesize, поэтому мокаем на уровне публичной функции.
    monkeypatch.setattr(rerank, "rerank", lambda query, candidates, top_n: candidates[:top_n])

    cli.cmd_index(argparse.Namespace(corpus_dir=str(corpus_dir), mcp_dirs=[]))
    index_out = capsys.readouterr().out
    assert "Индекс построен: 1 чанков" in index_out

    cli.cmd_ask(argparse.Namespace(question="Кто такой кит?"))
    ask_out = capsys.readouterr().out
    assert "Кит — млекопитающее [1]." in ask_out
    assert "[1] doc1" in ask_out


class _FakeMCPTool:
    def __init__(self, responses):
        self._responses = responses  # {frozenset(kwargs.items()): result}

    def invoke(self, kwargs):
        return self._responses[frozenset(kwargs.items())]


def test_index_pulls_extra_docs_via_mcp_filesystem_server(tmp_path, monkeypatch, capsys):
    """`--mcp-dir` — MCP filesystem-сервер замокан офлайн (search_files +
    read_text_file), реальный npx-процесс не поднимается."""
    monkeypatch.setattr(cli, "QdrantStore", lambda *a, **kw: QdrantStore(path=str(tmp_path / "qdrant")))
    monkeypatch.setattr(embed, "embed_texts_hybrid", lambda texts: ([[float(len(t))] for t in texts], [{} for _ in texts]))
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    search_tool = _FakeMCPTool(
        {
            frozenset({"path": "/notes", "pattern": "*.txt"}.items()): "no matches found",
            frozenset({"path": "/notes", "pattern": "*.md"}.items()): "/notes/whales.md",
            frozenset({"path": "/notes", "pattern": "*.html"}.items()): "No matches found",
            frozenset({"path": "/notes", "pattern": "*.htm"}.items()): "No matches found",
        }
    )
    search_tool.name = "search_files"
    read_tool = _FakeMCPTool(
        {frozenset({"path": "/notes/whales.md"}.items()): [{"type": "text", "text": "Киты живут в океане."}]}
    )
    read_tool.name = "read_text_file"

    monkeypatch.setattr(cli, "get_mcp_tools", lambda connections: [search_tool, read_tool])
    monkeypatch.setattr(embed, "embed_texts", lambda texts: [[float(len(t))] for t in texts])

    cli.cmd_index(argparse.Namespace(corpus_dir=str(corpus_dir), mcp_dirs=["/notes"]))
    out = capsys.readouterr().out
    assert "Индекс построен: 1 чанков" in out
    assert "1 MCP-директории" in out
