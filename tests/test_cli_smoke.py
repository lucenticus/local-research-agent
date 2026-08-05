"""Интеграционный smoke-тест end-to-end на крошечном корпусе.

LLM и эмбеддинги мокаются (§7 CLAUDE.md: тесты офлайн и быстрые) — проверяем
реальную склейку index -> LanceDB -> search -> synthesize -> cli.
"""

from __future__ import annotations

import argparse

from src import cli, config
from src.providers import embed, llm


def test_index_and_ask_smoke(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path / "lancedb")

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

    cli.cmd_index(argparse.Namespace(corpus_dir=str(corpus_dir)))
    index_out = capsys.readouterr().out
    assert "Индекс построен: 1 чанков" in index_out

    cli.cmd_ask(argparse.Namespace(question="Кто такой кит?"))
    ask_out = capsys.readouterr().out
    assert "Кит — млекопитающее [1]." in ask_out
    assert "[1] doc1" in ask_out
