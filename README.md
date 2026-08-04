# local-research-agent

Локальный агент научных исследований и трендов (MacBook M4 Air, 16 ГБ, всё
локально). Полный план и жёсткие ограничения — в
[DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) и [CLAUDE.md](CLAUDE.md).

## Milestone 0 — тупой сквозной путь

Single-pass RAG без цикла и воронки: `index` строит эмбеддинг-индекс из
`corpus/` в LanceDB, `ask` находит top-k чанков и просит локальную LLM
(Qwen через MLX) ответить с цитатами `[n]`.

### Установка

```bash
uv venv
uv pip install -r requirements.txt
```

### Запуск

```bash
python -m src.cli index
python -m src.cli ask "Что такое RAG?"
```

`index` читает все `.txt`/`.md` файлы из `corpus/` (в репозитории лежит
маленький тестовый корпус из 3 заметок про RAG/векторные БД/локальные LLM).

### Тесты

```bash
uv run pytest -q
```

Тесты офлайн: эмбеддинги и LLM мокаются, реальная загрузка Qwen3.5/bge-m3 не
требуется для прогона тестов.

## Известные допущения (`ARCH-Q`)

Непроверенные на реальном железе допущения помечены `# ARCH-Q:` прямо в коде
(`src/config.py`, `src/providers/embed.py`, `src/providers/llm.py`) —
HF-репо/тег LLM, kwarg устройства FlagEmbedding, поддержка
`enable_thinking` в chat-template. Проверить и зафиксировать при первом
реальном запуске на M4 Air, включая пик памяти (см. критерий готовности в
`DEVELOPMENT_PLAN.md`).
