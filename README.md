# local-research-agent

Локальный агент научных исследований и трендов (MacBook M4 Air, 16 ГБ, всё
локально). Полный план и жёсткие ограничения — в
[DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) и [CLAUDE.md](CLAUDE.md).

## Milestone 0 — тупой сквозной путь

Single-pass RAG без цикла и воронки: `index` строит эмбеддинг-индекс из
`corpus/` в LanceDB, `ask` находит top-k чанков и просит локальную LLM
(Qwen через MLX) ответить с цитатами `[n]`.

## Milestone 1 — hybrid retrieval + чистое извлечение

- `ingest/extract.py` — секция-осознанное извлечение (markdown/HTML/PDF),
  отбрасывает References/Bibliography/Acknowledgments/Appendix.
- `ingest/chunk.py` — `chunk_sections` режет чанки по границам секций.
- `store/lancedb_store.py` — гибридный поиск (`search_hybrid`): dense-вектор
  + LanceDB FTS (BM25, `language="Russian"`), слияние через RRF. `search_dense`
  оставлен как baseline для сравнения.
- `ask` теперь использует `search_hybrid` по умолчанию.

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

`index` читает `.txt`/`.md`/`.html`/`.pdf` из `corpus/` (в репозитории лежит
маленький тестовый корпус: 3 заметки про RAG/векторные БД/локальные LLM + одна
статья-образец с Abstract/Introduction/Method/Results/Conclusion/References/
Acknowledgments — проверить отброс References/Acknowledgments).

### Тесты

```bash
uv run pytest -q
```

Тесты офлайн: эмбеддинги и LLM мокаются, реальная загрузка Qwen3.5/bge-m3 не
требуется для прогона тестов.

### Eval retrieval@k (dense vs hybrid)

Не pytest — гоняет реальные bge-m3-эмбеддинги, требует уже построенного
индекса:

```bash
python -m src.cli index
python -m scripts.eval_retrieval
```

Последний прогон (2026-08-05, 8 эталонных пар, k=3): dense 8/8, hybrid 8/8 —
hybrid не хуже dense. Корпус пока мал, чтобы разница в recall проявилась, но
ранжирование внутри top-k реально разное (проверено вручную на лексическом
запросе про формулу RRF).

## Известные допущения (`ARCH-Q`)

Непроверенные на реальном железе допущения помечены `# ARCH-Q:` прямо в коде
(`src/config.py`, `src/providers/embed.py`, `src/providers/llm.py`) —
HF-репо/тег LLM, kwarg устройства FlagEmbedding, поддержка
`enable_thinking` в chat-template. Проверить и зафиксировать при первом
реальном запуске на M4 Air, включая пик памяти (см. критерий готовности в
`DEVELOPMENT_PLAN.md`).
