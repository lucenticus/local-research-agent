"""LangChain `Embeddings` поверх `providers/embed.py` (bge-m3, MPS,
резидентная модель — см. `embed.py`) — тонкая обёртка для экспериментов с
LangChain/LangGraph (например, `langchain`-совместимый vector store), сама
модель и её загрузка не дублируются."""

from __future__ import annotations

from . import embed


class MLXBGEEmbeddings:
    """Реализует протокол `langchain_core.embeddings.Embeddings`
    (`embed_documents`/`embed_query`) без прямой зависимости от него — сам
    протокол duck-typed, дополнительный импорт не нужен."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed.embed_texts(texts)

    def embed_query(self, text: str) -> list[float]:
        return embed.embed_texts([text])[0]
