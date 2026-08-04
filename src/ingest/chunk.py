"""Наивный фикс-чанкинг с перекрытием по символам.

Секция-осознанное извлечение и умный чанкинг по границам секций — Milestone 1.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import config


@dataclass
class RawChunk:
    text: str
    index: int


def chunk_text(
    text: str,
    size: int = config.CHUNK_SIZE_CHARS,
    overlap: int = config.CHUNK_OVERLAP_CHARS,
) -> list[RawChunk]:
    """Режет text на куски по `size` символов с перекрытием `overlap`."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    text = text.strip()
    if not text:
        return []

    chunks: list[RawChunk] = []
    start = 0
    index = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        piece = text[start:end].strip()
        if piece:
            chunks.append(RawChunk(text=piece, index=index))
            index += 1
        if end == n:
            break
        start = end - overlap
    return chunks
