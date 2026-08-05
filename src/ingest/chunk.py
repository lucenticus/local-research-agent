"""Фикс-чанкинг с перекрытием по символам, теперь с учётом границ секций.

`chunk_text` — наивная база (Milestone 0). `chunk_sections` (Milestone 1)
режет каждую секцию отдельно, так что чанк никогда не смешивает текст двух
секций (например, конец Method и начало Results).
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import config
from .extract import Section


@dataclass
class RawChunk:
    text: str
    index: int
    section: str = ""


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


def chunk_sections(
    sections: list[Section],
    size: int = config.CHUNK_SIZE_CHARS,
    overlap: int = config.CHUNK_OVERLAP_CHARS,
) -> list[RawChunk]:
    """Чанкинг по секциям — чанк не пересекает границу секции."""
    chunks: list[RawChunk] = []
    index = 0
    for section in sections:
        for raw in chunk_text(section.text, size=size, overlap=overlap):
            chunks.append(RawChunk(text=raw.text, index=index, section=section.name))
            index += 1
    return chunks
