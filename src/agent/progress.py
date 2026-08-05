"""Общий тип и хелпер для прогресс-колбэков agent/funnel.py и agent/loop.py.

Опциональные — используются веб-интерфейсом (src/web/app.py) для показа
прогресса долгого research(); CLI их не передаёт."""

from __future__ import annotations

from typing import Callable

ProgressCallback = Callable[[str], None]


def emit(on_progress: ProgressCallback | None, message: str) -> None:
    if on_progress is not None:
        on_progress(message)
