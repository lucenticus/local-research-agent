"""Qwen3.5 LLM provider через MLX.

Один резидентный инстанс на процесс (§1 CLAUDE.md) — модуль хранит модель и
токенайзер в module-level переменных и грузит их один раз; синтез (и в будущих
милестонах gap-check) обязаны переиспользовать этот инстанс, а не создавать
свой.
"""

from __future__ import annotations

from typing import Any

from .. import config

_model: Any = None
_tokenizer: Any = None


def _ensure_loaded() -> None:
    global _model, _tokenizer
    if _model is not None:
        return
    from mlx_lm import load  # lazy import: держим модуль импортируемым без mlx

    _model, _tokenizer = load(config.LLM_HF_REPO)


def build_chat_prompt(system_prompt: str, user_message: str) -> str:
    """Собирает prompt через chat-template токенайзера модели."""
    _ensure_loaded()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    # Подтверждено на реальном токенайзере mlx-community/Qwen3.5-4B-4bit:
    # без enable_thinking=False модель льёт "Thinking Process: ..." в ответ
    # вместо цитируемого ответа по контексту.
    return _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def generate(
    prompt: str,
    *,
    max_tokens: int = config.LLM_MAX_TOKENS,
    temperature: float = config.LLM_TEMPERATURE,
) -> str:
    _ensure_loaded()
    from mlx_lm import generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=temperature)
    return mlx_generate(
        _model,
        _tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        verbose=False,
    )
