"""Qwen3.5 LLM provider через MLX.

Один резидентный инстанс на процесс (§1 CLAUDE.md) — модуль хранит модель и
токенайзер в module-level переменных и грузит их один раз; синтез (и в будущих
милестонах gap-check) обязаны переиспользовать этот инстанс, а не создавать
свой.
"""

from __future__ import annotations

from typing import Any

from .. import config
from . import metrics

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
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> str:
    """`None` — взять значение из config В МОМЕНТ ВЫЗОВА.

    Не `config.X` в дефолте аргумента: Python вычисляет дефолты один раз при
    импорте, поэтому `config.LLM_TEMPERATURE = 0` из eval-прогона молча не
    имел бы никакого эффекта — прогон считался бы детерминированным, оставаясь
    случайным.
    """
    max_tokens = config.LLM_MAX_TOKENS if max_tokens is None else max_tokens
    temperature = config.LLM_TEMPERATURE if temperature is None else temperature
    _ensure_loaded()
    from mlx_lm import generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=temperature)
    with metrics.track("llm.generate"):
        answer = mlx_generate(
            _model,
            _tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        )
    # Токены считаем тем же токенайзером, что и генерировал — mlx_lm не
    # возвращает usage. Лишний encode на вызов, на фоне самой генерации
    # это шум.
    metrics.add_tokens(
        "llm.generate",
        prompt=len(_tokenizer.encode(prompt)),
        completion=len(_tokenizer.encode(answer)),
    )
    return answer
