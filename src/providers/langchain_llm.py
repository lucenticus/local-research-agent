"""LangChain `BaseChatModel` поверх `providers/llm.py` — тонкая обёртка для
экспериментов с LangChain/LangGraph, не отдельная реализация. Резидентная
модель по-прежнему одна на процесс (§1 CLAUDE.md) — `llm.generate`/
`llm.build_chat_prompt` продолжают жить в `providers/llm.py`, этот класс
только переводит их в интерфейс `BaseChatModel`, чтобы MLX-LLM можно было
подставлять в LangChain/LangGraph код (Runnable-цепочки, `.bind_tools`
и т.п.) без переписывания синтеза/gap-check.
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from .. import config
from . import llm as llm_provider


def _messages_to_system_and_user(messages: list[BaseMessage]) -> tuple[str, str]:
    """MLX chat-template (`llm.build_chat_prompt`) ждёт ровно system+user —
    сворачиваем LangChain-историю сообщений в эту пару: все system-сообщения
    объединяются, все остальные (human/ai) идут в один user-блок в исходном
    порядке. Этого достаточно для однократных вызовов (перевод запроса,
    gap-check) — многоходовый чат этим провайдером сейчас не используется."""
    system_parts = [m.content for m in messages if m.type == "system"]
    other_parts = [f"{m.type}: {m.content}" if m.type != "human" else m.content for m in messages if m.type != "system"]
    system_prompt = "\n".join(str(p) for p in system_parts) or "You are a helpful assistant."
    user_message = "\n".join(str(p) for p in other_parts)
    return system_prompt, user_message


class ChatMLX(BaseChatModel):
    """`Qwen3.5-4B` через `mlx_lm`, обёрнутый в интерфейс LangChain `BaseChatModel`."""

    max_tokens: int = config.LLM_MAX_TOKENS
    temperature: float = config.LLM_TEMPERATURE

    @property
    def _llm_type(self) -> str:
        return "mlx-qwen3.5"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        system_prompt, user_message = _messages_to_system_and_user(messages)
        prompt = llm_provider.build_chat_prompt(system_prompt, user_message)
        text = llm_provider.generate(prompt, max_tokens=self.max_tokens, temperature=self.temperature)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])
