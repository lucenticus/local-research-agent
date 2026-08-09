"""Юнит-тесты providers/mcp_client.py — MultiServerMCPClient замокан
(офлайн, без реального MCP-сервера/subprocess)."""

from __future__ import annotations

from langchain_core.tools import BaseTool
from pydantic import BaseModel

from src.providers import mcp_client


class _ReadFileArgs(BaseModel):
    path: str


class _FakeAsyncTool(BaseTool):
    """Async-only инструмент — как реально отдаёт langchain-mcp-adapters
    (`_run` не реализован, только `_arun`), с реальной args_schema (как у
    настоящего MCP-инструмента, а не сгенерированной по умолчанию — иначе
    входные kwargs теряются при валидации)."""

    name: str = "read_text_file"
    description: str = "read a file"
    args_schema: type[BaseModel] = _ReadFileArgs

    def _run(self, *args, **kwargs):
        raise NotImplementedError("async-only tool")

    async def _arun(self, *args, **kwargs):
        return [{"type": "text", "text": f"content of {kwargs.get('path')}"}]


def test_get_mcp_tools_wraps_async_only_tools_for_sync_invoke(monkeypatch):
    fake_tool = _FakeAsyncTool()

    class _FakeClient:
        def __init__(self, connections):
            self.connections = connections

        async def get_tools(self, *, server_name=None):
            return [fake_tool]

    monkeypatch.setattr(mcp_client, "MultiServerMCPClient", _FakeClient)

    tools = mcp_client.get_mcp_tools({"fs": {"transport": "stdio", "command": "x"}})
    assert len(tools) == 1
    result = tools[0].invoke({"path": "/tmp/x.md"})
    assert result == [{"type": "text", "text": "content of /tmp/x.md"}]


def test_get_single_tool_returns_matching_tool_by_name(monkeypatch):
    other = _FakeAsyncTool()
    other.name = "other_tool"
    wanted = _FakeAsyncTool()
    monkeypatch.setattr(mcp_client, "get_mcp_tools", lambda connections: [other, wanted])

    assert mcp_client.get_single_tool({}, "read_text_file") is wanted


def test_get_single_tool_returns_none_when_tool_name_not_found(monkeypatch):
    monkeypatch.setattr(mcp_client, "get_mcp_tools", lambda connections: [_FakeAsyncTool()])

    assert mcp_client.get_single_tool({}, "does_not_exist") is None


def test_get_single_tool_returns_none_when_get_mcp_tools_raises(monkeypatch):
    def _raise(connections):
        raise RuntimeError("server not installed")

    monkeypatch.setattr(mcp_client, "get_mcp_tools", _raise)

    assert mcp_client.get_single_tool({}, "read_text_file") is None


def test_content_to_text_extracts_and_joins_text_blocks():
    blocks = [
        {"type": "text", "text": "hello"},
        {"type": "image", "data": "..."},
        {"type": "text", "text": "world"},
    ]
    assert mcp_client.content_to_text(blocks) == "hello\nworld"


def test_content_to_text_passes_through_plain_string():
    assert mcp_client.content_to_text("already a string") == "already a string"


def test_content_to_text_stringifies_unknown_shape():
    assert mcp_client.content_to_text(42) == "42"
