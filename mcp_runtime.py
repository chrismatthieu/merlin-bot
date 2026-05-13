"""Shared MCP tool registry: main.py registers tools from Claude extensions; brain.py executes them."""

from __future__ import annotations

import json
import logging
from typing import Any

log = logging.getLogger("merlin.mcp_runtime")

_tools: dict[str, Any] = {}
_TOOL_RESULT_MAX_CHARS = 12_000


def register_mcp_tools(tools: list) -> None:
    """Replace registry with tools from load_mcp_tools() (MCPTool instances)."""
    global _tools
    _tools = {}
    for t in tools:
        name = getattr(t, "name", None)
        if name:
            _tools[name] = t
    log.info("Registered %d MCP tools for brain", len(_tools))


def clear_mcp_tools() -> None:
    _tools.clear()


def has_mcp_tools() -> bool:
    return len(_tools) > 0


def has_claude_code_delegate_tool() -> bool:
    """True if Claude Code delegate MCP server is registered (tool name ends with __delegate_to_claude_code)."""
    return any(str(n).endswith("__delegate_to_claude_code") for n in _tools)


def get_openai_tool_definitions() -> list[dict]:
    """OpenAI/Ollama-compatible tool list."""
    out: list[dict] = []
    for t in _tools.values():
        if hasattr(t, "to_ollama_schema"):
            out.append(t.to_ollama_schema())
    return out


def execute_tool(name: str, arguments: str | dict) -> str:
    tool = _tools.get(name)
    if not tool:
        return f"Error: unknown tool {name!r}"
    if isinstance(arguments, str):
        try:
            kwargs = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as e:
            return f"Error: invalid JSON arguments: {e}"
    else:
        kwargs = dict(arguments) if arguments else {}
    try:
        result = tool.execute(**kwargs)
    except Exception as e:
        log.exception("Tool %s failed", name)
        return f"Error executing {name}: {e}"
    if not isinstance(result, str):
        result = str(result)
    if len(result) > _TOOL_RESULT_MAX_CHARS:
        result = result[: _TOOL_RESULT_MAX_CHARS] + "\n… (truncated)"
    return result
