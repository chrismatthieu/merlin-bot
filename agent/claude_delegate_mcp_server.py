#!/usr/bin/env python3
"""MCP server (stdio) — lets Merlin's brain run Claude Code (`claude -p`) with optional Merlin TTS.

Merlin loads this as an MCP server (see agent/mcp_servers.json). Tool ``delegate_to_claude_code``
spawns ``claude -p`` with ``--mcp-config`` pointing at ``merlin_mcp_server.py`` so the headless
session can call ``merlin_claude_finished`` / ``merlin_speak`` when work is done.

Requires ``claude`` on PATH (Claude Code CLI). Merlin ``main.py`` must be up for voice callbacks.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROTOCOL_VERSION = "2024-11-05"
REPO_ROOT = Path(__file__).resolve().parent.parent
MERLIN_MCP_SCRIPT = REPO_ROOT / "merlin_mcp_server.py"

TOOLS: list[dict] = [
    {
        "name": "delegate_to_claude_code",
        "description": (
            "Run Claude Code in non-interactive mode (-p) on a coding or repo task. "
            "When announce_completion is true (default), Claude is instructed to call the "
            "merlin MCP tool merlin_claude_finished when done so Nova can speak in the room. "
            "Returns Claude's final printed output (and notes if stderr was non-empty)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Clear instructions for Claude Code (what to change, where, constraints).",
                },
                "working_directory": {
                    "type": "string",
                    "description": "Repo or cwd for Claude (defaults to MERLIN_CLAUDE_CODE_CWD or $HOME).",
                },
                "announce_completion": {
                    "type": "boolean",
                    "description": "If true, load Merlin MCP so Claude can call merlin_claude_finished when finished.",
                    "default": True,
                },
            },
            "required": ["task"],
        },
    },
]


def _default_cwd() -> str:
    return (
        os.environ.get("MERLIN_CLAUDE_CODE_CWD", "").strip()
        or str(Path.home())
    )


def _merlin_http_base() -> str:
    return os.environ.get("MERLIN_HTTP_BASE", "http://127.0.0.1:8900").rstrip("/")


def _delegate_timeout() -> float:
    try:
        return float(os.environ.get("MERLIN_CLAUDE_DELEGATE_TIMEOUT_SEC", "3600"))
    except ValueError:
        return 3600.0


def _permission_mode() -> str:
    return os.environ.get("MERLIN_CLAUDE_CODE_PERMISSION_MODE", "acceptEdits").strip() or "acceptEdits"


def _build_mcp_config_path() -> str | None:
    if not MERLIN_MCP_SCRIPT.is_file():
        return None
    cfg = {
        "mcpServers": {
            "merlin": {
                "command": sys.executable,
                "args": [str(MERLIN_MCP_SCRIPT.resolve())],
                "env": {"MERLIN_HTTP_BASE": _merlin_http_base()},
            }
        }
    }
    fd, path = tempfile.mkstemp(prefix="merlin-mcp-", suffix=".json", text=True)
    with os.fdopen(fd, "w") as f:
        json.dump(cfg, f)
    return path


def _run_claude(task: str, cwd: str, announce: bool) -> tuple[int, str]:
    exe = shutil.which("claude")
    if not exe:
        return 127, "claude CLI not found on PATH (install Claude Code / add to PATH)."

    user_task = task.strip()
    if not user_task:
        return 1, "Empty task."

    mcp_path: str | None = None
    if announce:
        mcp_path = _build_mcp_config_path()

    if announce and mcp_path:
        completion_hint = (
            "\n\n---\nNova voice bridge: You have MCP access to server `merlin`. "
            "When you have **fully** completed the task above (all edits and commands done), "
            "you MUST call the tool **merlin_claude_finished** once with a single short sentence "
            "that Nova should speak aloud to the operator (what finished and whether it succeeded). "
            "If that MCP tool is unavailable, finish the task anyway and say so in your printed reply."
        )
        full_prompt = user_task + completion_hint
    elif announce:
        full_prompt = user_task + (
            "\n\n---\nWhen finished, state clearly in your reply that the task is complete "
            "(Merlin voice MCP file was missing, so there is no spoken announcement tool)."
        )
    else:
        full_prompt = user_task

    cmd: list[str] = [
        exe,
        "-p",
        full_prompt,
        "--print",
        "--output-format",
        "text",
        "--permission-mode",
        _permission_mode(),
    ]

    if announce and mcp_path:
        cmd.extend(["--mcp-config", mcp_path])

    env = os.environ.copy()
    env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:" + env.get("PATH", "")

    proc: subprocess.CompletedProcess[str] | None = None
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_delegate_timeout(),
            env=env,
        )
    except subprocess.TimeoutExpired:
        return 124, f"claude timed out after {_delegate_timeout():.0f}s"
    finally:
        if mcp_path:
            try:
                os.unlink(mcp_path)
            except OSError:
                pass

    assert proc is not None
    out_parts = []
    if proc.stdout:
        out_parts.append(proc.stdout.strip())
    if proc.stderr and proc.stderr.strip():
        out_parts.append("[stderr]\n" + proc.stderr.strip())
    body = "\n\n".join(out_parts) if out_parts else "(no output)"
    if proc.returncode != 0:
        body = f"exit {proc.returncode}\n\n" + body
    return proc.returncode, body


def _text_result(text: str, is_error: bool = False) -> dict:
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


def _call_tool(name: str, arguments: dict) -> dict:
    if name != "delegate_to_claude_code":
        return _text_result(f"Unknown tool: {name}", is_error=True)

    task = (arguments.get("task") or "").strip()
    if not task:
        return _text_result("Missing required argument: task", is_error=True)

    cwd = (arguments.get("working_directory") or "").strip() or _default_cwd()
    if not Path(cwd).is_dir():
        return _text_result(f"working_directory is not a directory: {cwd!r}", is_error=True)

    announce = arguments.get("announce_completion", True)
    if not isinstance(announce, bool):
        announce = bool(announce)

    code, text = _run_claude(task, cwd, announce)
    return _text_result(text, is_error=(code != 0))


def _handle(msg: dict) -> dict | None:
    if "method" not in msg:
        return None

    method = msg["method"]
    params = msg.get("params") or {}
    req_id = msg.get("id")

    if req_id is None:
        return None

    if method == "initialize":
        client_pv = params.get("protocolVersion", PROTOCOL_VERSION)
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": client_pv if isinstance(client_pv, str) else PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "merlin-claude-delegate", "version": "1.0.0"},
            },
        }

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}

    if method == "tools/call":
        tname = params.get("name", "")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            arguments = {}
        inner = _call_tool(tname, arguments)
        return {"jsonrpc": "2.0", "id": req_id, "result": inner}

    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        if msg.get("jsonrpc") != "2.0":
            continue
        if "id" not in msg and "method" in msg:
            continue
        if "method" not in msg:
            continue

        try:
            out = _handle(msg)
        except Exception as e:
            req_id = msg.get("id")
            if req_id is not None:
                out = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32603, "message": str(e)},
                }
            else:
                out = None

        if out is not None:
            sys.stdout.write(json.dumps(out) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
