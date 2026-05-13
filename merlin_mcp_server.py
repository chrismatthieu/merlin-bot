#!/usr/bin/env python3
"""MCP server (stdio) — exposes Merlin's local HTTP API to Claude Desktop / Claude Code.

Merlin must be running (`python main.py` or your launcher) so requests reach
`MERLIN_HTTP_BASE` (default http://127.0.0.1:8900).

Protocol: JSON-RPC 2.0, one message per line (same framing as agent/mcp_client.py).
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

from ptz_actions import PTZ_ACTIONS

PROTOCOL_VERSION = "2024-11-05"
DEFAULT_BASE = "http://127.0.0.1:8900"

TOOLS: list[dict] = [
    {
        "name": "merlin_health",
        "description": (
            "GET Merlin /health: uptime, per-module alive/failed, muted state. "
            "Fails if Merlin is not running."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "merlin_scene",
        "description": (
            "GET Merlin /scene: last cached camera scene description and timestamp. "
            "Empty description if vision has not described a frame yet."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "merlin_think",
        "description": (
            "POST Merlin /think: run text through the same brain path as the desk "
            "(intent classification + LLM). Returns the reply string only; does not speak aloud."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "User utterance or instruction for Merlin's brain.",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "merlin_speak",
        "description": (
            "POST Merlin /speak: queue text for TTS on the Merlin machine (event bus speak). "
            "Use for short cues; long text may take a while to synthesize."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "What Merlin should say out loud.",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "merlin_claude_finished",
        "description": (
            "Same as merlin_speak: Nova speaks the given text aloud. "
            "Use this when you (Claude Code) have finished a task that Nova delegated via "
            "delegate_to_claude_code, so the room hears a clear completion line. "
            "One short sentence."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "One short sentence Nova should say (task done, outcome).",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "merlin_ptz",
        "description": (
            "POST Merlin /ptz: move the desk USB camera PTZ (same as voice commands look left/right/up/down, "
            "center, or scan around). Requires uvc-util + camera as configured in Merlin; no-ops if hardware missing."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": sorted(PTZ_ACTIONS),
                    "description": "PTZ preset motion (voice.py / uvc-util).",
                },
            },
            "required": ["action"],
        },
    },
]


def _base_url() -> str:
    return os.environ.get("MERLIN_HTTP_BASE", DEFAULT_BASE).rstrip("/")


def _http(method: str, path: str, body: dict | None = None, timeout: float = 30.0) -> tuple[int, str]:
    url = _base_url() + path
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.getcode(), resp.read().decode()
    except urllib.error.HTTPError as e:
        raw = e.read().decode() if e.fp else ""
        return e.code, raw or str(e)
    except urllib.error.URLError as e:
        return 0, str(e.reason if getattr(e, "reason", None) else e)
    except TimeoutError:
        return 0, "timeout"


def _text_result(text: str, is_error: bool = False) -> dict:
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


def _call_tool(name: str, arguments: dict) -> dict:
    if name == "merlin_health":
        code, body = _http("GET", "/health", timeout=5.0)
        if code == 200:
            return _text_result(body)
        return _text_result(f"HTTP {code}: {body}", is_error=True)

    if name == "merlin_scene":
        code, body = _http("GET", "/scene", timeout=5.0)
        if code == 200:
            return _text_result(body)
        return _text_result(f"HTTP {code}: {body}", is_error=True)

    if name == "merlin_think":
        text = (arguments.get("text") or "").strip()
        if not text:
            return _text_result("Missing required argument: text", is_error=True)
        code, body = _http("POST", "/think", {"text": text}, timeout=300.0)
        if code != 200:
            return _text_result(f"HTTP {code}: {body}", is_error=True)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return _text_result(body, is_error=True)
        reply = payload.get("reply", "")
        return _text_result(reply if reply else "(empty reply)")

    if name in ("merlin_speak", "merlin_claude_finished"):
        text = (arguments.get("text") or "").strip()
        if not text:
            return _text_result("Missing required argument: text", is_error=True)
        code, body = _http("POST", "/speak", {"text": text}, timeout=10.0)
        if code != 200:
            return _text_result(f"HTTP {code}: {body}", is_error=True)
        return _text_result(body)

    if name == "merlin_ptz":
        action = (arguments.get("action") or "").strip()
        if action not in PTZ_ACTIONS:
            return _text_result(
                f"Invalid action {action!r}. Allowed: {', '.join(sorted(PTZ_ACTIONS))}",
                is_error=True,
            )
        code, body = _http("POST", "/ptz", {"action": action}, timeout=60.0)
        if code != 200:
            return _text_result(f"HTTP {code}: {body}", is_error=True)
        return _text_result(body)

    return _text_result(f"Unknown tool: {name}", is_error=True)


def _handle(msg: dict) -> dict | None:
    """Return a JSON-RPC response dict, or None if no reply (notification)."""
    if "method" not in msg:
        return None

    method = msg["method"]
    params = msg.get("params") or {}
    req_id = msg.get("id")

    if req_id is None:
        # JSON-RPC notification — no response body
        return None

    if method == "initialize":
        client_pv = params.get("protocolVersion", PROTOCOL_VERSION)
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": client_pv if isinstance(client_pv, str) else PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "merlin-mcp", "version": "1.0.0"},
            },
        }

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}

    if method == "tools/call":
        name = params.get("name", "")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            arguments = {}
        inner = _call_tool(name, arguments)
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

        # Notifications: no id
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
