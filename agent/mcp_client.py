"""MCP Client — connects to MCP servers over stdio using JSON-RPC 2.0."""

import json
import subprocess
import sys
import threading
import queue


class MCPClient:
    """Connects to an MCP server via stdio (JSON-RPC 2.0, newline-delimited)."""

    def __init__(
        self,
        name: str,
        command: str,
        args: list[str],
        env: dict = None,
        tool_call_timeout: float = 30.0,
    ):
        self.name = name
        self.command = command
        self.args = args
        self.env = env
        self.tool_call_timeout = float(tool_call_timeout)
        self.process = None
        self.request_id = 0
        self.responses = {}
        self.pending = {}  # id -> threading.Event
        self._reader_thread = None
        self._lock = threading.Lock()

    def start(self):
        """Launch the MCP server subprocess."""
        import os
        full_env = os.environ.copy()
        # Ensure homebrew binaries are on PATH (node, uv, etc.)
        full_env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:" + full_env.get("PATH", "")
        if self.env:
            full_env.update(self.env)

        self.process = subprocess.Popen(
            [self.command] + self.args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=full_env,
        )

        # Background thread reads responses from stdout
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

        # Initialize the MCP connection
        result = self._request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "rbos-agent", "version": "1.0.0"},
            },
            timeout=30.0,
        )

        # Send initialized notification
        self._notify("notifications/initialized", {})
        return result

    def list_tools(self) -> list[dict]:
        """Discover available tools from the server."""
        result = self._request("tools/list", {}, timeout=30.0)
        return result.get("tools", []) if result else []

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool and return the result as a string."""
        result = self._request(
            "tools/call",
            {
                "name": tool_name,
                "arguments": arguments,
            },
            timeout=self.tool_call_timeout,
        )
        if result is None:
            return "Error: no response from MCP server"

        # MCP returns content as an array of content blocks
        content = result.get("content", [])
        parts = []
        for block in content:
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif block.get("type") == "image":
                parts.append(f"[image: {block.get('mimeType', 'unknown')}]")
            else:
                parts.append(json.dumps(block))

        if result.get("isError"):
            return "Error: " + "\n".join(parts)
        return "\n".join(parts) if parts else "(empty response)"

    def stop(self):
        """Shut down the MCP server."""
        if self.process and self.process.poll() is None:
            self.process.stdin.close()
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def _next_id(self) -> int:
        with self._lock:
            self.request_id += 1
            return self.request_id

    def _request(self, method: str, params: dict, timeout: float = 30.0) -> dict:
        """Send a JSON-RPC request and wait for the response."""
        req_id = self._next_id()
        event = threading.Event()

        with self._lock:
            self.pending[req_id] = event

        msg = json.dumps({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }) + "\n"

        try:
            self.process.stdin.write(msg.encode())
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            print(f"[mcp:{self.name}] Write error: {e}", file=sys.stderr)
            return None

        # Wait for response
        if event.wait(timeout=timeout):
            with self._lock:
                return self.responses.pop(req_id, None)
        else:
            print(f"[mcp:{self.name}] Timeout waiting for response to {method}", file=sys.stderr)
            with self._lock:
                self.pending.pop(req_id, None)
            return None

    def _notify(self, method: str, params: dict):
        """Send a JSON-RPC notification (no response expected)."""
        msg = json.dumps({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }) + "\n"

        try:
            self.process.stdin.write(msg.encode())
            self.process.stdin.flush()
        except (BrokenPipeError, OSError):
            pass

    def _read_loop(self):
        """Background thread: read JSON-RPC responses from stdout."""
        buffer = b""
        while self.process and self.process.poll() is None:
            try:
                chunk = self.process.stdout.read(1)
                if not chunk:
                    break
                buffer += chunk

                # Try to parse complete JSON messages
                if chunk == b"\n":
                    line = buffer.strip()
                    buffer = b""
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                        req_id = msg.get("id")
                        if req_id is not None:
                            with self._lock:
                                self.responses[req_id] = msg.get("result", msg.get("error"))
                                event = self.pending.pop(req_id, None)
                            if event:
                                event.set()
                        # Notifications from server (no id) — ignore for now
                    except json.JSONDecodeError:
                        pass
            except Exception:
                break
