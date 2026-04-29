"""MCP Bridge — discovers tools from MCP servers and wraps them as BaseTools."""

import json
import os
from pathlib import Path
from .base import BaseTool
from ..mcp_client import MCPClient

# Default config path
MCP_CONFIG_PATH = Path(__file__).parent.parent / "mcp_servers.json"

# Optional: force all Claude extension lookups under this directory (contains extension_id folders).
MERLIN_MCP_EXTENSIONS_ROOT = "MERLIN_MCP_EXTENSIONS_ROOT"


def _expand_path_tokens(value: str) -> str:
    """Expand ~ and env vars in command/arg strings."""
    if not isinstance(value, str):
        return value
    return os.path.expanduser(os.path.expandvars(value))


def _expand_env_map(env: dict | None) -> dict | None:
    """Expand ~ and env vars for MCP env values."""
    if not env:
        return env
    expanded = {}
    for key, value in env.items():
        expanded[key] = _expand_path_tokens(value) if isinstance(value, str) else value
    return expanded


def _claude_extension_search_roots() -> list[Path]:
    """Directories that contain per-extension folders (each with server/index.js)."""
    home = Path.home()
    roots: list[Path] = []
    override = os.environ.get(MERLIN_MCP_EXTENSIONS_ROOT, "").strip()
    if override:
        p = Path(_expand_path_tokens(override))
        if p.is_dir():
            roots.append(p.resolve())
        return roots

    candidates = [
        home / "Library/Application Support/Claude/Claude Extensions",
        home / "Library/Application Support/Claude Desktop/Claude Extensions",
        home / "Library/Application Support/Anthropic/Claude/Claude Extensions",
    ]
    for c in candidates:
        if c.is_dir():
            roots.append(c.resolve())
    return roots


def _find_claude_extension_script(extension_id: str) -> Path | None:
    """Locate server/index.js for a Claude Desktop-style extension folder name."""
    if not extension_id or not extension_id.strip():
        return None
    eid = extension_id.strip()
    for root in _claude_extension_search_roots():
        direct = root / eid / "server" / "index.js"
        if direct.is_file():
            return direct
    # Broader search under Claude-ish trees only (avoid scanning all of Application Support)
    app_support = Path.home() / "Library/Application Support"
    for tree in ("Claude", "Claude Desktop", "Anthropic"):
        base = app_support / tree
        if not base.is_dir():
            continue
        try:
            for match in base.rglob(f"{eid}/server/index.js"):
                if match.is_file():
                    return match
        except OSError:
            pass
    return None


def _env_script_override(server_name: str) -> str | None:
    """MERLIN_MCP_<SERVER>_SCRIPT=/path/to/index.js"""
    key = f"MERLIN_MCP_{server_name.upper().replace('-', '_')}_SCRIPT"
    raw = os.environ.get(key, "").strip()
    if not raw:
        return None
    return _expand_path_tokens(raw)


def _resolve_mcp_args(
    server_name: str,
    server_config: dict,
    verbose: bool,
) -> list[str] | None:
    """Build node (or other) argv: [script, ...extra]. Returns None if script missing."""
    extra = [
        _expand_path_tokens(a) if isinstance(a, str) else a
        for a in server_config.get("args", [])
        if not (isinstance(a, str) and a.strip() == "")
    ]
    extension_id = (server_config.get("extension_id") or "").strip()

    env_path = _env_script_override(server_name)
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return [str(p.resolve())] + extra
        if verbose:
            print(f"[mcp] {server_name}: MERLIN_MCP_*_SCRIPT set but not a file: {p}")

    # Explicit args: first element is usually the server entry script
    if extra:
        first = extra[0]
        if isinstance(first, str) and (first.endswith(".js") or first.endswith(".mjs")):
            p = Path(first)
            if p.is_file():
                return [str(p.resolve())] + extra[1:]
            if verbose:
                print(f"[mcp] {server_name}: configured script missing: {p}")

    if extension_id:
        found = _find_claude_extension_script(extension_id)
        if found:
            return [str(found)] + extra
        if verbose:
            print(
                f"[mcp] {server_name}: extension {extension_id!r} not found under "
                f"Claude Extensions (set {MERLIN_MCP_EXTENSIONS_ROOT} or MERLIN_MCP_{server_name.upper()}_SCRIPT)"
            )

    if extra:
        first = extra[0]
        if isinstance(first, str) and first.endswith((".js", ".mjs")) and not Path(first).is_file():
            return None
        return extra
    return None


class MCPTool(BaseTool):
    """Wraps a single MCP tool as a BaseTool for the agent kernel."""

    def __init__(self, client: MCPClient, tool_def: dict):
        self.client = client
        self.tool_def = tool_def
        # Prefix tool name with server name to avoid collisions
        self.name = f"{client.name}__{tool_def['name']}"
        self.description = tool_def.get("description", "")
        self._params = tool_def.get("inputSchema", {
            "type": "object",
            "properties": {},
            "required": [],
        })

    def parameters(self) -> dict:
        return self._params

    def execute(self, **kwargs) -> str:
        return self.client.call_tool(self.tool_def["name"], kwargs)


def load_mcp_tools(config_path: str = None, verbose: bool = True) -> tuple[list[BaseTool], list[MCPClient]]:
    """Load MCP servers from config and return all discovered tools.

    Returns (tools, clients) — caller should stop() clients on shutdown.
    """
    config_file = Path(config_path) if config_path else MCP_CONFIG_PATH
    if not config_file.exists():
        if verbose:
            print(f"[mcp] No config at {config_file} — skipping MCP tools")
        return [], []

    with open(config_file) as f:
        config = json.load(f)

    servers = config.get("servers", {})
    all_tools = []
    clients = []

    for name, server_config in servers.items():
        if not server_config.get("enabled", True):
            if verbose:
                print(f"[mcp] {name}: disabled, skipping")
            continue

        command = _expand_path_tokens(server_config.get("command", ""))
        env = _expand_env_map(server_config.get("env"))
        args = _resolve_mcp_args(name, server_config, verbose=verbose)
        if not args:
            continue

        if verbose:
            print(f"[mcp] {name}: starting... ({args[0]})")

        client = MCPClient(name=name, command=command, args=args, env=env)
        try:
            client.start()
            tools = client.list_tools()
            if verbose:
                print(f"[mcp] {name}: {len(tools)} tools discovered")
                for t in tools:
                    print(f"  - {t['name']}: {t.get('description', '')[:60]}")

            for tool_def in tools:
                all_tools.append(MCPTool(client, tool_def))
            clients.append(client)

        except Exception as e:
            print(f"[mcp] {name}: failed to start — {e}")
            client.stop()

    return all_tools, clients
