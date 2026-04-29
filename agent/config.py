"""Agent configuration — constants and defaults."""

import os
from pathlib import Path

# Ollama
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
DEFAULT_MODEL = os.environ.get("RBOS_AGENT_MODEL", "gemma4:26b")

# Paths
RBOS_ROOT = Path(os.environ.get("RBOS_ROOT", Path(__file__).parent.parent.parent)).resolve()

# Agent limits
MAX_TOOL_ROUNDS = 10
REQUEST_TIMEOUT = 120  # seconds — Gemma 27B can take 30-60s for complex reasoning
MAX_FILE_CHARS = 10000  # truncation limit for read_file tool

# System prompt
SYSTEM_PROMPT = """You are an RBOS agent running on Chris's Mac. You have tools to:
- Read and write files in the RBOS filesystem
- Read and send iMessages
- Read and write Apple Notes
- Run AppleScript via osascript to control any Mac app (Calendar, Reminders, Finder, Safari, etc.)

Use your tools to answer questions — don't guess. When you have an osascript tool, use it freely to access Calendar, Reminders, and other Apple apps. Be concise and direct."""
