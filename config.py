"""Merlin v2 — All configuration in one place."""

import os
import re
from pathlib import Path
from dotenv import load_dotenv
from requests.auth import HTTPDigestAuth

# Load .env from RBOS root
load_dotenv(Path(__file__).parent.parent / ".env")


def _load_soul(path: Path) -> dict[str, str]:
    """Load simple key/value config from soul.md."""
    defaults = {
        "name": "Merlin",
        "operator": "User",
        "character": "A playful, happy, curious desk companion.",
        "persona": "Warm and observant; concise, grounded, and supportive.",
        "personality": "Playful, happy, curious.",
    }
    if not path.exists():
        return defaults

    values = defaults.copy()
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z_]+)\s*:\s*(.+)$", line)
        if not match:
            continue
        key = match.group(1).strip().lower()
        value = match.group(2).strip()
        if key in values and value:
            values[key] = value
    return values


def _build_wake_words(name: str) -> list[str]:
    base = name.strip().lower()
    wake_words = [base, f"hey {base}", f"hi {base}", f"ok {base}"]
    return list(dict.fromkeys(wake_words))

# ── Network ──────────────────────────────────────────────────────
PI_HOST = os.getenv("MERLIN_PI_HOST", "100.87.156.70")
GO2RTC_RTSP = f"rtsp://{PI_HOST}:8554/merlin"
GO2RTC_API = f"http://{PI_HOST}:1984"
GO2RTC_STREAM = "merlin"
TRACKER_LISTEN_PORT = 8900
BRAIN_EVENT_URL = os.getenv("MERLIN_BRAIN_URL", f"http://localhost:{TRACKER_LISTEN_PORT}/event")

# ── Camera (direct RTSP) ────────────────────────────────────────
CAMERA_IP = os.getenv("MERLIN_CAMERA_IP", "192.168.1.26")
CAMERA_USER = os.getenv("MERLIN_CAMERA_USER", "admin")
CAMERA_PASS = os.getenv("MERLIN_CAMERA_PASS", "")
CAMERA_RTSP_SUB = (
    f"rtsp://{CAMERA_USER}:{CAMERA_PASS}@{CAMERA_IP}:554"
    f"/cam/realmonitor?channel=1&subtype=1"
)
# Audio input reads directly from camera — NOT through go2rtc.
# go2rtc's RTSP stream drops when speaker audio is pushed to it.
# Camera's own RTSP is independent and stays up during playback.
CAMERA_RTSP_AUDIO = (
    f"rtsp://{CAMERA_USER}:{CAMERA_PASS}@{CAMERA_IP}:554"
    f"/cam/realmonitor?channel=1&subtype=0"
)
CAMERA_RTSP_MAIN = CAMERA_RTSP_AUDIO  # alias — subtype=0 is main stream
CAMERA_AUTH = HTTPDigestAuth(CAMERA_USER, CAMERA_PASS)
CAMERA_PTZ_BASE = f"http://{CAMERA_IP}/cgi-bin/ptz.cgi"
CAMERA_ONVIF_PTZ = f"http://{CAMERA_IP}/onvif/ptz_service"

# ── LLM — LM Studio (OpenAI-compatible API) ─────────────────────
LLM_URL = os.getenv("MERLIN_LLM_URL", "http://localhost:1234/v1/chat/completions")
LLM_MODEL = os.getenv("MERLIN_MODEL", "qwen/qwen3-vl-4b")

# Legacy Ollama (kept for fallback)
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "gemma4:e4b"

# ── Audio Pipeline ───────────────────────────────────────────────
AUDIO_SOURCE = os.getenv("MERLIN_AUDIO_SOURCE", "rtsp")  # "rtsp" (Amcrest camera mic) or "usb" (PIXY — only if on same machine)
MIC_SAMPLE_RATE = 16000
VAD_THRESHOLD = 0.5
UTTERANCE_SILENCE_TIMEOUT = 1.5
ECHO_SUPPRESSION_PADDING = 0.5   # USB path is much shorter than RTSP (was 1.5s)

# ── TTS ──────────────────────────────────────────────────────────
KOKORO_VOICE = os.getenv("MERLIN_VOICE", "am_fenrir")  # nerdy sage in a security camera body
NONVERBAL_ENABLED = os.getenv("MERLIN_NONVERBAL", "1").strip().lower() not in {"0", "false", "no", "off"}

# ── USB Camera (EMEET PIXY) ─────────────────────────────────────
USB_CAMERA_INDEX = int(os.getenv("MERLIN_CAMERA_INDEX", "0"))
USB_CAMERA_WIDTH = 1920
USB_CAMERA_HEIGHT = 1080
USB_CAMERA_FPS = 30

# ── Vision ───────────────────────────────────────────────────────
VISION_MODEL = os.getenv("MERLIN_VISION_MODEL", "mlx-community/nanoLLaVA-1.5-4bit")
VISION_INTERVAL_DEFAULT = 5
VISION_INTERVAL_IDLE = 15
VISION_INTERVAL_ACTIVE = 3
VISION_INTERVAL_MUTED = 30
VISION_PROMPT = "Briefly describe what you see at this desk. One sentence."

# ── Agent / MCP ──────────────────────────────────────────────────
# When True, main.py launches Claude extension MCP servers at startup (same as agent CLI).
AUTOSTART_MCP = os.getenv("MERLIN_AUTOSTART_MCP", "1").strip().lower() not in {"0", "false", "no", "off"}
# When True, brain.py may call MCP tools (Notes, iMessage, Mac automation) via OpenAI-style tool messages.
BRAIN_MCP = os.getenv("MERLIN_BRAIN_MCP", "1").strip().lower() not in {"0", "false", "no", "off"}
try:
    BRAIN_MCP_MAX_ROUNDS = max(1, min(20, int(os.getenv("MERLIN_BRAIN_MCP_MAX_ROUNDS", "8"))))
except ValueError:
    BRAIN_MCP_MAX_ROUNDS = 8

# ── iMessage watcher (local chat.db poll) ─────────────────────────
try:
    IMESSAGE_POLL_INTERVAL = max(0, int(os.getenv("MERLIN_IMESSAGE_POLL_INTERVAL", "15")))
except ValueError:
    IMESSAGE_POLL_INTERVAL = 15
IMESSAGE_CHAT_DB = Path(
    os.getenv("MERLIN_IMESSAGE_CHAT_DB", str(Path.home() / "Library/Messages/chat.db"))
)
try:
    IMESSAGE_MIN_TEXT_LEN = max(1, int(os.getenv("MERLIN_IMESSAGE_MIN_TEXT_LEN", "1")))
except ValueError:
    IMESSAGE_MIN_TEXT_LEN = 1

# ── Conversation ─────────────────────────────────────────────────
SOUL_PATH = Path(__file__).parent / "soul.md"
SOUL = _load_soul(SOUL_PATH)
BOT_NAME = SOUL["name"]
BOT_OPERATOR = SOUL["operator"]
BOT_CHARACTER = SOUL["character"]
BOT_PERSONA = SOUL["persona"]
BOT_PERSONALITY = SOUL["personality"]
WAKE_WORDS = _build_wake_words(BOT_NAME)
CONVERSATION_WINDOW = 60  # seconds after Merlin speaks before requiring wake word again
CONVERSATION_HISTORY_SIZE = 10
MUTE_WORDS = ["stop listening", "mute", "go to sleep"]
UNMUTE_WORDS = ["start listening", "unmute", "wake up"]
NEVERMIND_WORDS = ["nevermind", "never mind"]

# ── RBOS ─────────────────────────────────────────────────────────
RBOS_ROOT = Path("/Users/ezradrake/Documents/RBOS")
STATE_PATH = RBOS_ROOT / "core" / "STATE.md"
BRIEFING_DIR = RBOS_ROOT / "merlin" / "briefing"
BRIEFING_POLL_INTERVAL = 900  # 15 minutes

# ── Paths ────────────────────────────────────────────────────────
LOG_FILE = Path("/tmp/merlin-v2.log")
FRAME_PATH = Path("/tmp/merlin_frame.jpg")
STATE_PERSIST_PATH = Path("/tmp/merlin-state.json")
SOUNDS_DIR = Path(__file__).parent / "sounds"

# Ensure homebrew binaries are on PATH
os.environ["PATH"] = "/opt/homebrew/bin:/usr/local/bin:" + os.environ.get("PATH", "")
