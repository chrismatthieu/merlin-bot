"""Merlin v2 — All configuration in one place."""

import os
import re
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv
from requests.auth import HTTPDigestAuth

# Load .env: optional parent workspace, then this repo (repo wins on duplicate keys)
_repo_root = Path(__file__).parent
load_dotenv(_repo_root.parent / ".env")
load_dotenv(_repo_root / ".env", override=True)

from ptz_actions import PTZ_ACTIONS


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


def llm_openai_request_extras() -> dict:
    """Extra fields for OpenAI-compatible `POST .../v1/chat/completions`.

    Ollama uses `reasoning_effort` so chain-of-thought stays out of spoken `content`.
    Set `MERLIN_LLM_REASONING_EFFORT` to `none|low|medium|high`; when unset and the
    URL targets Ollama (port 11434), default is "none" for voice. Other servers
    receive no extra keys.
    """
    u = (LLM_URL or "").lower()
    ollama = re.search(r":11434/", u) is not None
    raw = os.getenv("MERLIN_LLM_REASONING_EFFORT", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        raw = "none"
    if raw in ("none", "low", "medium", "high"):
        return {"reasoning_effort": raw}
    if not raw and ollama:
        return {"reasoning_effort": "none"}
    return {}

# Legacy Ollama (kept for fallback)
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "gemma4:e4b"

# ── Audio Pipeline ───────────────────────────────────────────────
AUDIO_SOURCE = os.getenv("MERLIN_AUDIO_SOURCE", "rtsp")  # "rtsp" (Amcrest camera mic) or "usb" (PIXY — only if on same machine)
MIC_SAMPLE_RATE = 16000
VAD_THRESHOLD = 0.5
UTTERANCE_SILENCE_TIMEOUT = 1.5
ECHO_SUPPRESSION_PADDING = 0.5   # USB path is much shorter than RTSP (was 1.5s)
try:
    MIN_UTTERANCE_SEC = float(os.getenv("MERLIN_MIN_UTTERANCE_SEC", "0.28"))
except ValueError:
    MIN_UTTERANCE_SEC = 0.28
# PCM length required before Whisper runs (was 1.0s hard-coded — blocked short "wake up" / "Nova").
MIN_UTTERANCE_BYTES = max(int(MIC_SAMPLE_RATE * 2 * MIN_UTTERANCE_SEC), 4000)
try:
    MIN_UTTERANCE_SEC_MUTED = float(os.getenv("MERLIN_MIN_UTTERANCE_SEC_MUTED", "0.12"))
except ValueError:
    MIN_UTTERANCE_SEC_MUTED = 0.12
# While muted, accept shorter clips so "wake up" / name reach Whisper (still has a small floor).
MIN_UTTERANCE_BYTES_MUTED = max(int(MIC_SAMPLE_RATE * 2 * MIN_UTTERANCE_SEC_MUTED), 1600)

# ── TTS ──────────────────────────────────────────────────────────
KOKORO_VOICE = os.getenv("MERLIN_VOICE", "am_fenrir")  # nerdy sage in a security camera body
NONVERBAL_ENABLED = os.getenv("MERLIN_NONVERBAL", "1").strip().lower() not in {"0", "false", "no", "off"}
# Default off: macOS say / Kokoro runs seconds of echo suppression on the USB mic and eats the next phrase.
VERBAL_UNMUTE_ACK = os.getenv("MERLIN_VERBAL_UNMUTE_ACK", "0").strip().lower() not in {"0", "false", "no", "off"}


def _detect_merlin_avfoundation_index():
    """Find AVFoundation video index for the EMEET PIXY (Merlin) USB camera.

    Index 0 on macOS is usually the built-in FaceTime camera; we must not default
    to it when MERLIN_CAMERA_INDEX is unset.

    Only lines under **AVFoundation video devices** are considered so we do not
    pick an EMEET USB *audio* endpoint (same composite device) or another mic line.
    When multiple EMEET video devices exist, prefer a line whose name contains **PIXY**.
    """
    try:
        r = subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    blob = (r.stderr or "") + (r.stdout or "")

    def _video_device_lines() -> list[str]:
        lines = blob.splitlines()
        out: list[str] = []
        in_video = False
        for line in lines:
            if "AVFoundation video devices" in line:
                in_video = True
                continue
            if "AVFoundation audio devices" in line:
                in_video = False
                continue
            if in_video:
                out.append(line)
        return out

    video_lines = _video_device_lines()
    # Older ffmpeg / odd builds: no section headers — fall back to full log (legacy behavior).
    scan_lines = video_lines if video_lines else blob.splitlines()

    matches = [ln for ln in scan_lines if re.search(r"emeet|pixy", ln, re.I)]
    if not matches:
        return None
    pixy_named = [ln for ln in matches if re.search(r"pixy", ln, re.I)]
    ordered = pixy_named if pixy_named else matches
    for line in ordered:
        m = re.search(r"\[(\d+)\]", line)
        if m:
            return int(m.group(1))
    return None


# ── USB Camera (EMEET PIXY) ─────────────────────────────────────
_explicit_cam = (os.environ.get("MERLIN_CAMERA_INDEX") or "").strip()
if _explicit_cam != "":
    USB_CAMERA_INDEX = int(_explicit_cam)
elif sys.platform == "darwin" and AUDIO_SOURCE == "usb":
    _auto = _detect_merlin_avfoundation_index()
    if _auto is None:
        raise RuntimeError(
            "MERLIN_AUDIO_SOURCE=usb but the Merlin camera (EMEET PIXY) was not found "
            "via ffmpeg AVFoundation. Plug in the Merlin USB camera, free the device from "
            "other apps, or set MERLIN_CAMERA_INDEX to its video index explicitly "
            "(do not use 0 unless that index is really the PIXY — 0 is usually FaceTime)."
        )
    USB_CAMERA_INDEX = _auto
else:
    USB_CAMERA_INDEX = int(os.getenv("MERLIN_CAMERA_INDEX", "0"))

USB_CAMERA_WIDTH = 1920
USB_CAMERA_HEIGHT = 1080
USB_CAMERA_FPS = 30

# USB tracker control API (tracker_usb.py): pause face PTZ during scripted gestures
try:
    TRACKER_CONTROL_PORT = int(os.getenv("MERLIN_TRACKER_CONTROL_PORT", "8903"))
except ValueError:
    TRACKER_CONTROL_PORT = 8903
TRACKER_CONTROL_URL = os.getenv(
    "MERLIN_TRACKER_CONTROL_URL",
    f"http://127.0.0.1:{TRACKER_CONTROL_PORT}",
)

# ── Vision ───────────────────────────────────────────────────────
VISION_MODEL = os.getenv("MERLIN_VISION_MODEL", "mlx-community/nanoLLaVA-1.5-4bit")
VISION_INTERVAL_DEFAULT = 5
VISION_INTERVAL_IDLE = 15
VISION_INTERVAL_ACTIVE = 3
VISION_INTERVAL_MUTED = 30
VISION_PROMPT = "Briefly describe what you see at this desk. One sentence."

# ── Agent / MCP ──────────────────────────────────────────────────
# When True, main.py launches Claude extension MCP servers at startup (same as agent CLI).
AUTOSTART_MCP = os.getenv("MERLIN_AUTOSTART_MCP", "0").strip().lower() not in {"0", "false", "no", "off"}
# When True, brain.py may call MCP tools (Notes, iMessage, Mac automation) via OpenAI-style tool messages.
BRAIN_MCP = os.getenv("MERLIN_BRAIN_MCP", "0").strip().lower() not in {"0", "false", "no", "off"}
try:
    BRAIN_MCP_MAX_ROUNDS = max(1, min(20, int(os.getenv("MERLIN_BRAIN_MCP_MAX_ROUNDS", "8"))))
except ValueError:
    BRAIN_MCP_MAX_ROUNDS = 8
try:
    BRAIN_MCP_LLM_TIMEOUT = max(30, int(os.getenv("MERLIN_BRAIN_MCP_LLM_TIMEOUT", "180")))
except ValueError:
    BRAIN_MCP_LLM_TIMEOUT = 180

# ── iMessage watcher (local chat.db poll) ─────────────────────────
try:
    IMESSAGE_POLL_INTERVAL = max(0, int(os.getenv("MERLIN_IMESSAGE_POLL_INTERVAL", "0")))
except ValueError:
    IMESSAGE_POLL_INTERVAL = 0
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
UNMUTE_WORDS = ["start listening", "unmute", "wake up", "wakeup"]
NEVERMIND_WORDS = ["nevermind", "never mind"]
# Ignore wake/unmute phrase matches for this long after entering sleep. Otherwise a
# second STT segment, motor noise, or Whisper bias from SLEEP_WAKE_WHISPER_PROMPT
# can fire a false "Hey Nova" / "wake up" within ~1s and undo mute immediately.
try:
    MUTE_UNMUTE_GUARD_SEC = float(os.getenv("MERLIN_MUTE_UNMUTE_GUARD_SEC", "3"))
except ValueError:
    MUTE_UNMUTE_GUARD_SEC = 3.0

# Whisper while "sleeping": room tone + short clips often get classified as no-speech (default 0.6).
try:
    WHISPER_NO_SPEECH_THRESHOLD_SLEEP = float(
        os.getenv("MERLIN_WHISPER_NO_SPEECH_SLEEP", "0.22")
    )
except ValueError:
    WHISPER_NO_SPEECH_THRESHOLD_SLEEP = 0.22
_n = BOT_NAME.strip()
_novo = " Novo. Hey Novo." if _n.lower() == "nova" else ""
SLEEP_WAKE_WHISPER_PROMPT = os.getenv(
    "MERLIN_WHISPER_SLEEP_WAKE_PROMPT",
    f"Wake up. Unmute. Start listening. {_n}. Hey {_n}. Hi {_n}. {_n}, wake up.{_novo}",
)


def normalize_heard_text(s: str) -> str:
    """Normalize STT text so phrase checks survive punctuation and hyphenation (wake-up → wake up)."""
    if not s:
        return ""
    t = s.lower().strip()
    t = re.sub(r"[\s\-_,;:!?.]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def heard_contains_phrase(text: str, phrase: str) -> bool:
    """True if normalized *text* contains normalized *phrase*."""
    t = normalize_heard_text(text)
    p = normalize_heard_text(phrase)
    return bool(p) and p in t


def is_mute_command(text: str) -> bool:
    """True for sleep/mute intents; avoids matching *mute* inside *unmute*."""
    t = normalize_heard_text(text)
    if heard_contains_phrase(t, "stop listening"):
        return True
    if heard_contains_phrase(t, "go to sleep") or heard_contains_phrase(t, "goto sleep"):
        return True
    # Standalone word "mute" — \bmute\b matches "mute" but not "unmute"
    return bool(re.search(r"\bmute\b", t))


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
