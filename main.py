"""Merlin v2 — Orchestrator: starts modules, supervises, serves HTTP."""

import atexit
import http.server
import json
import logging
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime

from event_bus import EventBus
from audio_pipeline import AudioPipeline
from voice import Voice
from brain import Brain
from vision import Vision
import config
import mcp_runtime
from imessage_watcher import start_imessage_watcher_if_enabled

# ── Logging ──────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.LOG_FILE, mode="a"),
    ],
)
log = logging.getLogger("merlin.main")

# MCP subprocesses (Claude extension servers) — started at boot when AUTOSTART_MCP is on
_MCP_CLIENTS: list = []


def _stop_mcp_clients() -> None:
    mcp_runtime.clear_mcp_tools()
    for client in _MCP_CLIENTS:
        try:
            client.stop()
        except Exception:
            log.exception("Error stopping MCP client %s", getattr(client, "name", "?"))
    _MCP_CLIENTS.clear()


def _maybe_autostart_mcp() -> None:
    """Pre-launch MCP servers so Notes / iMessage / osascript tools are warm for agent/main.py."""
    if not config.AUTOSTART_MCP:
        log.info("MCP autostart disabled (MERLIN_AUTOSTART_MCP)")
        return
    try:
        from agent.tools.mcp_bridge import load_mcp_tools
    except ImportError:
        log.warning("MCP autostart skipped — could not import agent.tools.mcp_bridge")
        return
    try:
        mcp_tools, clients = load_mcp_tools(verbose=False)
        _MCP_CLIENTS.extend(clients)
        if mcp_tools:
            mcp_runtime.register_mcp_tools(mcp_tools)
        if clients:
            log.info("MCP servers started: %s", ", ".join(c.name for c in clients))
        else:
            log.info(
                "MCP autostart: no servers connected "
                "(install Claude extensions or set MERLIN_MCP_*_SCRIPT / MERLIN_MCP_EXTENSIONS_ROOT)"
            )
    except Exception:
        log.exception("MCP autostart failed")

# ── Module Registry ──────────────────────────────────────────────


class ModuleInfo:
    def __init__(self, name, instance):
        self.name = name
        self.instance = instance
        self.restarts = 0
        self.last_restart = None
        self.restart_times = []  # timestamps of recent restarts
        self.failed = False


class Orchestrator:
    def __init__(self):
        self.bus = EventBus()
        self.start_time = time.time()
        self.modules: list[ModuleInfo] = []
        self._running = True
        self._muted = False
        self.bus.on("mute_toggled", self._on_mute)

    def _on_mute(self, muted=False, **kw):
        self._muted = muted
        # USB tracker: POST must not block the event bus (speech/audio often emit from audio thread).
        def _post_tracker() -> None:
            try:
                body = json.dumps({"muted": bool(muted)}).encode()
                req = urllib.request.Request(
                    f"{config.TRACKER_CONTROL_URL.rstrip('/')}/mute",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                urllib.request.urlopen(req, timeout=1.5)
            except (urllib.error.URLError, OSError, TimeoutError):
                log.debug("Tracker /mute not applied (no USB tracker?)", exc_info=False)

        threading.Thread(target=_post_tracker, daemon=True, name="tracker-mute").start()

    def register(self, name: str, module_class):
        info = ModuleInfo(name, module_class())
        self.modules.append(info)
        return info

    def start_all(self):
        for mod in self.modules:
            self._start_module(mod)

    def _start_module(self, mod: ModuleInfo):
        try:
            mod.instance.start(self.bus)
            log.info(f"Started: {mod.name}")
        except Exception:
            log.exception(f"Failed to start: {mod.name}")

    def _restart_module(self, mod: ModuleInfo):
        now = time.time()
        # Check if too many restarts
        mod.restart_times = [t for t in mod.restart_times if now - t < 60]
        if len(mod.restart_times) >= 3:
            mod.failed = True
            log.critical(f"{mod.name} failed — 3 restarts in 60s, giving up")
            self.bus.emit("module_failed", name=mod.name)
            return

        try:
            mod.instance.stop()
        except Exception:
            pass

        # Create fresh instance
        mod.instance = type(mod.instance)()
        self._start_module(mod)
        mod.restarts += 1
        mod.last_restart = now
        mod.restart_times.append(now)
        log.warning(f"Restarted: {mod.name} (#{mod.restarts})")
        self.bus.emit("module_restarted", name=mod.name)

    def supervision_loop(self):
        while self._running:
            time.sleep(5)
            for mod in self.modules:
                if mod.failed:
                    continue
                if not mod.instance.is_alive():
                    log.warning(f"{mod.name} is dead, restarting...")
                    self._restart_module(mod)

    def stop_all(self):
        self._running = False
        for mod in self.modules:
            try:
                mod.instance.stop()
                log.info(f"Stopped: {mod.name}")
            except Exception:
                log.exception(f"Error stopping: {mod.name}")

    def health(self) -> dict:
        uptime_s = time.time() - self.start_time
        hours = int(uptime_s // 3600)
        mins = int((uptime_s % 3600) // 60)
        return {
            "status": "ok" if all(not m.failed for m in self.modules) else "degraded",
            "uptime": f"{hours}h {mins}m",
            "modules": {
                m.name: {
                    "alive": m.instance.is_alive() and not m.failed,
                    "restarts": m.restarts,
                    "last_restart": (
                        f"{int(time.time() - m.last_restart)}s ago"
                        if m.last_restart else None
                    ),
                    "failed": m.failed,
                }
                for m in self.modules
            },
            "muted": self._muted,
        }


# ── HTTP Server (tracker bridge + health) ────────────────────────


class MerlinHTTPHandler(http.server.BaseHTTPRequestHandler):
    orchestrator = None  # set before serving

    def _get_module(self, name):
        for m in self.orchestrator.modules:
            if m.name == name:
                return m.instance
        return None

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))

        if self.path == "/event":
            body = self.rfile.read(length)
            try:
                data = json.loads(body)
                event_type = data.get("type", "")
                if event_type in ("face_arrived", "face_lost", "pir_motion"):
                    self.orchestrator.bus.emit(event_type)
                    self.send_response(200)
                else:
                    log.debug(f"Unknown event type: {event_type}")
                    self.send_response(200)
            except Exception:
                self.send_response(400)
            self.end_headers()

        elif self.path == "/stt":
            # Pi client sends WAV audio, we transcribe and return text
            import tempfile
            audio_data = self.rfile.read(length)
            audio_mod = self._get_module("audio_pipeline")
            if audio_mod and audio_mod._stt:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(audio_data)
                    wav_path = f.name
                t0 = time.time()
                text = audio_mod._stt.transcribe_file(wav_path)
                elapsed = time.time() - t0
                try:
                    import os; os.unlink(wav_path)
                except Exception:
                    pass
                log.info(f'[stt] "{text}" ({elapsed:.1f}s)')
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"text": text}).encode())
            else:
                self.send_response(503)
                self.end_headers()

        elif self.path == "/think":
            body = json.loads(self.rfile.read(length))
            text = body.get("text", "")
            brain = self._get_module("brain")
            if brain:
                log.info(f'[pi-heard] "{text}"')
                t0 = time.time()
                from brain import Intent, classify_intent
                intent = classify_intent(text)
                reply = brain._think(text, intent=intent)
                elapsed = time.time() - t0
                if reply:
                    log.info(f'[pi-reply] "{reply}" ({elapsed:.1f}s)')
                else:
                    reply = ""
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"reply": reply}).encode())
            else:
                self.send_response(503)
                self.end_headers()

        elif self.path == "/tts":
            body = json.loads(self.rfile.read(length))
            text = body.get("text", "")
            voice = self._get_module("voice")
            if voice:
                t0 = time.time()
                audio = voice._generate_tts(text)
                elapsed = time.time() - t0
                if audio:
                    log.info(f"[tts] {len(audio)} bytes ({elapsed:.1f}s)")
                    self.send_response(200)
                    self.send_header("Content-Type", "audio/wav")
                    self.send_header("Content-Length", str(len(audio)))
                    self.end_headers()
                    self.wfile.write(audio)
                else:
                    self.send_response(500)
                    self.end_headers()
            else:
                self.send_response(503)
                self.end_headers()

        elif self.path == "/speak":
            body = json.loads(self.rfile.read(length))
            text = (body.get("text") or "").strip()
            if not text:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": "missing text"}).encode())
            else:
                log.info(f'[http-speak] "{text[:200]}{"…" if len(text) > 200 else ""}"')
                self.orchestrator.bus.emit("speak", text=text)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": True}).encode())

        elif self.path == "/ptz":
            body = json.loads(self.rfile.read(length))
            action = (body.get("action") or "").strip()
            if action not in config.PTZ_ACTIONS:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps(
                        {
                            "ok": False,
                            "error": "invalid or missing action",
                            "allowed": sorted(config.PTZ_ACTIONS),
                        }
                    ).encode()
                )
            else:
                log.info(f"[http-ptz] {action}")
                self.orchestrator.bus.emit("ptz_action", action=action)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": True, "action": action}).encode())

        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            health = self.orchestrator.health()
            body = json.dumps(health, indent=2).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/scene":
            vision = self._get_module("vision")
            if vision and hasattr(vision, "scene_snapshot"):
                body = json.dumps(vision.scene_snapshot()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(503)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # suppress default HTTP logging


# ── Main ─────────────────────────────────────────────────────────

def main():
    orch = Orchestrator()
    atexit.register(_stop_mcp_clients)

    # Register modules
    orch.register("audio_pipeline", AudioPipeline)
    orch.register("voice", Voice)
    orch.register("brain", Brain)
    orch.register("vision", Vision)

    # Graceful shutdown
    def shutdown(sig, frame):
        log.info("Shutting down...")
        w = getattr(orch, "_imessage_watcher", None)
        if w:
            w.stop()
        _stop_mcp_clients()
        orch.stop_all()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Start HTTP server — threaded so STT/TTS don't block health checks
    MerlinHTTPHandler.orchestrator = orch

    class ThreadedHTTPServer(http.server.ThreadingHTTPServer):
        allow_reuse_address = True

    http_server = ThreadedHTTPServer(("0.0.0.0", config.TRACKER_LISTEN_PORT), MerlinHTTPHandler)
    http_thread = threading.Thread(target=http_server.serve_forever, daemon=True, name="http")
    http_thread.start()
    log.info(f"HTTP server on :{config.TRACKER_LISTEN_PORT}")

    # Start all modules
    orch.start_all()
    watcher = start_imessage_watcher_if_enabled(orch.bus)
    if watcher:
        orch._imessage_watcher = watcher
    _maybe_autostart_mcp()
    # Startup cue so boot completion is audible.
    orch.bus.emit("speak_nonverbal", sound="ready")

    log.info("=" * 50)
    log.info("Merlin v2 — All modules running")
    log.info(f"Health: http://localhost:{config.TRACKER_LISTEN_PORT}/health")
    log.info("=" * 50)

    # Supervision loop (blocking)
    orch.supervision_loop()


if __name__ == "__main__":
    main()
