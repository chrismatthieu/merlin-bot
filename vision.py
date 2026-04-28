"""Merlin v2 — Vision: frame capture + async scene description.

Captures frames from Pi's go2rtc. Periodically sends frames to the LLM
(Qwen3 VL 8B via LM Studio) for scene description. Caches the description
so brain.py can include it in conversation context with zero latency.

The LLM call happens in the background — never blocks conversation.
"""

import base64
import json
import logging
import subprocess
import threading
import time
import urllib.request

import requests as http_requests

from event_bus import EventBus
import config

log = logging.getLogger("merlin.vision")


class Vision:
    """Vision module. Captures frames and maintains a cached scene description."""

    def __init__(self):
        self._bus = None
        self._thread = None
        self._running = False
        self._interval = config.VISION_INTERVAL_DEFAULT
        self._conversation_active = False
        self._muted = False
        self._face_present = False
        self._consecutive_failures = 0
        self._camera_open_logged = False

        # Scene description cache
        self._scene_description = ""
        self._scene_timestamp = 0
        self._last_describe_time = 0
        self._describing = False  # Lock to prevent overlapping LLM calls

        # How often to refresh the description (seconds)
        self._describe_interval_idle = 120     # No face: every 2 min
        self._describe_interval_present = 45   # Face present: every 45s
        self._describe_interval_active = 30    # Conversation: every 30s

    def start(self, bus: EventBus, cfg=None) -> None:
        self._bus = bus
        bus.on("speaking_started", self._on_convo_active)
        bus.on("speech", self._on_convo_active)
        bus.on("speaking_finished", self._on_convo_done)
        bus.on("mute_toggled", self._on_mute)
        bus.on("face_arrived", self._on_face_arrived)
        bus.on("face_lost", self._on_face_lost)

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="vision")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _on_convo_active(self, **kw):
        self._conversation_active = True

    def _on_convo_done(self, **kw):
        threading.Timer(config.CONVERSATION_WINDOW, self._reset_convo).start()

    def _reset_convo(self):
        self._conversation_active = False

    def _on_mute(self, muted: bool = False, **kw):
        self._muted = muted

    def _on_face_arrived(self, **kw):
        self._face_present = True
        # Immediately describe the scene when someone arrives
        if not self._describing:
            threading.Thread(target=self._describe_current_frame, daemon=True).start()

    def _on_face_lost(self, **kw):
        self._face_present = False

    def _get_capture_interval(self) -> float:
        """How often to capture frames."""
        if self._muted:
            return config.VISION_INTERVAL_MUTED
        if self._conversation_active:
            return config.VISION_INTERVAL_ACTIVE
        return config.VISION_INTERVAL_DEFAULT

    def _get_describe_interval(self) -> float:
        """How often to send frames to LLM for description."""
        if self._conversation_active:
            return self._describe_interval_active
        if self._face_present:
            return self._describe_interval_present
        return self._describe_interval_idle

    def _capture_frame_go2rtc(self) -> bool:
        """Capture one frame via go2rtc snapshot API."""
        try:
            url = f"{config.GO2RTC_API}/api/frame.jpeg?src=merlin"
            resp = http_requests.get(url, timeout=5)
            if resp.status_code == 200 and len(resp.content) > 1000:
                config.FRAME_PATH.write_bytes(resp.content)
                self._consecutive_failures = 0
                return True
            self._consecutive_failures += 1
            return False
        except Exception:
            self._consecutive_failures += 1
            if self._consecutive_failures <= 3 or self._consecutive_failures % 30 == 0:
                log.debug(f"Frame capture failed ({self._consecutive_failures}x)")
            return False

    def _capture_frame_usb(self) -> bool:
        """Capture one frame from a local USB camera via AVFoundation."""
        try:
            # Use ffmpeg AVFoundation device indexing directly so we always
            # capture from the same camera index detected at startup.
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "avfoundation",
                    "-framerate",
                    str(config.USB_CAMERA_FPS),
                    "-video_size",
                    f"{config.USB_CAMERA_WIDTH}x{config.USB_CAMERA_HEIGHT}",
                    "-i",
                    f"{config.USB_CAMERA_INDEX}:none",
                    "-frames:v",
                    "1",
                    "-q:v",
                    "3",
                    str(config.FRAME_PATH),
                    "-y",
                ],
                capture_output=True,
                timeout=8,
            )
            if result.returncode != 0 or not config.FRAME_PATH.exists():
                self._consecutive_failures += 1
                return False

            if not self._camera_open_logged:
                log.info(f"Vision USB capture using AVFoundation index {config.USB_CAMERA_INDEX}")
                self._camera_open_logged = True

            self._consecutive_failures = 0
            return True
        except Exception:
            self._consecutive_failures += 1
            if self._consecutive_failures <= 3 or self._consecutive_failures % 30 == 0:
                log.debug(f"USB frame capture failed ({self._consecutive_failures}x)")
            return False

    def _capture_frame(self) -> bool:
        """Capture one frame from the configured vision source."""
        if config.AUDIO_SOURCE == "usb":
            return self._capture_frame_usb()
        return self._capture_frame_go2rtc()

    def _describe_current_frame(self):
        """Send the current frame to LLM for scene description. Runs in background."""
        if self._describing:
            return
        if not config.FRAME_PATH.exists():
            return

        self._describing = True
        try:
            frame_age = time.time() - config.FRAME_PATH.stat().st_mtime
            if frame_age > 30:
                return  # Frame is stale

            img_b64 = base64.b64encode(config.FRAME_PATH.read_bytes()).decode()

            body = json.dumps({
                "model": config.VISION_MODEL,
                "messages": [
                    {"role": "system", "content": "Describe what you see in one sentence. Be factual and brief. /no_think"},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                        {"type": "text", "text": "What do you see?"},
                    ]}
                ],
                "max_tokens": 300,
                "temperature": 0.3,
                "stream": False,
            }).encode()

            req = urllib.request.Request(
                config.LLM_URL,
                data=body,
                headers={"Content-Type": "application/json"},
            )

            with urllib.request.urlopen(req, timeout=45) as resp:
                result = json.loads(resp.read())

            text = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            if "." in text:
                # Keep the first sentence to avoid verbose visual monologues.
                text = text.split(".", 1)[0].strip() + "."
            if text and len(text) > 5:
                old = self._scene_description
                self._scene_description = text
                self._scene_timestamp = time.time()
                self._last_describe_time = time.time()

                if text != old:
                    self._bus.emit("scene_update", description=text, ts=time.time())
                    log.info(f"Scene: {text}")

        except Exception as e:
            log.debug(f"Scene description failed: {e}")
        finally:
            self._describing = False

    def _run(self):
        log.info("Vision started (frame capture + async scene description)")
        while self._running:
            interval = self._get_capture_interval()
            time.sleep(interval)

            if not self._running:
                break

            # Capture frame
            self._capture_frame()

            # Check if we should refresh the scene description
            describe_interval = self._get_describe_interval()
            since_last = time.time() - self._last_describe_time

            if since_last >= describe_interval and not self._describing:
                # Don't block the capture loop — describe in background
                threading.Thread(target=self._describe_current_frame, daemon=True).start()

        log.info("Vision stopped")


# ── Standalone test ──────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[vision] %(message)s")
    bus = EventBus()
    bus.on("scene_update", lambda description="", **kw: print(f"\n>>> SEES: {description}\n"))

    vision = Vision()
    vision.start(bus)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        vision.stop()
        print("\nStopped.")
