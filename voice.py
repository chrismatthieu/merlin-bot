"""Merlin v2 — Voice output: Kokoro TTS + speaker EQ + go2rtc push."""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import requests

from event_bus import EventBus
import config

log = logging.getLogger("merlin.voice")


def apply_speaker_eq(audio_bytes: bytes) -> bytes:
    """Apply EQ optimized for the Amcrest camera's tiny speaker."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", "pipe:0",
             "-af", (
                 "highpass=f=200,"
                 "lowpass=f=3800,"
                 "equalizer=f=300:width_type=o:width=2:g=-3,"
                 "equalizer=f=2500:width_type=o:width=2:g=4,"
                 "equalizer=f=3200:width_type=o:width=2:g=2,"
                 "acompressor=threshold=-18dB:ratio=3:attack=5:release=50:makeup=2,"
                 "loudnorm=I=-16:LRA=7:TP=-1.5"
             ),
             "-f", "mp3", "pipe:1"],
            input=audio_bytes, capture_output=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout:
            return result.stdout
        return audio_bytes
    except Exception:
        return audio_bytes


def get_audio_duration(audio_bytes: bytes) -> float:
    """Get duration of audio in seconds using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-i", "pipe:0", "-show_entries", "format=duration",
             "-v", "quiet", "-of", "csv=p=0"],
            input=audio_bytes, capture_output=True, timeout=5,
        )
        return float(result.stdout.strip())
    except Exception:
        # Fallback estimate: MP3 at ~6KB/s
        return max(len(audio_bytes) / 6000, 1.0)


class Voice:
    """Voice output module. Implements the Module contract."""

    def __init__(self):
        self._bus = None
        self._tts_model = None
        self._lock = threading.Lock()
        self._use_system_tts_fallback = False

    def start(self, bus: EventBus, cfg=None) -> None:
        self._bus = bus
        bus.on("speak", self._on_speak)
        bus.on("speak_nonverbal", self._on_speak_nonverbal)
        bus.on("ptz_action", self._on_ptz_action)
        self._load_tts()

    def stop(self) -> None:
        if self._bus:
            self._bus.off("speak", self._on_speak)
            self._bus.off("speak_nonverbal", self._on_speak_nonverbal)
            self._bus.off("ptz_action", self._on_ptz_action)

    def is_alive(self) -> bool:
        return True  # Voice is event-driven (no background thread to die)

    def _load_tts(self):
        """Load Kokoro TTS model."""
        try:
            from mlx_audio.tts import generate as _  # test import
            log.info(f"Kokoro TTS ready (voice: {config.KOKORO_VOICE})")
            self._use_system_tts_fallback = False
        except ImportError:
            log.warning("mlx-audio TTS not available — voice will be silent")
            self._use_system_tts_fallback = True

    def _on_speak(self, text: str = "") -> None:
        """Handle speak event — generate TTS and push to speaker."""
        if not text:
            return
        threading.Thread(
            target=self._speak_thread, args=(text,), daemon=True, name="speak"
        ).start()

    def _on_speak_nonverbal(self, sound: str = "") -> None:
        """Play a pre-recorded sound file."""
        if not sound:
            return
        sound_path = config.SOUNDS_DIR / f"{sound}.mp3"
        if sound_path.exists():
            threading.Thread(
                target=self._play_file, args=(sound_path,), daemon=True, name="nonverbal"
            ).start()
        else:
            log.warning(f"Sound not found: {sound_path}")

    def _on_ptz_action(self, action: str = "") -> None:
        """Handle explicit PTZ movement commands from brain."""
        if not action:
            return
        threading.Thread(
            target=self._run_ptz_action, args=(action,), daemon=True, name="ptz-action"
        ).start()

    def _speak_thread(self, text: str) -> None:
        """Generate TTS and push to camera speaker. Runs in a thread."""
        with self._lock:  # only one utterance at a time
            try:
                gesture = self._infer_gesture_from_text(text)
                if gesture:
                    # Run PTZ gesture in parallel so movement overlaps speech.
                    threading.Thread(
                        target=self._run_ptz_gesture, args=(gesture,), daemon=True, name="ptz-gesture"
                    ).start()

                audio = self._generate_tts(text)
                if not audio:
                    if self._use_system_tts_fallback:
                        if self._speak_with_system_tts(text):
                            return
                    log.warning(f"TTS failed, would say: {text}")
                    self._bus.emit("speak_failed")
                    return

                # Skip EQ — Mac speakers don't need camera speaker optimization
                self._bus.emit("speaking_started")
                self._push_to_speaker(audio)  # afplay blocks until done
                self._bus.emit("speaking_finished")

            except Exception:
                log.exception("Speak error")
                self._bus.emit("speak_failed")
                self._bus.emit("speaking_finished")

    def _play_file(self, path: Path) -> None:
        """Play a pre-recorded file through the camera speaker."""
        with self._lock:
            try:
                audio = path.read_bytes()
                self._bus.emit("speaking_started")
                self._push_to_speaker(audio)  # afplay blocks until done
                self._bus.emit("speaking_finished")
            except Exception:
                log.exception(f"Play file error: {path}")
                self._bus.emit("speaking_finished")

    def _generate_tts(self, text: str) -> bytes | None:
        """Generate speech audio from text using Kokoro via mlx-audio."""
        try:
            if self._tts_model is None:
                from mlx_audio.tts.generate import load_model
                self._tts_model = load_model("prince-canuma/Kokoro-82M")
                log.info("Kokoro model loaded")

            # Generate audio — returns a generator of GenerationResult chunks
            import numpy as np
            import mlx.core as mx

            # Clean text for TTS — strip newlines, extra spaces, special chars
            clean_text = " ".join(text.replace("\n", " ").split())
            clean_text = clean_text.strip()
            if not clean_text:
                return None

            audio_chunks = []
            sample_rate = 24000
            for chunk in self._tts_model.generate(text=clean_text, voice=config.KOKORO_VOICE):
                audio_arr = np.array(chunk.audio, dtype=np.float32) if not isinstance(chunk.audio, np.ndarray) else chunk.audio
                audio_chunks.append(audio_arr)
                sample_rate = chunk.sample_rate

            if not audio_chunks:
                return None

            audio_data = np.concatenate(audio_chunks)
            # Convert float32 to int16 PCM, then to MP3
            pcm = (audio_data * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
            result = subprocess.run(
                ["ffmpeg", "-f", "s16le", "-ar", str(sample_rate), "-ac", "1",
                 "-i", "pipe:0", "-f", "mp3", "pipe:1"],
                input=pcm, capture_output=True, timeout=10,
            )
            if result.returncode == 0:
                log.info(f"TTS generated ({len(result.stdout)} bytes)")
                return result.stdout

            log.warning("ffmpeg MP3 conversion failed")
            return None

        except ImportError:
            log.warning("mlx-audio not available for TTS")
            self._use_system_tts_fallback = True
            return None
        except Exception:
            log.exception("TTS generation error")
            self._use_system_tts_fallback = True
            return None

    def _speak_with_system_tts(self, text: str) -> bool:
        """Fallback voice output using macOS built-in `say`."""
        clean_text = " ".join(text.replace("\n", " ").split()).strip()
        if not clean_text:
            return False
        try:
            self._bus.emit("speaking_started")
            result = subprocess.run(
                ["say", clean_text],
                capture_output=True,
                timeout=30,
            )
            self._bus.emit("speaking_finished")
            if result.returncode == 0:
                log.info("System TTS fallback: played via macOS say")
                return True
            log.warning(f"System TTS fallback failed: {result.stderr.decode()[:120]}")
            return False
        except Exception:
            log.exception("System TTS fallback error")
            self._bus.emit("speaking_finished")
            return False

    def _infer_gesture_from_text(self, text: str) -> str | None:
        """Return 'yes' or 'no' gesture for short binary answers."""
        normalized = " ".join(text.strip().lower().split())
        if not normalized:
            return None

        yes_prefixes = ("yes", "yeah", "yep", "correct", "exactly", "affirmative", "true")
        no_prefixes = ("no", "nope", "nah", "negative", "incorrect", "false")

        if normalized.startswith(yes_prefixes):
            return "yes"
        if normalized.startswith(no_prefixes):
            return "no"

        # LLM often responds like "The answer is yes/no...".
        first_sentence = normalized.split(".", 1)[0]
        yes_match = re.search(r"\b(yes|yeah|yep|affirmative|correct|true)\b", first_sentence)
        no_match = re.search(r"\b(no|nope|nah|negative|incorrect|false)\b", first_sentence)
        if yes_match and no_match:
            return "yes" if yes_match.start() < no_match.start() else "no"
        if yes_match:
            return "yes"
        if no_match:
            return "no"
        return None

    def _run_ptz_gesture(self, gesture: str) -> None:
        """Best-effort PTZ gesture using uvc-util CLI."""
        uvc_paths = [
            str(Path.home() / ".local" / "bin" / "uvc-util"),
            "/usr/local/bin/uvc-util",
            "/opt/homebrew/bin/uvc-util",
            "uvc-util",
        ]
        uvc_bin = None
        for p in uvc_paths:
            try:
                r = subprocess.run([p, "--version"], capture_output=True, timeout=2)
                if r.returncode == 0:
                    uvc_bin = p
                    break
            except Exception:
                continue

        if not uvc_bin:
            return

        # AVFoundation camera index follows startup scripts where PIXY is auto-detected.
        cam_idx = str(config.USB_CAMERA_INDEX)
        log.info(f"PTZ gesture: {gesture} (camera index {cam_idx})")
        if gesture == "yes":
            # Nod: center -> slight down -> slight up -> center
            sequence = [
                "{pan=0,tilt=0}",
                "{pan=0,tilt=-43200}",
                "{pan=0,tilt=43200}",
                "{pan=0,tilt=0}",
            ]
        else:
            # Shake: center -> left -> right -> center
            sequence = [
                "{pan=0,tilt=0}",
                "{pan=-54000,tilt=0}",
                "{pan=54000,tilt=0}",
                "{pan=-36000,tilt=0}",
                "{pan=36000,tilt=0}",
                "{pan=0,tilt=0}",
            ]

        for value in sequence:
            try:
                subprocess.run(
                    [uvc_bin, "-I", cam_idx, "-s", f"pan-tilt-abs={value}"],
                    capture_output=True,
                    timeout=3,
                    check=False,
                )
                time.sleep(0.28)
            except Exception:
                return

    def _run_ptz_action(self, action: str) -> None:
        """Execute larger directional PTZ moves like look left/right/around."""
        uvc_paths = [
            str(Path.home() / ".local" / "bin" / "uvc-util"),
            "/usr/local/bin/uvc-util",
            "/opt/homebrew/bin/uvc-util",
            "uvc-util",
        ]
        uvc_bin = None
        for p in uvc_paths:
            try:
                r = subprocess.run([p, "--version"], capture_output=True, timeout=2)
                if r.returncode == 0:
                    uvc_bin = p
                    break
            except Exception:
                continue
        if not uvc_bin:
            return

        cam_idx = str(config.USB_CAMERA_INDEX)
        if action == "look_left":
            sequence = ["{pan=-72000,tilt=0}", "{pan=0,tilt=0}"]
        elif action == "look_right":
            sequence = ["{pan=72000,tilt=0}", "{pan=0,tilt=0}"]
        elif action == "look_up":
            sequence = ["{pan=0,tilt=54000}", "{pan=0,tilt=0}"]
        elif action == "look_down":
            sequence = ["{pan=0,tilt=-54000}", "{pan=0,tilt=0}"]
        elif action == "look_center":
            sequence = ["{pan=0,tilt=0}"]
        elif action == "look_around":
            sequence = [
                "{pan=0,tilt=0}",
                "{pan=-72000,tilt=0}",
                "{pan=72000,tilt=0}",
                "{pan=0,tilt=0}",
            ]
        else:
            return

        log.info(f"PTZ action: {action} (camera index {cam_idx})")
        for value in sequence:
            try:
                subprocess.run(
                    [uvc_bin, "-I", cam_idx, "-s", f"pan-tilt-abs={value}"],
                    capture_output=True,
                    timeout=4,
                    check=False,
                )
                time.sleep(0.5)
            except Exception:
                return

    def _push_to_speaker(self, audio_bytes: bytes) -> None:
        """Play audio on Nate's Mac speakers via afplay.

        Simple, reliable, instant. No SCP, no Pi, no go2rtc speaker push.
        Future: switch back to camera speaker when hardware is better.
        """
        import uuid
        filename = f"merlin_speak_{uuid.uuid4().hex[:8]}.mp3"
        local_path = f"/tmp/{filename}"

        try:
            with open(local_path, "wb") as f:
                f.write(audio_bytes)

            # afplay is macOS built-in, plays immediately, blocks until done
            result = subprocess.run(
                ["afplay", local_path],
                capture_output=True, timeout=30,
            )
            if result.returncode != 0:
                log.warning(f"afplay failed: {result.stderr.decode()[:100]}")
            else:
                log.info("Speaker: played via Mac speakers")

        except subprocess.TimeoutExpired:
            log.warning("afplay timed out")
        except Exception:
            log.exception("Speaker error")
        finally:
            try:
                Path(local_path).unlink()
            except Exception:
                pass


# ── Standalone test ──────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[voice] %(message)s")
    bus = EventBus()
    bus.on("speaking_started", lambda: print(">>> Speaking..."))
    bus.on("speaking_finished", lambda: print(">>> Done speaking."))

    voice = Voice()
    voice.start(bus)

    import sys
    text = " ".join(sys.argv[1:]) or "Morning."
    print(f'Saying: "{text}"')
    bus.emit("speak", text=text)

    time.sleep(10)  # wait for playback
    print("Test complete.")
