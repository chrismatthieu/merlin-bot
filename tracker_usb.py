#!/usr/bin/env python3
"""
Merlin Face Tracker v5 — USB Edition (EMEET PIXY).

YuNet face detection + UVC PTZ control. Everything on Nate's Mac.
No Pi, no RTSP, no ONVIF. USB camera = zero latency.

Replaces tracker.py (v3, ONVIF/RTSP) when EMEET PIXY is connected.

The PD controller, smoothing, deadband, and face detection logic are
preserved from v3. Only the I/O layer changes: OpenCV USB capture
replaces RTSP, UVCPTZController replaces ONVIF SOAP.

Run:  python3 merlin/tracker_usb.py
Stop: Ctrl+C (returns camera to home)
"""

import csv
import http.server
import json
import os
import queue
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

# ── Config ──────────────────────────────────────────────────────

# Camera index (0 = first camera, 1 = second — set after probe)
CAMERA_INDEX = int(os.getenv("MERLIN_CAMERA_INDEX", "0"))  # PIXY confirmed at index 0 on Chris's Mac, index 1 on Nate's Mac
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_FPS = 30

# YuNet model
YUNET_MODEL = os.path.join(os.path.dirname(__file__), "models", "face_detection_yunet_2023mar.onnx")

# Tracking parameters (preserved from v3 — tuned for Amcrest, retune for PIXY)
DEADBAND = 0.03
SPEED_FAST = 5.0
SPEED_FINE = 0.7
FINE_ZONE = 0.20
MIN_VELOCITY = 0.12
FACE_LOST_TIMEOUT = 2.5
FACE_MISS_FRAMES = 12
FACE_HIT_FRAMES = 6
FACE_CONF_MIN = 0.62
FACE_CONF_MIN_SCAN_LOCK = 0.80
SCAN_LOCK_CENTER_TOL_X = 0.22
SCAN_ENABLED = True
SCAN_HOLD_SEC = 1.2
SCAN_HORIZONTAL_PASS = [-60.0, -20.0, 20.0, 60.0, 20.0, -20.0, 0.0]
SCAN_PASS_TILTS = [0.0, 32.0, -32.0]  # middle, up, down

# Smoothing + PD control
SMOOTH_ALPHA = 0.7
KP = 1.0
KD = 0.7
VELOCITY_RAMP = 0.8
VELOCITY_THRESHOLD = 0.03

# Axis mapping — MUST be verified empirically with PIXY
# These may need to be flipped from the Amcrest values
PAN_SIGN = 1.0
TILT_SIGN = -1.0

# PTZ scale: convert PD output (0-1 range velocity) to degrees for UVC
# Amcrest used ONVIF velocity (-1 to +1). PIXY uses absolute degrees.
# This controls how aggressively the camera moves per frame.
PTZ_SCALE_PAN = 2.0   # degrees per unit of PD output
PTZ_SCALE_TILT = 1.5  # degrees per unit of PD output

# Brain notification
BRAIN_URL = os.getenv("MERLIN_BRAIN_URL", "http://localhost:8900/event")

# Local HTTP: voice.py pauses face tracking during nod/shake/look PTZ scripts
TRACKER_CONTROL_HOST = os.getenv("MERLIN_TRACKER_CONTROL_HOST", "127.0.0.1")
try:
    TRACKER_CONTROL_PORT = int(os.getenv("MERLIN_TRACKER_CONTROL_PORT", "8903"))
except ValueError:
    TRACKER_CONTROL_PORT = 8903

# When brain mutes/sleeps, point the lens away (desk). When unmuted, room-scan like face-lost.
# Default tilt is not -90°: some PIXY/UVC stacks glitch or stall at full down (black frames, dead USB),
# which breaks wake UX. Use MERLIN_PRIVACY_TILT_DEG=-90 only if your unit handles it reliably.
PRIVACY_PAN_DEG = float(os.getenv("MERLIN_PRIVACY_PAN_DEG", "0"))
PRIVACY_TILT_DEG = float(os.getenv("MERLIN_PRIVACY_TILT_DEG", "-55"))


# ── Brain Notification ────────────────────────────────────────

_last_notified = None

def notify_brain(event_type):
    """Notify brain module of face events via HTTP."""
    global _last_notified
    if event_type == _last_notified:
        return
    _last_notified = event_type
    try:
        import requests
        requests.post(BRAIN_URL, json={"type": event_type}, timeout=1)
    except Exception:
        pass


# ── PTZ Control ───────────────────────────────────────────────

class PTZController:
    """Wraps UVC PTZ for the tracking loop.

    The tracker outputs velocity-like values (from PD controller).
    This converts them to absolute position updates for UVC.
    Tracks current position internally and applies deltas.
    """

    def __init__(self):
        self._pan = 0.0   # current pan in degrees
        self._tilt = 0.0  # current tilt in degrees
        self._ptz = None

        try:
            from ptz_uvc import UVCPTZController
            self._ptz = UVCPTZController()
            print(f"[tracker] PTZ: UVC connected")
        except Exception as e:
            print(f"[tracker] PTZ: FAILED — {e}")
            print(f"[tracker] Running in DETECTION-ONLY mode (no motor control)")

    def move(self, pan_vel, tilt_vel):
        """Apply velocity as position delta. Pan/tilt_vel are PD output values."""
        if self._ptz is None:
            return

        # Convert velocity to degree delta
        self._pan += pan_vel * PTZ_SCALE_PAN
        self._tilt += tilt_vel * PTZ_SCALE_TILT

        # Clamp to PIXY range (±155° pan, ±90° tilt)
        self._pan = max(-155.0, min(155.0, self._pan))
        self._tilt = max(-90.0, min(90.0, self._tilt))

        try:
            self._ptz.set_pantilt(self._pan, self._tilt)
        except Exception as e:
            print(f"[tracker] PTZ error: {e}")

    def stop(self):
        """No-op for absolute positioning (no momentum to stop)."""
        pass

    def home(self):
        """Return to center."""
        self._pan = 0.0
        self._tilt = 0.0
        if self._ptz:
            try:
                self._ptz.home()
            except Exception:
                pass

    def close(self):
        if self._ptz:
            self._ptz.close()

    def set_absolute(self, pan_deg: float, tilt_deg: float):
        """Set explicit pan/tilt target (used by idle room scan)."""
        self._pan = max(-155.0, min(155.0, float(pan_deg)))
        self._tilt = max(-90.0, min(90.0, float(tilt_deg)))
        if self._ptz:
            try:
                self._ptz.set_pantilt(self._pan, self._tilt)
            except Exception as e:
                print(f"[tracker] PTZ error: {e}")


class GesturePause:
    """While True, the main loop does not apply face-driven or scan PTZ (voice runs uvc-util)."""

    __slots__ = ("_lock", "paused", "saved_pan", "saved_tilt")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.paused = False
        self.saved_pan = 0.0
        self.saved_tilt = 0.0

    def begin(self, ptz: PTZController) -> dict:
        with self._lock:
            self.saved_pan = ptz._pan
            self.saved_tilt = ptz._tilt
            self.paused = True
            sp, st = self.saved_pan, self.saved_tilt
        return {
            "pan_deg": sp,
            "tilt_deg": st,
            "pan_arcsec": int(round(sp * 3600)),
            "tilt_arcsec": int(round(st * 3600)),
        }

    def end(self, ptz: PTZController, reset_pd, finalize_deg: tuple[float, float] | None) -> None:
        with self._lock:
            self.paused = False
            if finalize_deg is not None:
                pan, tilt = finalize_deg
            else:
                pan, tilt = self.saved_pan, self.saved_tilt
        ptz.set_absolute(pan, tilt)
        reset_pd()

    def abandon(self) -> None:
        """Exit gesture pause without restoring PTZ (used when muting / privacy)."""
        with self._lock:
            self.paused = False


def _start_tracker_control_http(
    gesture_paused: GesturePause,
    ptz: PTZController,
    reset_pd,
    mute_queue: "queue.Queue[bool]",
) -> None:
    """Serve POST /gesture/begin, /gesture/end, /mute on TRACKER_CONTROL_PORT (daemon thread)."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            pass

        def do_POST(self) -> None:
            content_length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(content_length) if content_length else b"{}"
            try:
                payload = json.loads(raw.decode() or "{}")
            except json.JSONDecodeError:
                payload = {}

            try:
                if self.path == "/mute":
                    mute_queue.put(bool(payload.get("muted", False)))
                    self.send_response(204)
                    self.end_headers()
                    return

                if self.path == "/gesture/begin":
                    data = gesture_paused.begin(ptz)
                    body = json.dumps(data).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if self.path == "/gesture/end":
                    fin = payload.get("finalize_deg")
                    finalize: tuple[float, float] | None = None
                    if isinstance(fin, dict):
                        finalize = (float(fin.get("pan", 0)), float(fin.get("tilt", 0)))
                    gesture_paused.end(ptz, reset_pd, finalize)
                    self.send_response(204)
                    self.end_headers()
                    return

                self.send_response(404)
                self.end_headers()
            except Exception as e:
                print(f"[tracker] control HTTP error: {e}")
                self.send_response(500)
                self.end_headers()

    try:
        srv = http.server.ThreadingHTTPServer((TRACKER_CONTROL_HOST, TRACKER_CONTROL_PORT), Handler)
    except OSError as e:
        print(f"[tracker] Control API not started ({e}) — voice gestures fall back without pause")
        return

    thread = threading.Thread(target=srv.serve_forever, daemon=True, name="tracker-ctl")
    thread.start()
    print(
        f"[tracker] Control API http://{TRACKER_CONTROL_HOST}:{TRACKER_CONTROL_PORT} "
        "(POST /gesture/begin, /gesture/end, /mute)"
    )


# ── Face Detection (YuNet) ────────────────────────────────────

yunet = cv2.FaceDetectorYN.create(YUNET_MODEL, "", (640, 480), 0.5, 0.3, 5000)

# Smaller detection input improves scan/reacquire latency on Mac.
DETECT_SIZE = (256, 144)

def detect_face(frame):
    """Detect largest face via YuNet at reduced resolution.
    Returns (cx, cy, confidence) normalized 0-1, or None."""
    small = cv2.resize(frame, DETECT_SIZE)
    yunet.setInputSize(DETECT_SIZE)
    _, faces = yunet.detect(small)

    if faces is None or len(faces) == 0:
        return None

    best = max(range(len(faces)), key=lambda i: faces[i][14])
    f = faces[best]
    conf = float(f[14])
    if conf < FACE_CONF_MIN:
        return None
    cx = (f[0] + f[2] / 2) / DETECT_SIZE[0]
    cy = (f[1] + f[3] / 2) / DETECT_SIZE[1]
    return (cx, cy, conf)


# ── Performance Logger ────────────────────────────────────────

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")

class TrackingLogger:
    def __init__(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        date_str = datetime.now().strftime("%Y-%m-%d")
        self.path = os.path.join(LOG_DIR, f"tracking-usb-{date_str}.csv")
        self.file = open(self.path, "a", newline="")
        self.writer = csv.writer(self.file)
        if os.path.getsize(self.path) == 0:
            self.writer.writerow([
                "timestamp", "face_x", "face_y", "err_x", "err_y",
                "pan_vel", "tilt_vel", "speed_mode", "detect_ms",
            ])
        self.session_start = time.monotonic()
        self.moves = 0
        self.overshoots = 0
        self.prev_err_x = 0
        print(f"[tracker] Logging to {self.path}")

    def log(self, face_x, face_y, err_x, err_y, pan_vel, tilt_vel, speed_mode, detect_ms):
        self.writer.writerow([
            f"{time.monotonic() - self.session_start:.2f}",
            f"{face_x:.3f}", f"{face_y:.3f}",
            f"{err_x:.3f}", f"{err_y:.3f}",
            f"{pan_vel:.3f}", f"{tilt_vel:.3f}",
            speed_mode, f"{detect_ms:.1f}",
        ])
        self.moves += 1
        if self.prev_err_x * err_x < 0 and abs(err_x) > DEADBAND:
            self.overshoots += 1
        self.prev_err_x = err_x
        if self.moves % 50 == 0:
            self.file.flush()

    def summary(self):
        elapsed = time.monotonic() - self.session_start
        rate = self.moves / elapsed if elapsed > 0 else 0
        print(f"[tracker] Session: {elapsed:.0f}s, {self.moves} moves, "
              f"{rate:.1f}/s, {self.overshoots} overshoots")

    def close(self):
        self.summary()
        self.file.close()


# ── Main Tracking Loop ────────────────────────────────────────

def main():
    global running
    running = True

    def shutdown(sig, frame):
        global running
        running = False

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Open USB camera (no RTSP, no buffer drain thread needed)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[tracker] ERROR: Cannot open camera at index {CAMERA_INDEX}")
        print(f"[tracker] Try: MERLIN_CAMERA_INDEX=1 python3 tracker_usb.py")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[tracker] USB camera opened: {actual_w}x{actual_h} at index {CAMERA_INDEX}")

    # Initialize PTZ + loop state (velocities referenced by gesture resume callback)
    ptz = PTZController()
    gesture_paused = GesturePause()

    logger = TrackingLogger()
    face_lost_since = None
    face_miss_frames = 0
    face_hit_frames = 0
    is_tracking = False
    is_moving = False
    last_log = 0
    scan_active = SCAN_ENABLED
    scan_idx = 0
    last_scan_move = 0.0

    smooth_x = 0.5
    smooth_y = 0.5
    prev_err_x = 0.0
    prev_err_y = 0.0
    current_pan_vel = 0.0
    current_tilt_vel = 0.0
    _last_sent_pan = 0.0
    _last_sent_tilt = 0.0

    def reset_pd_after_gesture() -> None:
        nonlocal current_pan_vel, current_tilt_vel, _last_sent_pan, _last_sent_tilt, is_moving
        current_pan_vel = 0.0
        current_tilt_vel = 0.0
        _last_sent_pan = 0.0
        _last_sent_tilt = 0.0
        is_moving = False

    mute_queue: queue.Queue[bool] = queue.Queue()
    privacy_muted = False

    _start_tracker_control_http(gesture_paused, ptz, reset_pd_after_gesture, mute_queue)

    print(f"[tracker] YuNet + UVC PTZ tracker (USB)")
    print(f"[tracker] Deadband={DEADBAND}, fast={SPEED_FAST}, fine={SPEED_FINE}")
    print("[tracker] Running.")

    try:
        while running:
            # Mute/sleep (from main.py) — privacy pose or resume room scan
            while True:
                try:
                    m = mute_queue.get_nowait()
                except queue.Empty:
                    break
                if m:
                    privacy_muted = True
                    gesture_paused.abandon()
                    reset_pd_after_gesture()
                    ptz.set_absolute(PRIVACY_PAN_DEG, PRIVACY_TILT_DEG)
                    is_tracking = False
                    is_moving = False
                    scan_active = False
                    face_hit_frames = 0
                    face_miss_frames = 0
                    face_lost_since = None
                    smooth_x = 0.5
                    smooth_y = 0.5
                    prev_err_x = 0.0
                    prev_err_y = 0.0
                    print(
                        f"[tracker] Privacy/mute — PTZ parked "
                        f"(pan={PRIVACY_PAN_DEG:.0f}°, tilt={PRIVACY_TILT_DEG:.0f}°); "
                        f"face detection off until wake"
                    )
                else:
                    privacy_muted = False
                    gesture_paused.abandon()
                    reset_pd_after_gesture()
                    is_tracking = False
                    face_hit_frames = 0
                    face_miss_frames = 0
                    face_lost_since = None
                    scan_active = SCAN_ENABLED
                    scan_idx = 0
                    last_scan_move = 0.0
                    print("[tracker] Unmuted — face detection on, seeking face (room scan)")

            # USB capture is synchronous — always returns latest frame
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # Sleep/mute: keep grabbing frames for USB stability but do not run YuNet or tracking.
            if privacy_muted:
                time.sleep(0.01)
                continue

            t_detect = time.monotonic()
            face = detect_face(frame)
            detect_ms = (time.monotonic() - t_detect) * 1000

            if face is not None:
                raw_x, raw_y, raw_conf = face
                face_lost_since = None
                face_miss_frames = 0
                face_hit_frames += 1

                if not is_tracking:
                    # During scan, only lock when detection is very confident.
                    if raw_conf < FACE_CONF_MIN_SCAN_LOCK:
                        time.sleep(0.01)
                        continue
                    # Ignore edge detections during scan; wait until target is
                    # near center of frame to avoid locking on false positives.
                    if abs(raw_x - 0.5) > SCAN_LOCK_CENTER_TOL_X:
                        time.sleep(0.01)
                        continue
                    if face_hit_frames < FACE_HIT_FRAMES:
                        # Require a stable hit streak before exiting room scan.
                        time.sleep(0.01)
                        continue
                    smooth_x = raw_x
                    smooth_y = raw_y
                    prev_err_x = raw_x - 0.5
                    prev_err_y = raw_y - 0.5
                    print(f"[tracker] Face acquired ({raw_x:.2f}, {raw_y:.2f})")
                    is_tracking = True
                    scan_active = False
                    notify_brain("face_arrived")

                # 1. Exponential smoothing
                smooth_x = SMOOTH_ALPHA * raw_x + (1 - SMOOTH_ALPHA) * smooth_x
                smooth_y = SMOOTH_ALPHA * raw_y + (1 - SMOOTH_ALPHA) * smooth_y

                # 2. Error from center
                err_x = smooth_x - 0.5
                err_y = smooth_y - 0.5

                # 3. Dead zone
                if abs(err_x) < DEADBAND and abs(err_y) < DEADBAND:
                    if is_moving and not gesture_paused.paused:
                        ptz.stop()
                        is_moving = False
                        current_pan_vel = 0.0
                        current_tilt_vel = 0.0
                    prev_err_x = err_x
                    prev_err_y = err_y
                elif not gesture_paused.paused:
                    # 4. PD controller
                    d_err_x = err_x - prev_err_x
                    d_err_y = err_y - prev_err_y

                    dist = max(abs(err_x), abs(err_y))
                    speed = SPEED_FINE if dist < FINE_ZONE else SPEED_FAST

                    target_pan = PAN_SIGN * (KP * err_x * speed + KD * d_err_x * speed)
                    target_tilt = TILT_SIGN * (KP * err_y * speed + KD * d_err_y * speed)

                    # 5. Velocity ramping
                    pan_delta = target_pan - current_pan_vel
                    tilt_delta = target_tilt - current_tilt_vel

                    if abs(pan_delta) > VELOCITY_RAMP:
                        pan_delta = VELOCITY_RAMP if pan_delta > 0 else -VELOCITY_RAMP
                    if abs(tilt_delta) > VELOCITY_RAMP:
                        tilt_delta = VELOCITY_RAMP if tilt_delta > 0 else -VELOCITY_RAMP

                    current_pan_vel += pan_delta
                    current_tilt_vel += tilt_delta

                    pan_vel = current_pan_vel
                    tilt_vel = current_tilt_vel
                    if 0 < abs(pan_vel) < MIN_VELOCITY:
                        pan_vel = MIN_VELOCITY if pan_vel > 0 else -MIN_VELOCITY
                    if 0 < abs(tilt_vel) < MIN_VELOCITY:
                        tilt_vel = MIN_VELOCITY if tilt_vel > 0 else -MIN_VELOCITY

                    pan_vel = max(-0.8, min(0.8, pan_vel))
                    tilt_vel = max(-0.8, min(0.8, tilt_vel))

                    if abs(err_x) < DEADBAND:
                        pan_vel = 0.0
                    if abs(err_y) < DEADBAND:
                        tilt_vel = 0.0

                    pan_changed = abs(pan_vel - _last_sent_pan) > VELOCITY_THRESHOLD
                    tilt_changed = abs(tilt_vel - _last_sent_tilt) > VELOCITY_THRESHOLD
                    if pan_changed or tilt_changed or not is_moving:
                        ptz.move(pan_vel, tilt_vel)
                        _last_sent_pan = pan_vel
                        _last_sent_tilt = tilt_vel
                    is_moving = True

                    speed_mode = "fine" if dist < FINE_ZONE else "fast"
                    logger.log(smooth_x, smooth_y, err_x, err_y,
                              pan_vel, tilt_vel, speed_mode, detect_ms)

                    prev_err_x = err_x
                    prev_err_y = err_y
                else:
                    # Gesture script owns PTZ — keep errors warm for smooth handoff
                    prev_err_x = err_x
                    prev_err_y = err_y

                now = time.monotonic()
                if now - last_log > 2.0:
                    print(f"[tracker] face=({smooth_x:.2f},{smooth_y:.2f}) "
                          f"err=({err_x:+.2f},{err_y:+.2f}) "
                          f"vel=({current_pan_vel:+.2f},{current_tilt_vel:+.2f}) "
                          f"{'fine' if max(abs(err_x),abs(err_y)) < FINE_ZONE else 'fast'} "
                          f"detect={detect_ms:.0f}ms")
                    last_log = now

            else:
                if is_tracking:
                    face_miss_frames += 1
                    # Ignore brief detector flicker to avoid unnecessary scanning.
                    if face_miss_frames < FACE_MISS_FRAMES:
                        time.sleep(0.01)
                        continue
                    if face_lost_since is None:
                        face_lost_since = time.monotonic()
                        ptz.stop()
                        is_moving = False
                        current_pan_vel = 0.0
                        current_tilt_vel = 0.0

                    if time.monotonic() - face_lost_since > FACE_LOST_TIMEOUT:
                        print("[tracker] Face lost → room scan")
                        is_tracking = False
                        face_hit_frames = 0
                        face_miss_frames = 0
                        face_lost_since = None
                        scan_active = SCAN_ENABLED
                        scan_idx = 0
                        last_scan_move = 0.0
                        notify_brain("face_lost")
                elif (
                    SCAN_ENABLED
                    and scan_active
                    and not gesture_paused.paused
                    and not privacy_muted
                ):
                    now = time.monotonic()
                    if now - last_scan_move >= SCAN_HOLD_SEC:
                        pass_len = len(SCAN_HORIZONTAL_PASS)
                        pass_count = len(SCAN_PASS_TILTS)
                        total_steps = pass_len * pass_count
                        step = scan_idx % total_steps
                        tilt = SCAN_PASS_TILTS[step // pass_len]
                        pan = SCAN_HORIZONTAL_PASS[step % pass_len]
                        ptz.set_absolute(pan, tilt)
                        print(f"[tracker] scanning pan={pan:+.0f} tilt={tilt:+.0f}")
                        scan_idx = (scan_idx + 1) % total_steps
                        last_scan_move = now

            time.sleep(0.01)

    finally:
        print("[tracker] Shutting down...")
        ptz.stop()
        ptz.home()
        ptz.close()
        logger.close()
        cap.release()


if __name__ == "__main__":
    main()
