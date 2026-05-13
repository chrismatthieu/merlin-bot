# Merlin

An ambient AI companion that lives on your desk. Sees, hears, thinks, speaks. All local, no cloud.

Merlin is built as an executive functioning (EF) prosthetic for ADHD. It runs on commodity hardware using only open-source, locally-hosted models. When you talk to Merlin, it responds with awareness of your state, schedule, and environment. When you leave the desk and come back, it tells you what you were doing. When you drift for 90 minutes, it says "Still here." When you vent, it reflects instead of advising.

---

## Architecture

Single Mac. No cloud. USB camera + local LLMs.

```
EMEET PIXY (USB-C) ─── Video + Audio ───> Mac (Apple Silicon)
                                            ├── tracker_usb.py  (YuNet + UVC PTZ; room scan when face lost)
                                            ├── audio_usb.py    (sounddevice mic capture)
                                            ├── audio_pipeline  (VAD + mlx-whisper STT; ffmpeg where needed)
                                            ├── brain.py        (intent classifier; OpenAI-compatible LLM)
                                            ├── vision.py       (ffmpeg/AVFoundation frames + MERLIN_VISION_MODEL)
                                            ├── voice.py        (Kokoro TTS, macOS say fallback, PTZ yes/no gestures)
                                            └── main.py         (orchestrator + HTTP :8900 /health)

USB Speaker ◄──── afplay ────────────────── voice.py
```

Chat and vision can use different models (for example **Ollama** `qwen3-vl:2b` for both chat and scene description). See `start-merlin-ollama.sh` for defaults.

| Module | File | What it does |
|--------|------|-------------|
| Orchestrator | `main.py` | Starts modules, supervises, restarts on crash, HTTP server |
| Audio Pipeline | `audio_pipeline.py` | Mic capture (USB or RTSP), Silero VAD, Whisper STT |
| Audio USB | `audio_usb.py` | Drop-in USB mic capture via sounddevice |
| Voice | `voice.py` | Kokoro TTS with **macOS `say` fallback** if mlx-audio fails; afplay; PTZ nod/shake for yes/no/true/false; directional PTZ from voice (`look left`, etc.) |
| Brain | `brain.py` | Intent classifier, state machine, OpenAI-compatible LLM (LM Studio or Ollama); prompts refer to the user by the `operator` value in `soul.md` |
| Vision | `vision.py` | Async capture (USB: AVFoundation/ffmpeg) + VLM scene description via `MERLIN_VISION_MODEL` |
| Event Bus | `event_bus.py` | In-process pub/sub connecting all modules |
| Config | `config.py` | All settings, env var overrides |
| Face Tracker | `tracker_usb.py` | YuNet face detection + UVC PTZ; **room scan** (multi-pass pan/tilt) when face is lost, with hysteresis and confidence gating |
| PTZ Controller | `ptz_uvc.py` | libuvc ctypes wrapper; **uvc-util** CLI fallback if libuvc cannot open the device |
| Camera Probe | `probe_camera.py` | Hardware detection and verification script |

### Agent Subsystem

The `agent/` directory contains a separate ReAct agent that gives Merlin's local LLM access to the filesystem, Apple Notes, iMessages, and arbitrary Mac apps via MCP:

| File | What it does |
|------|-------------|
| `agent/kernel.py` | ReAct loop with tool execution |
| `agent/mcp_client.py` | JSON-RPC 2.0 MCP client over stdio |
| `agent/tools/filesystem.py` | Sandboxed file read/write/list |
| `agent/tools/mcp_bridge.py` | Discovers and wraps MCP server tools |

`agent/mcp_servers.json` is machine-agnostic. Each server entry uses `extension_id` (Claude extension folder name under *Claude Extensions*); at runtime `agent/tools/mcp_bridge.py` searches common install locations on macOS and picks `…/<extension_id>/server/index.js`. You can override discovery with:

- `MERLIN_MCP_EXTENSIONS_ROOT` — directory that directly contains the extension folders (e.g. if Claude stores extensions somewhere nonstandard)
- `MERLIN_MCP_<SERVER>_SCRIPT` — full path to that server’s `index.js` (e.g. `MERLIN_MCP_NOTES_SCRIPT`)

You can still set explicit `args` with `~` / env vars; those are tried before `extension_id` search.

When you run **`main.py`** (including `./start-merlin-ollama.sh`), Merlin can **pre-start** the same MCP extension servers as `agent/mcp_servers.json` (**off by default** via `MERLIN_AUTOSTART_MCP`). Set **`MERLIN_AUTOSTART_MCP=1`**, set **`"enabled": true`** on the servers you want, and install the matching Claude extensions (or `MERLIN_MCP_*_SCRIPT` paths). Voice **`brain.py`** uses those tools only when **`MERLIN_BRAIN_MCP=1`**. The ReAct CLI is still `python agent/main.py` (use `--no-mcp` to skip servers there). If both app and agent start MCP, you may get duplicate Node processes.

With **`MERLIN_BRAIN_MCP=1`**, voice/chat **`brain.py`** can call the same tool definitions (Apple Notes, iMessage/SMS, Mac automation) if MCP servers are running and your LLM supports **OpenAI-compatible** `tools` / `tool_calls` (e.g. recent Ollama or LM Studio); otherwise the brain replies without tools.

With integrations off (defaults), the brain is told **not** to pretend Notes/Messages/Contacts actions succeeded.

**If you turn iMessage tools on:** the model is instructed to call **`search_contacts`** before **`send_imessage`** and not to guess numbers. Wrong recipients often come from **STT** mangling a name into digits — say the contact name clearly, or use explicit digits for numbers not in Contacts.

---

## How brain.py Works

brain.py v2 uses an intent-aware conversation architecture:

1. **Speech arrives** from the audio pipeline (via event bus)
2. **Echo detection** filters out Merlin hearing its own voice
3. **Wake word check** -- "Hey Merlin" or within 60s conversation window
4. **Intent classification** -- regex rules classify into 7 intents:
   - `GREETING`, `VENT`, `CHECK_IN`, `COMMAND`, `TRANSITION`, `QUESTION`, `GENERAL`
5. **Command short-circuit** -- capture, time, remind bypass the LLM entirely
6. **Conversation state machine** -- tracks phase (idle, greeted, working, winding down, venting) with time-based decay
7. **Intent-specific prompting** -- each intent gets a tailored system prompt injection and token limit
8. **LLM call** via OpenAI-compatible API (LM Studio, **Ollama** `/v1/chat/completions`, or any compatible server) with assembled context:
   - Character prompt (voice rules)
   - RBOS context (today's focus, energy, shift, schedule, shipped items)
   - Scene description (what the camera sees, pre-computed in background)
   - Conversation history (last 10 exchanges)
9. **Response** emitted on the event bus, picked up by voice module

**User identity:** In-app prompts and context describe the human at the desk using `operator` from `soul.md` (currently **Chris**). For reliable yes/no **PTZ gestures**, the brain nudges very short binary answers to start with “Yes.”/“No.” or “True.”/“False.”.

**Commands:** `COMMAND` intent includes phrases that move the camera (e.g. look left/right/up/down/around); these emit `ptz_action` for `voice.py` / PTZ.

### Voice Command Phrases (from `config.py`)

These phrases are configured in `config.py` and are matched by the conversation/audio flow:

- **Wake words** (`WAKE_WORDS`): auto-generated from `name` in `soul.md`.
  - For `name: Nova`, defaults are: `nova`, `hey nova`, `hi nova`, `ok nova`
- **Mute phrases** (`MUTE_WORDS`): `stop listening`, `mute`, `go to sleep`
- **Unmute phrases** (`UNMUTE_WORDS`): `start listening`, `unmute`, `wake up`
- **Cancel phrases** (`NEVERMIND_WORDS`): `nevermind`, `never mind`

To change wake words, update `name` in `soul.md`. To change mute/unmute/cancel phrases, edit the arrays in `config.py`.

### EF Prosthetic Modes

- **Context recovery**: When you return to the desk after 5+ minutes, Merlin tells you what you were working on, graduated by absence length
- **Shift cues**: Proactive time-of-day announcements at shift boundaries
- **Drift detection**: After 90 minutes of silence during work hours, a gentle "Still here."
- **Evening send-off**: When face lost after 10pm, names what shipped today
- **Vent mode**: Emotional expression triggers reflection, not advice

---

## AI Stack

All models run locally. No API keys required for core operation.

| Component | Model (examples) | Notes |
|-----------|------------------|--------|
| Chat LLM | `qwen/qwen3-vl-4b` (LM Studio), **`qwen3-vl:2b`** (Ollama) | `MERLIN_MODEL` — text + tool routing |
| Vision / scene | Same multimodal model *or* dedicated VLM, e.g. **`qwen3-vl:2b`** (Ollama) | `MERLIN_VISION_MODEL` — what `vision.py` sends to the API |
| STT | Whisper via **mlx-whisper** | Requires **ffmpeg** (e.g. `brew install ffmpeg`) on macOS |
| TTS | Kokoro (mlx-audio) | Falls back to **macOS `say`** if Kokoro/mlx-audio errors |
| VAD | Silero (torch) | Pipeline can fall back to RMS-based VAD if torch/VAD unavailable |
| Face detection | **YuNet** ONNX | `models/face_detection_yunet_2023mar.onnx` |

Typical footprint depends on which chat/vision models you load; a split Ollama setup (small chat + small VLM) is lighter than a single large MLX vision LLM.

**Model evaluation:** A custom eval harness (`tools/merlin-model-eval.py`) tests models across 5 tiers: speed, instruction following, context grounding, conversation quality, and vision. Qwen3 VL 4B scored 81% vs 8B's 80% — the 4B wins on speed with nearly identical quality.

---

## Hardware

### Current Setup
- **Camera**: EMEET PIXY (USB-C PTZ webcam, 4K, 310° pan, 180° tilt, 3-mic array)
- **Speaker**: USB speaker for voice output
- **Compute**: Apple Silicon Mac with 16GB+ RAM
- **Connection**: Single USB-C cable. No network, no Pi, no RTSP.

### Also Works With
- **IP Camera**: Amcrest IP4M-1041B or similar ONVIF PTZ camera (use `tracker.py` instead of `tracker_usb.py`)
- **Raspberry Pi 5**: For remote face tracking with ONVIF cameras (legacy `tracker.py`)

---

## macOS: PIXY + Ollama (recommended demo path)

Use this when the **EMEET PIXY** is the mic and camera and you want **local Ollama** instead of LM Studio.

**Prerequisites**

- [Ollama](https://ollama.com) running (`ollama serve`) with models pulled, e.g. `ollama pull qwen3-vl:2b`
- **ffmpeg** on PATH (`brew install ffmpeg`) — STT-related tooling and AVFoundation device probing
- **uvc-util** for reliable PTZ on some PIXY/macOS setups: build from [jtfrey/uvc-util](https://github.com/jtfrey/uvc-util) and install the binary to `~/.local/bin` (or PATH). `ptz_uvc.py` and `voice.py` also search common paths.
- YuNet weights: `models/face_detection_yunet_2023mar.onnx`
- Python 3.11+ with deps installed (venv or Conda — the repo scripts default to `PYTHON_BIN=/Users/.../miniconda3/envs/merlin311/bin/python`; override with `export PYTHON_BIN=...`)

**Start Merlin** (refuses to start if PIXY is not listed by AVFoundation when `MERLIN_AUDIO_SOURCE=usb`). This single command starts the orchestrator, waits for `http://localhost:8900/health`, then starts **`tracker_usb.py`** so the PIXY follows your face:

```bash
./start-merlin-ollama.sh
```

To run the orchestrator **without** the USB face tracker (e.g. debugging): `MERLIN_START_TRACKER=0 ./start-merlin-ollama.sh`

**Tracker only** (same as before): `./start-tracker-pixy.sh` — useful if you split processes across terminals.

The tracker notifies the brain at `http://localhost:8900/event` for face arrived/lost.

**Useful environment overrides** (see `config.py`):

| Variable | Role |
|----------|------|
| `MERLIN_LLM_URL` | OpenAI-compatible chat URL (default Ollama: `http://localhost:11434/v1/chat/completions`) |
| `MERLIN_MODEL` | Chat model id (default **`qwen3-vl:2b`** with `start-merlin-ollama.sh`; LM Studio default in `config.py` is `qwen/qwen3-vl-4b`) |
| `MERLIN_VISION_MODEL` | VLM for scene description (default `qwen3-vl:2b`) |
| `MERLIN_AUDIO_SOURCE` | `usb` for PIXY on the same Mac |
| `MERLIN_CAMERA_INDEX` | Video device index; start scripts usually set this from `ffmpeg -f avfoundation -list_devices` |
| `MERLIN_TRACKER_CONTROL_PORT` | **`tracker_usb.py`** listens here so **`voice.py`** can pause face tracking during PTZ gestures (`8903` default). Set **`MERLIN_TRACKER_CONTROL_URL`** if host/port differ. |
| `MERLIN_NONVERBAL` | Nonverbal sound cues on/off (`1` default, set `0`/`false` to disable `open/close/thinking/ready`) |
| `MERLIN_AUTOSTART_MCP` | Start Claude MCP extension servers when `main.py` launches (`0` default; set `1` to enable) |
| `MERLIN_BRAIN_MCP` | Let **`brain.py`** call MCP tools in chat (`0` default; set `1` with servers enabled in `mcp_servers.json`) |
| `MERLIN_BRAIN_MCP_MAX_ROUNDS` | Max tool-call rounds per utterance (default `8`, cap `20`) |
| `MERLIN_IMESSAGE_POLL_INTERVAL` | Poll for **new inbound** iMessages (read-only `chat.db`) every *N* seconds (`0` default = off; set e.g. `15` to enable) |
| `MERLIN_IMESSAGE_CHAT_DB` | Path to Messages DB (default `~/Library/Messages/chat.db`) |
| `MERLIN_IMESSAGE_MIN_TEXT_LEN` | Skip notifications shorter than this (default `1`) |

**Proactive iMessage readouts:** Optional — set **`MERLIN_IMESSAGE_POLL_INTERVAL`** to a positive number (e.g. `15`). Merlin announces new texts via `imessage_watcher.py` (not via MCP). The app process needs **Full Disk Access** for `chat.db` reads. Sending/replying uses MCP + voice only when those integrations are enabled.

**Restart / port 8900 in use:** `./start-merlin-ollama.sh` stops anything listening on **8900** and stray **`tracker_usb.py`** processes before starting, so you should not see “address already in use” from a stale Merlin. If you still do, free the port manually:

```bash
lsof -nP -iTCP:8900 -sTCP:LISTEN   # note PID
kill <pid>                         # or: pkill -f main.py; pkill -f tracker_usb.py
./start-merlin-ollama.sh
curl -sS http://localhost:8900/health
```

---

## Setup (generic)

### 1. Clone and create venv

```bash
cd merlin-bot
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Python dependencies

```bash
pip install mlx-audio mlx-whisper sounddevice requests python-dotenv opencv-python
pip install torch  # for Silero VAD
```

Install **ffmpeg** separately (required on macOS for many audio/camera paths): `brew install ffmpeg`.

### 3. LLM server

**Option A — LM Studio**

1. Download [LM Studio](https://lmstudio.ai) for Apple Silicon
2. Search for and download `qwen/qwen3-vl-4b` (MLX format)
3. Start the server (Developer tab → Start Server, port 1234)

**Option B — Ollama**

Use `MERLIN_LLM_URL=http://localhost:11434/v1/chat/completions` and set `MERLIN_MODEL` / `MERLIN_VISION_MODEL` to pulled model names. The `start-merlin-ollama.sh` script sets sensible defaults.

### 4. libuvc + PTZ

```bash
brew install libusb cmake
git clone https://github.com/libuvc/libuvc.git /tmp/libuvc
cd /tmp/libuvc && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/.local ..
make && make install
```

If motor commands still fail, install **uvc-util** (see macOS section above).

### 5. Configure environment

```bash
cp .env.example .env
# Edit .env — most defaults work out of the box for USB setup
```

### 6. Plug in camera and verify

```bash
python probe_camera.py
```

### 7. Run

```bash
# Full system (after configuring LLM URL + models in env / .env)
python main.py

# macOS PIXY + Ollama (orchestrator + face tracker)
./start-merlin-ollama.sh

# Test audio only
python audio_usb.py

# Test face tracking only
MERLIN_CAMERA_INDEX=0 python tracker_usb.py

# Agent REPL
python agent/main.py
```

### 8. Health check

```bash
curl http://localhost:8900/health
```

---

## Project Structure

```
merlin-bot/
  soul.md              # Bot identity: name, character, persona, personality
  start-merlin-ollama.sh   # macOS: Ollama + PIXY guard + env defaults
  start-tracker-pixy.sh    # macOS: tracker_usb.py with auto PIXY index
  main.py              # Orchestrator
  audio_pipeline.py    # VAD + STT pipeline (source-agnostic)
  audio_usb.py         # USB mic capture via sounddevice
  voice.py             # TTS + speaker output (afplay)
  brain.py             # Intent classifier + LLM conversation
  vision.py            # Async frame capture + VLM scene description
  event_bus.py         # Pub/sub event system
  config.py            # All configuration
  tracker_usb.py       # USB face tracking (OpenCV + YuNet + UVC PTZ)
  tracker.py           # Legacy ONVIF face tracking (for IP cameras)
  ptz_uvc.py           # UVC PTZ controller (libuvc ctypes)
  probe_camera.py      # Camera hardware probe script
  gestures.py          # PTZ body language

  agent/               # ReAct agent with tool use
  personality/         # Character source material
  sounds/              # Nonverbal audio (oho, hmm, mmhmm, huh)
  models/              # YuNet face detection model
  briefing/            # RBOS state JSONs (gitignored)
  systemd/             # LaunchAgent service files
  archive/             # Previous versions
```

---

## Event Bus

All modules communicate through a simple in-process pub/sub bus.

Key events:
- `speech(text, rms, duration)` -- utterance transcribed
- `speak(text)` -- request Merlin to say something
- `face_arrived()` / `face_lost()` -- presence from tracker
- `scene_update(description, ts)` -- what the camera sees
- `frame_ready(ts)` -- fresh camera frame available
- `speaking_started()` / `speaking_finished()` -- echo suppression
- `mute_toggled(muted)` -- mute/unmute

---

## Status

This is an active build. The platform (hear, think, speak, see, track) is functional. The character and conversation design are being iterated. Future programs (morning quest, drift nudges, capture system) plug into the event bus when ready.

Built by [Chris Drake](https://x.com/Ezra_Drake) as part of the Rebel-Builder Operating System (RBOS).

---

## License

MIT License. See [LICENSE](LICENSE).

## Organon Concepts

- [[Extended Mind]]
- [[Integration (Mental)]]
- [[Automatization]]
- [[Abstraction (process of)]]
