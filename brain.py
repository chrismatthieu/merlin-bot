"""Merlin v2 — Brain: Intent-aware LLM conversation + EF prosthetic modes."""

from __future__ import annotations

import collections
import copy
import enum
import json
import logging
import random
import re
import threading
import time
from datetime import datetime, date
from difflib import SequenceMatcher
from pathlib import Path

import requests

from event_bus import EventBus
import config
import mcp_runtime

log = logging.getLogger("merlin.brain")


def _assistant_visible_text(msg: object) -> str:
    """Final spoken line from an OpenAI-style chat `message` (never internal reasoning)."""
    if not isinstance(msg, dict):
        return ""
    return (msg.get("content") or "").strip()


def _clip_for_voice(text: str, limit: int = 480) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut + "…" if cut else text[:limit]


# When sleeping, Whisper may return "Wake", "Wake up.", "Un-mute", etc.
_MUTED_UNMUTE_FLEX_RE = re.compile(
    r"\b(wake(?:[\s\-]+up)?|wakeup|unmute|unmutes|start\s+listening)\b",
    re.I,
)


def _speech_unmutes_merlin(text: str) -> bool:
    """True if this transcript should unmute: list phrases, wake name, or flexible STT variants."""
    if any(config.heard_contains_phrase(text, w) for w in config.UNMUTE_WORDS):
        return True
    if any(config.heard_contains_phrase(text, w) for w in config.WAKE_WORDS):
        return True
    t = config.normalize_heard_text(text)
    if not t:
        return False
    if _MUTED_UNMUTE_FLEX_RE.search(t):
        return True
    n = re.escape(config.BOT_NAME.strip().lower())
    if re.search(rf"\b({n}|hey\s+{n}|hi\s+{n}|ok\s+{n})\b", t, re.I):
        return True
    # Whisper often turns "Nova" → "Novo" on USB mic.
    if n == "nova" and re.search(r"\bnovo\b", t, re.I):
        return True
    return False


# ── Intent Classification ───────────────────────────────────────

class Intent(enum.Enum):
    GREETING = "greeting"
    VENT = "vent"
    CHECK_IN = "check_in"
    COMMAND = "command"
    TRANSITION = "transition"
    QUESTION = "question"
    GENERAL = "general"


# Rules checked in order — first match wins
INTENT_RULES = [
    # COMMAND — short-circuits LLM entirely
    (Intent.COMMAND, [
        r"^capture[:\s]", r"^remind me", r"^set timer", r"^mute", r"^unmute",
        r"^what time is it", r"^what date is it", r"^what day is it",
        r"^date", r"^time",
        r"\b(what('s| is)\s+the\s+time|current\s+time|time\s+now)\b",
        r"\b(what('s| is)\s+the\s+date|today'?s\s+date|current\s+date)\b",
        r"^look\b", r"^scan\b", r"^pan\b",
        r"\blook\s+(left|right|up|down|around|center|centre|straight|ahead|forward)\b",
        r"\bscan\s+(the\s+)?room\b",
        r"\bpan\s+(left|right|up|down)\b",
    ]),
    # GREETING
    (Intent.GREETING, [
        r"good morning", r"morning", r"hey merlin", r"hi merlin",
        r"^hello", r"^hey$", r"^hi$", r"what's up", r"how are you",
    ]),
    # VENT — emotional expression
    (Intent.VENT, [
        r"frustrated", r"overwhelmed", r"anxious", r"angry", r"pissed",
        r"can't do this", r"i give up", r"i'm done", r"hate this",
        r"i'm stuck", r"i don't know what", r"falling apart",
        r"i feel like", r"i'm so", r"i can't",
    ]),
    # TRANSITION — shift/mode changes
    (Intent.TRANSITION, [
        r"going to bed", r"heading out", r"taking a break", r"back to work",
        r"shift change", r"first shift", r"second shift", r"night shift",
        r"winding down", r"done for the day", r"signing off",
    ]),
    # CHECK_IN — asking about state/progress
    (Intent.CHECK_IN, [
        r"what('s| is) (my |the )?thing", r"what am i (doing|working on)",
        r"how('s| is) (my |the )?day", r"what('s| is) (my |the )?sprint",
        r"orient me", r"brief me", r"status", r"how am i doing",
        r"what('s| is) next", r"what should i",
    ]),
    # QUESTION — knowledge-seeking
    (Intent.QUESTION, [
        r"^(what|how|why|when|where|who|can|does|is|are|do|will|should)\b",
        r"\?$",
    ]),
]


def classify_intent(text: str) -> Intent:
    """Classify user intent from text. First match wins."""
    text_lower = text.lower().strip()
    bot_name = config.BOT_NAME.lower()
    if re.search(rf"\b(hey|hi|hello)\s+{re.escape(bot_name)}\b", text_lower):
        return Intent.GREETING
    for intent, patterns in INTENT_RULES:
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return intent
    return Intent.GENERAL


def is_scene_query(text: str) -> bool:
    """Detect direct requests for current visual scene."""
    text_lower = text.lower().strip()
    patterns = [
        r"\bwhat do you see\b",
        r"\bwhat can you see\b",
        r"\bwhat are you seeing\b",
        r"\bdescribe (what you see|the scene|the room|my desk)\b",
        r"\blook around\b",
        r"\bwhat's in (the room|front of you)\b",
    ]
    return any(re.search(p, text_lower) for p in patterns)


# ── Conversation State Machine ──────────────────────────────────

class ConvoPhase(enum.Enum):
    IDLE = "idle"
    GREETED = "greeted"
    WORKING = "working"
    WINDING_DOWN = "winding_down"
    VENTING = "venting"


# Phase decay timeouts (seconds)
PHASE_DECAY = {
    ConvoPhase.GREETED: 300,       # 5 min → IDLE
    ConvoPhase.WORKING: 1800,      # 30 min → IDLE
    ConvoPhase.WINDING_DOWN: 900,  # 15 min → IDLE
    ConvoPhase.VENTING: 600,       # 10 min → IDLE
}

# Phase transitions: (current_phase, intent) → new_phase
PHASE_TRANSITIONS = {
    (ConvoPhase.IDLE, Intent.GREETING): ConvoPhase.GREETED,
    (ConvoPhase.IDLE, Intent.VENT): ConvoPhase.VENTING,
    (ConvoPhase.IDLE, Intent.CHECK_IN): ConvoPhase.WORKING,
    (ConvoPhase.IDLE, Intent.QUESTION): ConvoPhase.WORKING,
    (ConvoPhase.GREETED, Intent.CHECK_IN): ConvoPhase.WORKING,
    (ConvoPhase.GREETED, Intent.QUESTION): ConvoPhase.WORKING,
    (ConvoPhase.GREETED, Intent.VENT): ConvoPhase.VENTING,
    (ConvoPhase.WORKING, Intent.VENT): ConvoPhase.VENTING,
    (ConvoPhase.WORKING, Intent.TRANSITION): ConvoPhase.WINDING_DOWN,
    (ConvoPhase.VENTING, Intent.CHECK_IN): ConvoPhase.WORKING,
    (ConvoPhase.VENTING, Intent.TRANSITION): ConvoPhase.WINDING_DOWN,
}


class ConversationStateMachine:
    """Tracks conversation phase with time-based decay."""

    def __init__(self):
        self.phase = ConvoPhase.IDLE
        self._last_update = time.time()

    def update(self, intent: Intent, hour: int) -> ConvoPhase:
        """Update phase based on new intent. Returns current phase."""
        # Check decay first
        elapsed = time.time() - self._last_update
        decay_limit = PHASE_DECAY.get(self.phase)
        if decay_limit and elapsed > decay_limit:
            self.phase = ConvoPhase.IDLE

        # Check for transition
        key = (self.phase, intent)
        if key in PHASE_TRANSITIONS:
            self.phase = PHASE_TRANSITIONS[key]

        # Time-based overrides
        if hour >= 22 and intent == Intent.GENERAL:
            self.phase = ConvoPhase.WINDING_DOWN

        self._last_update = time.time()
        return self.phase


# ── Prompt Templates ────────────────────────────────────────────

def greeting_prompt(hour: int) -> str:
    if hour < 12:
        return f"""{config.BOT_OPERATOR} just greeted you in the morning. Respond with a brief morning greeting.
If you know The Thing for today, mention it. If not, ask.
Keep it to one sentence."""
    elif hour < 18:
        return f"{config.BOT_OPERATOR} greeted you. Brief acknowledgment. One sentence."
    else:
        return f"{config.BOT_OPERATOR} greeted you in the evening. Brief, warm. One sentence."


def question_prompt() -> str:
    return f"""{config.BOT_OPERATOR} asked a question. Give the direct answer only — no reasoning, no setup, no "well" or "so".
For yes/no questions, start your first word with exactly "Yes." or "No.".
For true/false questions, start your first word with exactly "True." or "False.".
World trivia, capitals, definitions, and how things work: use normal knowledge and answer directly. Do not treat those as questions about the camera.
Only for what is physically in the room, on the desk, or in camera view: use the line that starts with "What you see:" in your instructions. If that line is empty for a *room* question, say you are not sure yet.
If you need to reference RBOS files, say what you know from context.
Under 50 words."""


def vent_prompt() -> str:
    return f"""{config.BOT_OPERATOR} is expressing frustration or emotional distress.
DO NOT: motivate, give advice, list solutions, or say "I understand."
DO: Reflect what you hear. Ask one question. Keep space open.
Use a Branden stem if appropriate: "If I bring 5% more awareness to what I'm feeling..."
Under 30 words."""


def transition_prompt(phase_name: str) -> str:
    return f"""{config.BOT_OPERATOR} is transitioning ({phase_name}). Acknowledge briefly.
If ending the day: name one thing that shipped.
If starting: name The Thing.
One sentence."""


def checkin_prompt() -> str:
    return f"""{config.BOT_OPERATOR} wants a status check. Use your context to answer:
- What's The Thing today?
- What shift is it?
- What's the energy?
Be direct. Bullet points. Under 50 words."""


def general_prompt() -> str:
    return "Respond naturally. Direct answer only, no reasoning trail. Brief. Under 30 words."


INTENT_PROMPTS = {
    Intent.GREETING: lambda h: greeting_prompt(h),
    Intent.QUESTION: lambda h: question_prompt(),
    Intent.VENT: lambda h: vent_prompt(),
    Intent.TRANSITION: lambda h: transition_prompt("transition"),
    Intent.CHECK_IN: lambda h: checkin_prompt(),
    Intent.GENERAL: lambda h: general_prompt(),
}

# Max tokens per intent — shorter for simple, longer for questions
INTENT_MAX_TOKENS = {
    Intent.GREETING: 60,
    Intent.VENT: 80,
    Intent.CHECK_IN: 150,
    Intent.COMMAND: 30,
    Intent.TRANSITION: 60,
    Intent.QUESTION: 280,
    Intent.GENERAL: 100,
}


# ── Command Handler ─────────────────────────────────────────────

def handle_command(text: str, bus) -> str | None:
    """Handle direct commands without LLM. Returns response or None."""
    text_lower = text.lower().strip()

    # Capture
    if re.match(r"^capture[:\s]+(.+)", text_lower):
        item = re.match(r"^capture[:\s]+(.+)", text, re.IGNORECASE).group(1).strip()
        _save_capture(item)
        return f"Captured: {item}"

    # Time / date (always from system clock)
    if (
        "what time is it" in text_lower
        or re.search(r"\b(what('s| is)\s+the\s+time|current\s+time|time\s+now)\b", text_lower)
        or text_lower in {"time", "time?"}
    ):
        return datetime.now().strftime("It's %I:%M %p.")
    if (
        "what date is it" in text_lower
        or "what day is it" in text_lower
        or re.search(r"\b(what('s| is)\s+the\s+date|today'?s\s+date|current\s+date)\b", text_lower)
        or text_lower in {"date", "date?"}
    ):
        return datetime.now().strftime("Today is %A, %B %-d, %Y.")

    # Remind
    if re.match(r"^remind me[:\s]+(.+)", text_lower):
        item = re.match(r"^remind me[:\s]+(.+)", text, re.IGNORECASE).group(1).strip()
        _save_capture(f"REMINDER: {item}")
        return f"I'll remind you: {item}"

    # Timer — no OS alarm yet; log like a reminder so nothing is silently dropped.
    if re.search(r"\b(set\s+(a\s+)?timer|start\s+(a\s+)?timer)\b", text_lower):
        _save_capture(f"TIMER: {text.strip()}")
        return "I can't fire the system clock yet. I saved that as a reminder line. Use the Clock app for a real alarm."

    # Camera movement
    if re.search(r"\b(look|scan|pan)\b", text_lower):
        if re.search(r"\b(left)\b", text_lower):
            bus.emit("ptz_action", action="look_left")
            return "Looking left."
        if re.search(r"\b(right)\b", text_lower):
            bus.emit("ptz_action", action="look_right")
            return "Looking right."
        if re.search(r"\b(up)\b", text_lower):
            bus.emit("ptz_action", action="look_up")
            return "Looking up."
        if re.search(r"\b(down)\b", text_lower):
            bus.emit("ptz_action", action="look_down")
            return "Looking down."
        if re.search(r"\b(center|centre|straight|ahead|forward)\b", text_lower):
            bus.emit("ptz_action", action="look_center")
            return "Centering."
        if re.search(r"\b(around|room|scan)\b", text_lower):
            bus.emit("ptz_action", action="look_around")
            return "Scanning the room."

    return None


def _save_capture(item: str):
    """Save a captured item to RBOS inbox."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    capture_dir = Path(config.RBOS_ROOT) / "inbox" if hasattr(config, 'RBOS_ROOT') else Path.home() / "Documents/RBOS/inbox"
    capture_file = capture_dir / "merlin-captures.md"
    try:
        capture_dir.mkdir(parents=True, exist_ok=True)
        with open(capture_file, "a") as f:
            f.write(f"- [ ] {item} *(Merlin capture, {timestamp})*\n")
        log.info(f"Captured to {capture_file}: {item}")
    except Exception as e:
        log.error(f"Capture failed: {e}")

# ── System Prompt ────────────────────────────────────────────────

SYSTEM_PROMPT = """You are {bot_name}, an ambient AI companion on {bot_operator}'s desk.

Character: {bot_character}
Persona: {bot_persona}
Personality: {bot_personality}

Voice rules:
- One or two short sentences. Under 30 words total.
- Plain declarative speech. No exclamation points. No therapy language.
- Answer the question asked only. Never narrate reasoning, steps, uncertainty, or how you decided.
- No "I think", "first", "let me", "I'd say", or thinking out loud. Give the conclusion only.
- Straight trivia (capitals, definitions, how-to): answer directly. That is not doing his thinking for him.
- You help {bot_operator} think. You do not think for him.
- You do not motivate, lecture, or list tasks. You observe and reflect.
- When he's stuck, ask one question. When he succeeds, name it simply.
- Never say: should, need to, just, obviously, productive, remember, try.

{intent_prompt}

Current time: {time}
Conversation phase: {phase}
{rbos_context}
{scene_context}
/no_think"""

# Appended to system prompt when MCP tools are available (Notes, Messages, Mac automation).
MCP_TOOL_GUIDANCE = """You have function-calling tools for Apple Notes, Messages (iMessage/SMS), and Mac automation (AppleScript).
When {operator} asks to create or change a note, search Notes, send a text, or drive Mac apps (including opening or typing into Claude Desktop), call the correct tool.
You must use a tool for any real-world action outside this chat. Do not say you saved a note or sent a message unless a tool returned success.
After tools finish, reply in one or two short sentences for voice, under 30 words unless reporting an error.

Messages / iMessage — strict rules:
- Before send_imessage, call search_contacts using the name or email {operator} said. Use only a phone number or handle that search_contacts returns (or that {operator} clearly spoke as digits). Never invent, guess, or use placeholder numbers (e.g. 555 numbers).
- If search_contacts finds no one, or several possible people, do not send. Ask one short clarifying question.
- If the transcribed request looks like a name but the payload looks like a random number, treat it as unreliable — search_contacts first, never send to an unverified number."""

CLAUDE_DELEGATE_MCP_GUIDANCE = """Claude Code (desktop / CLI) delegation:
- When {operator} asks you to have Claude Code do real work in a repository (refactor, fix bugs, run a multi-step coding task), call the tool **claude-delegate__delegate_to_claude_code** with a clear **task** string. Pass **working_directory** as the absolute path to the repo when you can infer it; otherwise the server default applies.
- Do not claim Claude did the work unless that tool returned a result. Summarize the result briefly for voice.
- Use delegation for coding/repo work, not for casual chat you can answer directly. Never ask the nested Claude to delegate back to Nova or to spawn another `claude` process."""

# When MCP did not start (no Claude extension servers), stop the model from fabricating actions.
NO_MCP_TOOLS_GUIDANCE = """Integration status: you have NO connected tools for Apple Notes, iMessage/SMS, or Mac apps in this session.
If {operator} asks to save a note, send a text, or change anything outside this chat, say honestly that you cannot — integrations are not connected. Do not pretend the action succeeded. One short sentence."""

# When brain MCP is disabled in config — do not steer the model toward Notes/Messages/Contacts at all.
BRAIN_APPLE_INTEGRATIONS_OFF_GUIDANCE = """Apple Notes, Contacts, and Messages integrations are off for this session. Do not call tools for them, do not tell {operator} you saved notes, sent messages, or looked up contacts, and do not imply those actions happened."""


# ── Context Loaders ──────────────────────────────────────────────


def load_briefing_context():
    """Load RBOS context from briefing JSONs, fallback to STATE.md."""
    context_parts = []

    # Try briefing JSONs first
    state_file = config.BRIEFING_DIR / "state.json"
    today_file = config.BRIEFING_DIR / "today.json"

    if state_file.exists():
        try:
            data = json.loads(state_file.read_text())
            if data.get("the_thing"):
                context_parts.append(f"Today's focus: {data['the_thing']}")
            if data.get("energy"):
                context_parts.append(f"Energy: {data['energy']}")
            if data.get("mode"):
                context_parts.append(f"Mode: {data['mode']}")
            if data.get("shift"):
                context_parts.append(f"Shift: {data['shift']}")
            if data.get("week_focus"):
                context_parts.append(f"This week: {data['week_focus']}")
        except Exception as e:
            log.debug(f"Briefing state.json error: {e}")

    if today_file.exists():
        try:
            data = json.loads(today_file.read_text())
            if data.get("shipped"):
                context_parts.append(f"Shipped today: {', '.join(data['shipped'][:5])}")
            if data.get("schedule"):
                context_parts.append(f"Schedule: {', '.join(data['schedule'][:3])}")
            if data.get("open_loops"):
                context_parts.append(f"Open loops: {', '.join(data['open_loops'][:3])}")
        except Exception as e:
            log.debug(f"Briefing today.json error: {e}")

    context_file = config.BRIEFING_DIR / "context.json"
    if context_file.exists():
        try:
            data = json.loads(context_file.read_text())
            if data.get("mood_history"):
                latest = data["mood_history"][-1]
                context_parts.append(f"Recent mood: {latest.get('mindset', 'unknown')}")
            if data.get("stems_to_try"):
                context_parts.append(f"Stem to try: {data['stems_to_try'][0]}")
        except Exception as e:
            log.debug(f"Briefing context.json error: {e}")

    # Fallback to STATE.md if no briefing data
    if not context_parts:
        try:
            state = config.STATE_PATH.read_text()
            for line in state.split("\n"):
                if line.startswith("**The Thing:**"):
                    context_parts.append(f"Today's focus: {line.replace('**The Thing:**', '').strip()}")
                elif line.startswith("**Energy:**"):
                    context_parts.append(f"Energy: {line.replace('**Energy:**', '').strip()}")
                elif line.startswith("**Mode:**"):
                    context_parts.append(f"Mode: {line.replace('**Mode:**', '').strip()}")
                elif line.startswith("**Current Shift:**"):
                    context_parts.append(f"Shift: {line.replace('**Current Shift:**', '').strip()}")
        except Exception as e:
            log.debug(f"STATE.md error: {e}")

    if context_parts:
        return f"What you know about {config.BOT_OPERATOR}:\n" + "\n".join(f"- {c}" for c in context_parts)
    return ""


# ── Brain Module ─────────────────────────────────────────────────


class Brain:
    """Brain module. Implements the Module contract."""

    def __init__(self):
        self._bus = None
        self._history = collections.deque(maxlen=config.CONVERSATION_HISTORY_SIZE)
        self._last_response_time = 0.0
        self._muted = False
        self._scene_description = ""
        self._rbos_context = ""
        self._rbos_cache_time = 0.0
        self._greeted_today = False
        self._greeting_date = None
        self._last_seen_time = 0.0
        self._last_face_lost_time = 0.0
        self._last_voice_activity = 0.0
        self._thread = None
        self._last_spoken = ""  # echo detection
        self._muted_at = 0.0  # time.time() when muted=True (for post-mute unmute guard)
        self._state_machine = ConversationStateMachine()
        self._last_intent = Intent.GENERAL
        self._fired_shift_cues = set()  # reset daily
        self._startup_face_greeted = False

    def start(self, bus: EventBus, cfg=None) -> None:
        self._bus = bus
        bus.on("speech", self._on_speech)
        bus.on("face_arrived", self._on_face_arrived)
        bus.on("face_lost", self._on_face_lost)
        bus.on("scene_update", self._on_scene_update)
        bus.on("imessage_received", self._on_imessage_received)

        # Load persisted state
        self._load_persisted_state()

        # Initial context load
        self._refresh_context()
        log.info("Brain started (intent-aware v2)")
        # Sync mute flag with listeners (e.g. AudioPipeline STT floors) after subscriptions exist.
        self._bus.emit("mute_toggled", muted=self._muted)

        # Background context refresh thread
        self._ctx_running = True
        self._thread = threading.Thread(target=self._context_refresh_loop, daemon=True, name="brain-ctx")
        self._thread.start()

    def stop(self) -> None:
        self._ctx_running = False
        if self._bus:
            self._bus.off("speech", self._on_speech)
            self._bus.off("face_arrived", self._on_face_arrived)
            self._bus.off("face_lost", self._on_face_lost)
            self._bus.off("scene_update", self._on_scene_update)
            self._bus.off("imessage_received", self._on_imessage_received)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── Event Handlers ───────────────────────────────────────────

    def _on_speech(self, text: str = "", rms: float = 0, duration: float = 0, **kw) -> None:
        """Handle transcribed speech — intent-aware v2."""
        if not text:
            return

        text_lower = text.lower().strip()
        self._last_voice_activity = time.time()

        # 0. Echo detection (not while muted — we must not drop wake / unmute attempts)
        if not self._muted and self._last_spoken:
            similarity = SequenceMatcher(None, text_lower, self._last_spoken.lower()).ratio()
            if similarity > 0.5:
                log.debug(f"Echo detected (similarity={similarity:.2f}), ignoring: {text[:50]}")
                return

        # 1. Muted — any unmute / wake variant (lists + flexible regex + name)
        if self._muted:
            if _speech_unmutes_merlin(text):
                if time.time() - self._muted_at < config.MUTE_UNMUTE_GUARD_SEC:
                    log.info(
                        "Muted — wake phrase ignored (post-mute guard %.1fs left)",
                        config.MUTE_UNMUTE_GUARD_SEC
                        - (time.time() - self._muted_at),
                    )
                    return
                self._set_muted(False)
                reply = "I'm listening."
                # Don't latch TTS text for echo detection — it often blocks the very next real utterance
                # (STT picks up speaker bleed or short phrases score similar to "I'm listening.").
                self._last_spoken = ""
                self._last_response_time = time.time()
                # Verbal "I'm listening." suppresses the mic for the whole say + padding — skips next command.
                if config.VERBAL_UNMUTE_ACK:
                    self._bus.emit("speak", text=reply)
                elif config.NONVERBAL_ENABLED:
                    self._bus.emit("speak_nonverbal", sound="open")
                else:
                    self._bus.emit("speak", text=reply)
                log.info("Unmuted (from sleep)")
                return
            log.info(f"Muted — ignored (no wake match): {text[:120]!r}")
            return

        # 2. Conversation controls
        if any(w in text_lower for w in config.NEVERMIND_WORDS):
            self._last_response_time = 0
            self._bus.emit("speak_nonverbal", sound="close")
            log.info("Conversation closed (nevermind)")
            return

        if config.is_mute_command(text_lower):
            self._set_muted(True)
            self._bus.emit("speak_nonverbal", sound="close")
            return

        # UNMUTE_WORDS are handled while muted (section 1). Do not match them here:
        # phrases like "Nova, wake up" would otherwise return before the LLM.

        # 3. Wake word check
        has_wake = any(text_lower.startswith(w) for w in config.WAKE_WORDS) or any(
            config.heard_contains_phrase(text_lower, w) for w in config.WAKE_WORDS
        )
        in_convo = (time.time() - self._last_response_time) < config.CONVERSATION_WINDOW

        if not has_wake and not in_convo:
            log.debug(f"Ignoring (no wake word, outside window): {text[:50]}")
            return
        if has_wake and not in_convo:
            self._bus.emit("speak_nonverbal", sound="open")

        # Extract message (strip wake word)
        message = text
        if has_wake:
            wake_prefixes = []
            for wake in sorted(config.WAKE_WORDS, key=len, reverse=True):
                wake_prefixes.extend([f"{wake},", f"{wake}"])
            for prefix in wake_prefixes:
                if text_lower.startswith(prefix):
                    message = text[len(prefix):].strip()
                    break

        if not message:
            message = "you said my name"

        # Nonverbal cue: start processing this turn.
        self._bus.emit("speak_nonverbal", sound="thinking")

        # Direct scene query: answer from live vision cache immediately.
        # This avoids LLM drift and guarantees "what do you see?" works.
        if is_scene_query(message):
            response = self._scene_description.strip() if self._scene_description else ""
            if not response:
                response = "I'm still scanning the scene. Ask again in a moment."
            self._last_spoken = response
            self._bus.emit("speak", text=response)
            self._last_response_time = time.time()
            return

        # 4. Classify intent
        intent = classify_intent(message)
        hour = datetime.now().hour
        phase = self._state_machine.update(intent, hour)
        self._last_intent = intent
        log.info(f"Intent: {intent.name} | Phase: {phase.name} | \"{message[:50]}\"")

        # 5. COMMAND short-circuit — only when we have a built-in handler; else LLM
        if intent == Intent.COMMAND:
            response = handle_command(message, self._bus)
            if response:
                self._last_spoken = response
                self._bus.emit("speak", text=response)
                self._last_response_time = time.time()
                return
            intent = Intent.GENERAL
            phase = self._state_machine.update(intent, hour)
            log.info("COMMAND had no handler — falling back to LLM (GENERAL)")

        # 6. Think with intent context
        self._refresh_context_if_stale()
        response = self._think(message, intent, phase)

        if response:
            self._last_spoken = response
            self._bus.emit("speak", text=response)
            self._last_response_time = time.time()

    def _on_face_arrived(self, **kw) -> None:
        """Handle face arrival — with context recovery."""
        now = time.time()
        today = date.today()
        hour = datetime.now().hour

        if self._muted:
            self._last_seen_time = now
            return

        # Reset daily state
        if self._greeting_date != today:
            self._greeted_today = False
            self._greeting_date = today
            self._fired_shift_cues = set()

        did_startup_greet = False
        # Always greet once per app run on first face lock,
        # even if "greeted_today" was restored from persisted state.
        if not self._startup_face_greeted:
            # Use full time-of-day greeting for a fresh day, otherwise a
            # slightly longer "return" line for app-start face lock.
            greeting = (
                self._build_arrival_greeting(hour)
                if not self._greeted_today
                else self._build_startup_face_greeting(hour)
            )
            self._bus.emit("speak", text=greeting)
            self._startup_face_greeted = True
            did_startup_greet = True
            log.info(f"Startup face greeting: {greeting}")

        if not self._greeted_today and not did_startup_greet:
            # First arrival today — explicit time-aware greeting with operator name.
            greeting = self._build_arrival_greeting(hour)
            self._bus.emit("speak", text=greeting)
            self._greeted_today = True
            self._state_machine.update(Intent.GREETING, hour)
            log.info(f"Greeted: {greeting}")
        elif not self._greeted_today and did_startup_greet:
            self._greeted_today = True
            self._state_machine.update(Intent.GREETING, hour)
        elif self._last_face_lost_time > 0 and self._greeted_today and (now - self._last_seen_time) > 10:
            # Context recovery — only if genuinely returned (last seen > 60s ago)
            absence = now - self._last_face_lost_time
            the_thing = self._extract_the_thing()

            if 10 <= absence < 300:
                msg = self._build_return_greeting()
                self._bus.emit("speak", text=msg)
                log.info(f"Context recovery (brief): {msg}")
            elif 300 <= absence < 900:  # 5-15 min
                msg = f"Welcome back. {the_thing}" if the_thing else "Welcome back."
                self._bus.emit("speak", text=msg)
                log.info(f"Context recovery (short): {msg}")
            elif 900 <= absence < 2700:  # 15-45 min
                minutes = int(absence / 60)
                msg = f"You left {minutes} minutes ago. {the_thing}" if the_thing else f"Welcome back. {minutes} minutes."
                self._bus.emit("speak", text=msg)
                log.info(f"Context recovery (medium): {msg}")
            elif absence >= 2700:  # 45+ min
                msg = f"Been a while. {the_thing} Still on it?" if the_thing else "Been a while."
                self._bus.emit("speak", text=msg)
                log.info(f"Context recovery (long): {msg}")

        self._last_seen_time = now
        self._persist_state()

    def _build_arrival_greeting(self, hour: int) -> str:
        operator = config.BOT_OPERATOR
        if hour < 12:
            options = [
                f"Good morning, {operator}. There you are.",
                f"Morning, {operator}. Nice to see you.",
                f"Good morning, {operator}. Ready when you are.",
                f"Hey {operator}, morning. I missed your face.",
            ]
            return random.choice(options)
        if hour < 18:
            options = [
                f"Good afternoon, {operator}. There you are.",
                f"Hey {operator}. Afternoon mode, engaged.",
                f"Afternoon, {operator}. Nice timing.",
                f"Welcome back, {operator}. Good afternoon.",
            ]
            return random.choice(options)
        options = [
            f"Good evening, {operator}. There you are.",
            f"Evening, {operator}. Glad you're back.",
            f"Hey {operator}. Evening check-in complete.",
            f"Good evening, {operator}. I was keeping watch.",
        ]
        return random.choice(options)

    def _build_return_greeting(self) -> str:
        operator = config.BOT_OPERATOR
        return random.choice([
            f"There you are, {operator}.",
            f"Welcome back, {operator}.",
            f"Found you again, {operator}.",
            f"Ah, there you are, {operator}.",
            f"Back on radar, {operator}.",
        ])

    def _build_startup_face_greeting(self, hour: int) -> str:
        operator = config.BOT_OPERATOR
        if hour < 12:
            return random.choice([
                f"Good morning again, {operator}. There you are.",
                f"Morning, {operator}. Eyes on.",
                f"Good morning, {operator}. Back on radar.",
            ])
        if hour < 18:
            return random.choice([
                f"Good afternoon, {operator}. There you are.",
                f"Afternoon, {operator}. I can see you now.",
                f"Hey {operator}, afternoon. Back online with you.",
            ])
        return random.choice([
            f"Good evening, {operator}. There you are.",
            f"Evening, {operator}. I see you now.",
            f"Good evening, {operator}. Back online and tracking.",
        ])

    def _on_face_lost(self, **kw) -> None:
        """Handle face departure — record time, evening send-off."""
        self._last_face_lost_time = time.time()
        hour = datetime.now().hour

        # Evening send-off
        if hour >= 22 and self._greeted_today:
            shipped = self._extract_shipped_count()
            if shipped:
                self._bus.emit("speak", text=f"You shipped {shipped} things today. Rest.")
            else:
                self._bus.emit("speak", text="Rest.")
            log.info("Evening send-off")
        else:
            log.debug("Face lost")

    def _on_scene_update(self, description: str = "", ts: float = 0, **kw) -> None:
        """Cache latest scene description from vision module."""
        self._scene_description = description

    def _on_imessage_received(self, text: str = "", sender: str = "", **kw) -> None:
        """Announce new inbound iMessage (from imessage_watcher poll)."""
        if self._muted:
            log.debug("iMessage notify skipped (muted)")
            return
        text = (text or "").strip()
        if len(text) < config.IMESSAGE_MIN_TEXT_LEN:
            return
        sender = (sender or "someone").strip()
        line = f"New message from {sender}: {text}"
        if len(line) > 420:
            line = line[:417] + "…"
        self._last_spoken = line
        self._bus.emit("speak", text=line)
        self._last_response_time = time.time()
        log.info("Announced iMessage (proactive)")

    # ── LLM ──────────────────────────────────────────────────────

    def _think_with_mcp_tools(
        self,
        messages: list,
        max_tokens: int,
        intent: Intent,
        message: str,
    ) -> str | None:
        """Multi-round OpenAI-style tool calling via MCP (Notes, Messages, Mac automation)."""
        tools = mcp_runtime.get_openai_tool_definitions()
        if not tools:
            return None
        req_max = max(max_tokens, 512)
        for round_i in range(config.BRAIN_MCP_MAX_ROUNDS):
            try:
                _payload = {
                    "model": config.LLM_MODEL,
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "auto",
                    "stream": False,
                    "temperature": 0.5,
                    "max_tokens": req_max,
                }
                _payload.update(config.llm_openai_request_extras())
                resp = requests.post(
                    config.LLM_URL,
                    json=_payload,
                    timeout=config.BRAIN_MCP_LLM_TIMEOUT,
                )
            except Exception:
                log.exception("LLM error (tool round)")
                return None

            if resp.status_code != 200:
                log.warning(
                    "LLM tool round failed (%s): %s",
                    resp.status_code,
                    resp.text[:400],
                )
                return None

            msg = resp.json().get("choices", [{}])[0].get("message", {}) or {}
            tool_calls = msg.get("tool_calls")

            if tool_calls:
                messages.append(msg)
                for i, tc in enumerate(tool_calls):
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function") or {}
                    name = fn.get("name") or tc.get("name")
                    raw_args = fn.get("arguments", "{}")
                    if raw_args is None:
                        raw_args = "{}"
                    elif not isinstance(raw_args, str):
                        raw_args = json.dumps(raw_args)
                    tid = tc.get("id") or f"call_{round_i}_{i}"
                    log.info("MCP tool call: %s %s", name, raw_args[:300])
                    result = mcp_runtime.execute_tool(name, raw_args)
                    messages.append({"role": "tool", "tool_call_id": tid, "content": result})
                continue

            text = _assistant_visible_text(msg)
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            text = re.sub(r"<\|channel>thought.*?<channel\|>", "", text, flags=re.DOTALL).strip()
            text = _clip_for_voice(text)
            if not text:
                log.warning(
                    "LLM tool round: empty content (model=%s); keys: %s",
                    config.LLM_MODEL,
                    list(msg.keys()),
                )
                text = "I drew a blank on that one. Ask again in a few words."
            self._history.append({"user": message, "assistant": text})
            log.info(f"[{intent.name}] Response (tools): {text}")
            return text

        log.warning("MCP tool loop exceeded max rounds")
        return None

    def _think(self, message: str, intent: Intent = Intent.GENERAL, phase: ConvoPhase = ConvoPhase.IDLE) -> str | None:
        """Send message to LLM with intent-specific prompting; use MCP tools when enabled."""
        hour = datetime.now().hour

        # Get intent-specific prompt
        prompt_fn = INTENT_PROMPTS.get(intent, INTENT_PROMPTS[Intent.GENERAL])
        intent_prompt = prompt_fn(hour)

        system = SYSTEM_PROMPT.format(
            bot_name=config.BOT_NAME,
            bot_operator=config.BOT_OPERATOR,
            bot_character=config.BOT_CHARACTER,
            bot_persona=config.BOT_PERSONA,
            bot_personality=config.BOT_PERSONALITY,
            time=datetime.now().strftime("%I:%M %p"),
            intent_prompt=intent_prompt,
            phase=phase.name.lower().replace("_", " "),
            rbos_context=self._rbos_context,
            scene_context=f"What you see: {self._scene_description}" if self._scene_description else "",
        )
        if config.BRAIN_MCP and mcp_runtime.has_mcp_tools():
            system = system + "\n\n" + MCP_TOOL_GUIDANCE.format(operator=config.BOT_OPERATOR)
            if mcp_runtime.has_claude_code_delegate_tool():
                system = system + "\n\n" + CLAUDE_DELEGATE_MCP_GUIDANCE.format(
                    operator=config.BOT_OPERATOR
                )
        elif config.BRAIN_MCP and not mcp_runtime.has_mcp_tools():
            system = system + "\n\n" + NO_MCP_TOOLS_GUIDANCE.format(operator=config.BOT_OPERATOR)
        else:
            system = system + "\n\n" + BRAIN_APPLE_INTEGRATIONS_OFF_GUIDANCE.format(operator=config.BOT_OPERATOR)

        messages = [{"role": "system", "content": system}]
        for ex in self._history:
            messages.append({"role": "user", "content": ex["user"]})
            messages.append({"role": "assistant", "content": ex["assistant"]})
        # Nudge binary questions toward explicit yes/no starts so PTZ gestures can fire.
        binary_question = re.match(r"^\s*(is|are|do|does|did|can|could|will|would|should|has|have|had)\b", message.lower())
        if binary_question:
            user_text = f'{config.BOT_OPERATOR} says: "{message}"\nAnswer with "Yes." or "No." as the first word.'
        else:
            user_text = f'{config.BOT_OPERATOR} says: "{message}"'
        messages.append({"role": "user", "content": user_text})

        # Intent-specific token limit
        max_tokens = INTENT_MAX_TOKENS.get(intent, 100)

        if config.BRAIN_MCP and mcp_runtime.has_mcp_tools():
            out = self._think_with_mcp_tools(
                copy.deepcopy(messages), max_tokens, intent, message
            )
            if out is not None:
                return out
            log.warning("Brain MCP tool path failed or unsupported; falling back without tools")

        try:
            _payload = {
                "model": config.LLM_MODEL,
                "messages": messages,
                "stream": False,
                "temperature": 0.5,
                "max_tokens": max_tokens,
            }
            _payload.update(config.llm_openai_request_extras())
            resp = requests.post(
                config.LLM_URL,
                json=_payload,
                timeout=60,
            )

            if resp.status_code == 200:
                raw = resp.json()
                msg = raw.get("choices", [{}])[0].get("message", {}) or {}
                text = _assistant_visible_text(msg)
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
                text = re.sub(r"<\|channel>thought.*?<channel\|>", "", text, flags=re.DOTALL).strip()
                text = _clip_for_voice(text)
                if text:
                    self._history.append({"user": message, "assistant": text})
                    log.info(f"[{intent.name}] Response: {text}")
                else:
                    log.warning(
                        "LLM returned empty content (model=%s); raw message keys: %s",
                        config.LLM_MODEL,
                        list(msg.keys()),
                    )
                    text = "I drew a blank on that one. Ask again in a few words."
                    self._history.append({"user": message, "assistant": text})
                return text
            log.error(f"LLM error: {resp.status_code} — {resp.text[:200]}")
            return None
        except Exception:
            log.exception("LLM error")
            return None

    # ── Helpers ──────────────────────────────────────────────────

    def _extract_the_thing(self) -> str:
        """Extract The Thing from cached RBOS context."""
        if not self._rbos_context:
            return ""
        for line in self._rbos_context.split("\n"):
            if "focus:" in line.lower() or "thing:" in line.lower():
                # Extract the value after the colon
                parts = line.split(":", 1)
                if len(parts) > 1:
                    return parts[1].strip()
        return ""

    def _extract_shipped_count(self) -> int:
        """Count shipped items from today's briefing context."""
        if not self._rbos_context:
            return 0
        count = 0
        for line in self._rbos_context.split("\n"):
            if "shipped" in line.lower():
                # Try to extract items after "Shipped today:"
                parts = line.split(":", 1)
                if len(parts) > 1:
                    items = parts[1].strip()
                    count = len([i for i in items.split(",") if i.strip()])
        return count

    # ── Context ──────────────────────────────────────────────────

    def _refresh_context(self):
        self._rbos_context = load_briefing_context()
        self._rbos_cache_time = time.time()
        if self._rbos_context:
            log.debug(f"Context refreshed ({len(self._rbos_context)} chars)")

    def _refresh_context_if_stale(self):
        if time.time() - self._rbos_cache_time > 300:  # 5 min
            self._refresh_context()

    def _context_refresh_loop(self):
        while self._ctx_running:
            time.sleep(60)  # Check every minute for shift cues + drift
            if self._ctx_running:
                self._check_shift_cues()
                self._check_drift()
                # Full context refresh every 5 min
                if time.time() - self._rbos_cache_time > 300:
                    self._refresh_context()

    # ── Shift Cues + Evening Mode ───────────────────────────────

    def _check_shift_cues(self):
        """Fire time-based cues at shift boundaries. Only if face is present."""
        if self._last_seen_time == 0:
            return
        # Only fire if face was seen recently (within 5 min)
        if time.time() - self._last_seen_time > 300:
            return

        hour = datetime.now().hour
        minute = datetime.now().minute

        cues = [
            (17, 0, "first_shift_end", "First shift's over."),
            (19, 0, "second_shift_start", "Second shift. What's the thing?"),
            (22, 0, "winding_down", "Winding down?"),
            (23, 30, "late_night", "It's 11:30. The night shift has it."),
            (1, 0, "night_shift", "It's one. Night shift takes over."),
        ]

        for cue_hour, cue_min, cue_id, cue_text in cues:
            if hour == cue_hour and minute >= cue_min and cue_id not in self._fired_shift_cues:
                self._fired_shift_cues.add(cue_id)
                self._bus.emit("speak", text=cue_text)
                log.info(f"Shift cue: {cue_text}")

    def _check_drift(self):
        """Detect extended desk time with no voice activity."""
        if self._last_seen_time == 0 or self._last_voice_activity == 0:
            return
        # Only if face is present (seen within 5 min)
        if time.time() - self._last_seen_time > 300:
            return

        silence = time.time() - self._last_voice_activity
        hour = datetime.now().hour

        # 90-min silence during work hours
        if silence > 5400 and 9 <= hour < 22:
            drift_id = f"drift_{int(self._last_voice_activity)}"
            if drift_id not in self._fired_shift_cues:
                self._fired_shift_cues.add(drift_id)
                self._bus.emit("speak", text="Still here.")
                log.info(f"Drift check after {int(silence/60)} min silence")

    # ── Mute ─────────────────────────────────────────────────────

    def _set_muted(self, muted: bool):
        self._muted = muted
        if muted:
            self._muted_at = time.time()
        self._bus.emit("mute_toggled", muted=muted)
        log.info(f"{'Muted' if muted else 'Unmuted'}")

    # ── State Persistence ────────────────────────────────────────

    def _persist_state(self):
        try:
            data = {
                "greeted_today": self._greeted_today,
                "greeting_date": str(self._greeting_date),
                "last_seen_time": self._last_seen_time,
                "last_face_lost_time": self._last_face_lost_time,
                "convo_phase": self._state_machine.phase.name,
            }
            config.STATE_PERSIST_PATH.write_text(json.dumps(data))
        except Exception:
            pass

    def _load_persisted_state(self):
        try:
            data = json.loads(config.STATE_PERSIST_PATH.read_text())
            saved_date = data.get("greeting_date", "")
            if saved_date == str(date.today()):
                self._greeted_today = data.get("greeted_today", False)
                self._greeting_date = date.today()
            self._last_seen_time = data.get("last_seen_time", 0.0)
            self._last_face_lost_time = data.get("last_face_lost_time", 0.0)
            # Restore conversation phase
            saved_phase = data.get("convo_phase", "IDLE")
            try:
                self._state_machine.phase = ConvoPhase[saved_phase]
            except KeyError:
                pass
            log.info(f"Loaded state: greeted={self._greeted_today}, phase={self._state_machine.phase.name}")
        except Exception:
            log.debug("No persisted state (clean start)")


# ── Standalone test ──────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[brain] %(message)s")

    bus = EventBus()
    bus.on("speak", lambda text="", **kw: print(f'\n>>> {config.BOT_NAME.upper()}: "{text}"\n'))

    brain = Brain()
    brain.start(bus)

    # Test conversation
    print(f"Type messages to {config.BOT_NAME} (prefix with 'Hey {config.BOT_NAME}' or just type after first response):")
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            bus.emit("speech", text=user_input, rms=200, duration=2.0)
            time.sleep(3)  # wait for LLM response
        except (KeyboardInterrupt, EOFError):
            break
    print("\nDone.")
