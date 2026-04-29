"""Merlin v2 — Brain: Intent-aware LLM conversation + EF prosthetic modes."""

from __future__ import annotations

import collections
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

log = logging.getLogger("merlin.brain")


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
        r"^date", r"^time", r"^timer", r"^look\b", r"^scan\b", r"^pan\b",
        r"\b(what('s| is)\s+the\s+time|current\s+time|time\s+now)\b",
        r"\b(what('s| is)\s+the\s+date|today'?s\s+date|current\s+date)\b",
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
    return f"""{config.BOT_OPERATOR} asked a question. Answer directly and concisely.
For yes/no questions, start your first word with exactly "Yes." or "No.".
For true/false questions, start your first word with exactly "True." or "False.".
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
    return "Respond naturally. Brief. Under 30 words."


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
    Intent.QUESTION: 200,
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

        # Load persisted state
        self._load_persisted_state()

        # Initial context load
        self._refresh_context()
        log.info("Brain started (intent-aware v2)")

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

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── Event Handlers ───────────────────────────────────────────

    def _on_speech(self, text: str = "", rms: float = 0, duration: float = 0, **kw) -> None:
        """Handle transcribed speech — intent-aware v2."""
        if not text:
            return

        text_lower = text.lower().strip()
        self._last_voice_activity = time.time()

        # 0. Echo detection
        if self._last_spoken:
            similarity = SequenceMatcher(None, text_lower, self._last_spoken.lower()).ratio()
            if similarity > 0.5:
                log.debug(f"Echo detected (similarity={similarity:.2f}), ignoring: {text[:50]}")
                return

        # 1. Check mute
        if self._muted:
            if any(text_lower.startswith(w) for w in config.WAKE_WORDS):
                self._set_muted(False)
                log.info("Unmuted via wake word")
            else:
                return

        # 2. Conversation controls
        if any(w in text_lower for w in config.NEVERMIND_WORDS):
            self._last_response_time = 0
            self._bus.emit("speak_nonverbal", sound="close")
            log.info("Conversation closed (nevermind)")
            return

        if any(w in text_lower for w in config.MUTE_WORDS):
            self._set_muted(True)
            self._bus.emit("speak_nonverbal", sound="close")
            return

        if any(w in text_lower for w in config.UNMUTE_WORDS):
            self._set_muted(False)
            self._bus.emit("speak_nonverbal", sound="open")
            return

        # 3. Wake word check
        has_wake = any(text_lower.startswith(w) for w in config.WAKE_WORDS) or \
                   any(w in text_lower for w in config.WAKE_WORDS)
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

        # 5. COMMAND short-circuit — no LLM needed
        if intent == Intent.COMMAND:
            response = handle_command(message, self._bus)
            if response:
                self._last_spoken = response
                self._bus.emit("speak", text=response)
                self._last_response_time = time.time()
            return

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

    # ── LLM ──────────────────────────────────────────────────────

    def _think(self, message: str, intent: Intent = Intent.GENERAL, phase: ConvoPhase = ConvoPhase.IDLE) -> str | None:
        """Send message to Ollama with intent-specific prompting."""
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

        try:
            # Vision context is included as text in the system prompt via
            # scene_update events from vision.py. No inline image needed —
            # vision.py pre-computes descriptions in the background.

            resp = requests.post(config.LLM_URL, json={
                "model": config.LLM_MODEL,
                "messages": messages,
                "stream": False,
                "temperature": 0.5,
                "max_tokens": max_tokens,
            }, timeout=60)

            if resp.status_code == 200:
                text = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                # Strip thinking tags
                text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
                text = re.sub(r'<\|channel>thought.*?<channel\|>', '', text, flags=re.DOTALL).strip()
                if text:
                    self._history.append({"user": message, "assistant": text})
                    log.info(f"[{intent.name}] Response: {text}")
                return text or None
            else:
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
