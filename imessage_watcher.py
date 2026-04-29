"""Poll macOS Messages database for new inbound texts and emit bus events.

Requires Full Disk Access for the Python/terminal process that runs Merlin
(read-only access to ~/Library/Messages/chat.db).
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from pathlib import Path

import config
from event_bus import EventBus

log = logging.getLogger("merlin.imessage")

_BOOTSTRAP_SQL = """
SELECT MAX(m.ROWID) FROM message AS m
WHERE IFNULL(m.is_from_me, 0) = 0
  AND m.text IS NOT NULL
  AND LENGTH(TRIM(m.text)) > 0
"""

_NEW_MESSAGES_SQL = """
SELECT m.ROWID, m.text, IFNULL(h.id, '') AS sender
FROM message AS m
LEFT JOIN handle AS h ON m.handle_id = h.ROWID
WHERE m.ROWID > ?
  AND IFNULL(m.is_from_me, 0) = 0
  AND m.text IS NOT NULL
  AND LENGTH(TRIM(m.text)) > 0
ORDER BY m.ROWID ASC
"""


def _open_chat_db_readonly():
    path = Path(config.IMESSAGE_CHAT_DB).expanduser().resolve()
    if not path.is_file():
        return None
    try:
        uri = f"file:{path}?mode=ro"
        return sqlite3.connect(uri, uri=True, timeout=2.0, check_same_thread=False)
    except sqlite3.Error as e:
        log.debug("iMessage DB open failed: %s", e)
        return None


class IMessageWatcher:
    """Background poll for new inbound iMessages."""

    def __init__(self, bus: EventBus, interval_sec: float):
        self._bus = bus
        self._interval = max(1.0, float(interval_sec))
        self._stop = threading.Event()
        self._last_rowid: int | None = None
        self._thread: threading.Thread | None = None
        self._warned_access = False

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="imessage-watch")
        self._thread.start()
        log.info("iMessage watcher started (interval=%.0fs)", self._interval)

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        log.info("iMessage watcher stopped")

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._poll_once()
            except Exception:
                log.exception("iMessage poll error")
            self._stop.wait(self._interval)

    def _poll_once(self) -> None:
        conn = _open_chat_db_readonly()
        if conn is None:
            if not self._warned_access:
                log.warning(
                    "iMessage poll: cannot open chat.db (path=%s). "
                    "Grant Full Disk Access to your terminal/app, or set MERLIN_IMESSAGE_POLL_INTERVAL=0.",
                    config.IMESSAGE_CHAT_DB,
                )
                self._warned_access = True
            return

        try:
            cur = conn.cursor()
            if self._last_rowid is None:
                row = cur.execute(_BOOTSTRAP_SQL).fetchone()
                self._last_rowid = int(row[0] or 0)
                log.info("iMessage watcher baseline ROWID=%s (no announce)", self._last_rowid)
                return

            cur.execute(_NEW_MESSAGES_SQL, (self._last_rowid,))
            rows = cur.fetchall()
            for rowid, text, sender in rows:
                text = (text or "").strip()
                sender = (sender or "").strip() or "unknown"
                if len(text) < config.IMESSAGE_MIN_TEXT_LEN:
                    self._last_rowid = int(rowid)
                    continue
                log.info("New iMessage rowid=%s from=%s", rowid, sender[:40])
                self._bus.emit("imessage_received", text=text, sender=sender, rowid=int(rowid))
                self._last_rowid = int(rowid)
        except sqlite3.Error as e:
            if not self._warned_access:
                log.warning("iMessage poll query failed (permissions or schema): %s", e)
                self._warned_access = True
        finally:
            conn.close()


def start_imessage_watcher_if_enabled(bus: EventBus) -> IMessageWatcher | None:
    if config.IMESSAGE_POLL_INTERVAL <= 0:
        log.info("iMessage watcher disabled (MERLIN_IMESSAGE_POLL_INTERVAL=0)")
        return None
    w = IMessageWatcher(bus, float(config.IMESSAGE_POLL_INTERVAL))
    w.start()
    return w
