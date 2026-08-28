"""
events.py — Append-only event log for experiment provenance.

Every significant action is logged with timestamp, event type, and
relevant metadata. The log is append-only and atomically flushed.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from typing import Any, Dict, Optional


# Known event types
EVENT_TYPES = {
    "RUNTIME_STARTED",
    "RUNTIME_DISCONNECTED",
    "RUNTIME_RESUMED",
    "MANIFEST_LOADED",
    "MANIFEST_CREATED",
    "MANIFEST_CONFIG_MISMATCH",
    "CHECKPOINT_COMMITTED",
    "CHECKPOINT_VERIFIED",
    "CHECKPOINT_CORRUPTED",
    "CHECKPOINT_FALLBACK",
    "SEED_STARTED",
    "SEED_COMPLETED",
    "SEED_FAILED",
    "STAGE_STARTED",
    "STAGE_COMPLETED",
    "STAGE_FAILED",
    "EVALUATION_COMPLETED",
    "LENS_COMPLETED",
    "REPORT_GENERATED",
    "HF_UPLOAD_COMPLETE",
    "HF_UPLOAD_FAILED",
    "HF_DOWNLOAD_COMPLETE",
    "HF_DOWNLOAD_FAILED",
    "HEALTH_GATE_PASSED",
    "HEALTH_GATE_FAILED",
    "RECOVERY",
    "CONFIG_VERIFIED",
    "CONFIG_MISMATCH_DETECTED",
}


class EventLog:
    """Append-only event log for experiment provenance.

    Events are buffered in memory and flushed atomically to disk.
    Each flush writes the complete file (append-only at the OS level,
    but the file itself is rewritten atomically to prevent corruption).
    """

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        self._buffer: list = []
        self._load_existing()

    def _load_existing(self):
        """Load existing events from disk (if any)."""
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._buffer.append(json.loads(line))
        except (json.JSONDecodeError, OSError):
            pass

    def log(self, event: str, **kwargs):
        """Append an event to the log.

        Args:
            event: event type string (should be from EVENT_TYPES).
            **kwargs: arbitrary metadata fields.
        """
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event": event,
        }
        entry.update(kwargs)
        self._buffer.append(entry)

        # Auto-flush every 10 events to limit data loss on crash
        if len(self._buffer) % 10 == 0:
            self.flush()

    def flush(self) -> str:
        """Atomically write all events to disk."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        lines = "\n".join(json.dumps(e, default=str) for e in self._buffer) + "\n"
        content = lines.encode()

        fd, tmp = tempfile.mkstemp(
            dir=os.path.dirname(self.path), suffix=".tmp")
        try:
            os.write(fd, content)
            os.fsync(fd)
            os.close(fd)
            os.rename(tmp, self.path)
        except Exception:
            try:
                os.close(fd)
            except Exception:
                pass
            try:
                os.unlink(tmp)
            except Exception:
                pass
            raise

        return self.path

    @property
    def events(self) -> list:
        """All events (read-only copy)."""
        return list(self._buffer)

    def events_of_type(self, event_type: str) -> list:
        """Filter events by type."""
        return [e for e in self._buffer if e.get("event") == event_type]

    def last_event(self) -> Optional[Dict[str, Any]]:
        """Return the most recent event, or None."""
        return self._buffer[-1] if self._buffer else None

    def summary(self) -> Dict[str, int]:
        """Count events by type."""
        counts: Dict[str, int] = {}
        for e in self._buffer:
            t = e.get("event", "UNKNOWN")
            counts[t] = counts.get(t, 0) + 1
        return counts
