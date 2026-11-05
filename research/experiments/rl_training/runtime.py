"""Completion-driven runtime policy with optional, explicit live controls.

This applies only to processes started with this implementation. It cannot alter
an already-running older trainer or an external infrastructure/budget deadline.
"""
import hashlib
import json
import math
from pathlib import Path
import signal
import time


class RuntimeLimit:
    def __init__(self, seconds=0, control_path=None, *, started=None, clock=None, poll_seconds=1):
        self.clock = clock or time.monotonic
        self.started = self.clock() if started is None else started
        self.seconds = self.validate_seconds(seconds)
        self.control_path = Path(control_path) if control_path else None
        self.poll_seconds = poll_seconds
        self.last_checked = float("-inf")
        self.last_signature = None
        self.last_error = None
        self.last_accepted_sha256 = None
        self.accepted_updates = 0
        self.stop_requested = False
        self.stop_reason = None

    @staticmethod
    def validate_seconds(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("max_seconds must be finite and nonnegative; zero disables the deadline")
        return float(value)

    def request_stop(self, reason="signal"):
        self.stop_requested = True
        if self.stop_reason is None or reason == "signal":
            self.stop_reason = reason

    def on_signal(self, signum, frame):
        self.request_stop("signal")

    def install_signal_handlers(self):
        signal.signal(signal.SIGTERM, self.on_signal)
        signal.signal(signal.SIGINT, self.on_signal)

    def refresh(self):
        if self.control_path is None or self.clock() - self.last_checked < self.poll_seconds:
            return
        self.last_checked = self.clock()
        try:
            stat = self.control_path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
            if signature == self.last_signature:
                return
            self.last_signature = signature
            if stat.st_size > 4096:
                raise ValueError("Runtime control file exceeds 4096 bytes")
            raw = self.control_path.read_bytes()
            value = json.loads(raw)
            if not isinstance(value, dict) or not value or set(value) - {"max_seconds", "stop"}:
                raise ValueError("Runtime control must contain only max_seconds and/or stop")
            seconds = self.validate_seconds(value["max_seconds"]) if "max_seconds" in value else self.seconds
            if "stop" in value and type(value["stop"]) is not bool:
                raise ValueError("Runtime control stop must be boolean")
            # Validate the entire update before changing any policy.
            self.seconds = seconds
            if value.get("stop"):
                self.request_stop("runtime_control")
            self.accepted_updates += 1
            self.last_error = None
            self.last_accepted_sha256 = hashlib.sha256(raw).hexdigest()
        except FileNotFoundError:
            pass  # Optional file may be created later; retain the last policy.
        except (OSError, ValueError) as exc:
            # Partial writes cannot silently disable a previously active limit.
            self.last_error = str(exc)

    @property
    def elapsed(self):
        return self.clock() - self.started

    @property
    def expired(self):
        self.refresh()
        return self.stop_requested or (self.seconds > 0 and self.elapsed >= self.seconds)

    @property
    def reason(self):
        return self.stop_reason if self.stop_requested else "time_limit"

    def summary(self):
        return {"effective_max_seconds": self.seconds, "deadline_enabled": self.seconds > 0,
                "control_path": str(self.control_path) if self.control_path else None,
                "accepted_control_updates": self.accepted_updates,
                "last_accepted_control_sha256": self.last_accepted_sha256,
                "last_control_error": self.last_error, "stop_requested": self.stop_requested,
                "stop_reason": self.stop_reason}
