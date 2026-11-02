"""Frame-driven alert, recording and arming policies, independent of cameras and UI.

Only pass newly processed camera frames to :meth:`AlertPolicy.update`. A UI
timer or a cached result must never call it: elapsed time alone is not evidence
that a scene is still occupied. Long gaps between actual frames also restart
the confirmation and clearing windows.
"""

from __future__ import annotations

import math
from numbers import Real


class AlertPolicy:
    """Open an alert once per confirmed scene containing multiple faces.

    Dismissing an alert acknowledges the current incident. Another alert is
    eligible only after a continuous run of frames below ``face_threshold``
    lasting ``clear_seconds``. Clearing the scene does not close a visible
    alert; the user controls that window.

    ``now`` must use one monotonic clock for the lifetime of this instance.
    Call :meth:`reset` after stopping detection or on a camera error.
    """

    def __init__(
        self,
        confirm_seconds: float = 0.8,
        clear_seconds: float = 3.0,
        face_threshold: int = 2,
        max_frame_gap_seconds: float = 1.0,
    ) -> None:
        self.confirm_seconds = self._seconds("confirm_seconds", confirm_seconds)
        self.clear_seconds = self._seconds("clear_seconds", clear_seconds)
        self.max_frame_gap_seconds = self._seconds(
            "max_frame_gap_seconds", max_frame_gap_seconds, positive=True
        )
        if isinstance(face_threshold, bool) or not isinstance(face_threshold, int):
            raise ValueError("face_threshold must be an integer of at least 2")
        if face_threshold < 2:
            raise ValueError("face_threshold must be an integer of at least 2")
        self.face_threshold = face_threshold
        self.reset()

    @staticmethod
    def _seconds(name: str, value: float, *, positive: bool = False) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a finite number")
        result = float(value)
        if not math.isfinite(result) or result < 0 or (positive and result == 0):
            qualifier = "positive" if positive else "non-negative"
            raise ValueError(f"{name} must be a finite {qualifier} number")
        return result

    @property
    def alert_visible(self) -> bool:
        """Whether an alert has been opened and has not been dismissed."""
        return self._alert_visible

    @property
    def incident_latched(self) -> bool:
        """Whether the current incident has already been acknowledged."""
        return self._incident_latched

    def reset(self) -> None:
        """Clear visibility, acknowledgement, timers, and timestamp history."""
        self._alert_visible = False
        self._incident_latched = False
        self._above_since: float | None = None
        self._below_since: float | None = None
        self._last_time: float | None = None
        self._last_count: int | None = None

    def dismiss(self) -> None:
        """Close the alert and acknowledge any currently occupied incident.

        A visible alert can outlive its original incident. If that incident
        has already cleared, dismissing it while the scene is still clear
        leaves the policy armed for the next incident. If multiple faces
        have returned, dismissing acknowledges that current scene as well.
        """
        self._alert_visible = False
        if self._last_count is not None and self._last_count >= self.face_threshold:
            self._incident_latched = True
            self._above_since = None

    def update(self, face_count: int, now: float) -> bool:
        """Consume one fresh frame; return ``True`` only when UI should open.

        Invalid counts, non-finite timestamps, and timestamps moving backward
        raise ``ValueError`` without modifying the policy. Repeated timestamps
        are accepted, but add no confirmation or clearing time.
        """
        if isinstance(face_count, bool) or not isinstance(face_count, int):
            raise ValueError("face_count must be a non-negative integer")
        if face_count < 0:
            raise ValueError("face_count must be a non-negative integer")
        if isinstance(now, bool) or not isinstance(now, Real):
            raise ValueError("now must be a finite monotonic timestamp")
        timestamp = float(now)
        if not math.isfinite(timestamp):
            raise ValueError("now must be a finite monotonic timestamp")
        if self._last_time is not None and timestamp < self._last_time:
            raise ValueError("now must not move backward; reset for a new clock")

        if (
            self._last_time is not None
            and timestamp - self._last_time > self.max_frame_gap_seconds
        ):
            # Missing frames prove neither continued occupancy nor clearance.
            self._above_since = None
            self._below_since = None
        self._last_time = timestamp
        self._last_count = face_count

        if face_count < self.face_threshold:
            self._above_since = None
            if self._below_since is None:
                self._below_since = timestamp
            if timestamp - self._below_since >= self.clear_seconds:
                self._incident_latched = False
            return False

        self._below_since = None
        if self._incident_latched:
            return False
        if self._above_since is None:
            self._above_since = timestamp
        if (
            not self._alert_visible
            and timestamp - self._above_since >= self.confirm_seconds
        ):
            self._alert_visible = True
            self._incident_latched = True
            self._above_since = None
            return True
        return False


class HoldTrigger:
    """Debounce a per-frame condition for starting and ending recordings.

    The trigger turns on after the condition holds continuously for
    ``on_seconds`` and turns off after it is false continuously for
    ``off_seconds``. Like :class:`AlertPolicy`, only fresh frames may be fed,
    and a frame gap longer than ``max_frame_gap_seconds`` restarts both
    windows without changing the current state.
    """

    def __init__(
        self,
        on_seconds: float,
        off_seconds: float,
        max_frame_gap_seconds: float = 1.0,
    ) -> None:
        self.on_seconds = AlertPolicy._seconds("on_seconds", on_seconds)
        self.off_seconds = AlertPolicy._seconds("off_seconds", off_seconds)
        self.max_frame_gap_seconds = AlertPolicy._seconds(
            "max_frame_gap_seconds", max_frame_gap_seconds, positive=True
        )
        self.reset()

    @property
    def active(self) -> bool:
        return self._active

    def reset(self) -> None:
        self._active = False
        self._true_since: float | None = None
        self._false_since: float | None = None
        self._last_time: float | None = None

    def update(self, condition: bool, now: float) -> bool:
        """Consume one frame and return whether the trigger is now active."""
        if isinstance(now, bool) or not isinstance(now, Real) or not math.isfinite(float(now)):
            raise ValueError("now must be a finite monotonic timestamp")
        timestamp = float(now)
        if self._last_time is not None and timestamp < self._last_time:
            raise ValueError("now must not move backward; reset for a new clock")
        if (
            self._last_time is not None
            and timestamp - self._last_time > self.max_frame_gap_seconds
        ):
            self._true_since = None
            self._false_since = None
        self._last_time = timestamp
        if condition:
            self._false_since = None
            if self._true_since is None:
                self._true_since = timestamp
            if not self._active and timestamp - self._true_since >= self.on_seconds:
                self._active = True
        else:
            self._true_since = None
            if self._false_since is None:
                self._false_since = timestamp
            if self._active and timestamp - self._false_since >= self.off_seconds:
                self._active = False
        return self._active


class ArmState:
    """Whether privacy blackouts may open: armed, snoozed until a time, or off.

    Esc on a blackout disarms until the user arms again; any other key
    snoozes and re-arms automatically. ``now`` uses a monotonic clock.
    """

    ARMED = "armed"
    SNOOZED = "snoozed"
    DISARMED = "disarmed"

    def __init__(self) -> None:
        self.state = self.ARMED
        self._rearm_at: float | None = None

    @property
    def armed(self) -> bool:
        return self.state == self.ARMED

    def arm(self) -> None:
        self.state = self.ARMED
        self._rearm_at = None

    def disarm(self) -> None:
        self.state = self.DISARMED
        self._rearm_at = None

    def snooze(self, now: float, seconds: float) -> None:
        seconds = AlertPolicy._seconds("seconds", seconds)
        self.state = self.SNOOZED
        self._rearm_at = float(now) + seconds

    def remaining(self, now: float) -> float:
        if self.state != self.SNOOZED or self._rearm_at is None:
            return 0.0
        return max(0.0, self._rearm_at - float(now))

    def tick(self, now: float) -> bool:
        """Re-arm an expired snooze; return ``True`` only on that transition."""
        if self.state == self.SNOOZED and self._rearm_at is not None and now >= self._rearm_at:
            self.arm()
            return True
        return False
