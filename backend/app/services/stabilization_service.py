"""
Prediction stabilization service.

Prevents the same gesture from being appended on every frame by using
a sliding window majority vote + cooldown.
"""
from __future__ import annotations

import time
from collections import Counter, deque
from typing import Optional


class StabilizationService:
    """
    Maintains a rolling window of recent predictions and emits a confirmed
    gesture only when:
      1. The majority-vote winner has appeared in ≥ `required_votes` of the
         last `window_size` frames.
      2. At least `cooldown_seconds` have elapsed since the last confirmation
         of the same gesture.

    This avoids flooding the output with the same letter every frame.
    """

    def __init__(
        self,
        window_size: int = 10,
        required_votes: int = 7,
        cooldown_seconds: float = 1.5,
    ) -> None:
        self.window_size      = window_size
        self.required_votes   = required_votes
        self.cooldown_seconds = cooldown_seconds

        self._window: deque[str] = deque(maxlen=window_size)
        self._last_confirmed: Optional[str]  = None
        self._last_confirmed_at: float       = 0.0

    def push(self, gesture: Optional[str]) -> Optional[str]:
        """
        Push a new raw prediction.
        Returns the confirmed gesture label if one should be emitted, else None.
        """
        if gesture is None:
            # No hand / low confidence — treat as empty frame
            self._window.append("")
            return None

        self._window.append(gesture)

        if len(self._window) < self.window_size:
            return None   # not enough data yet

        counts = Counter(g for g in self._window if g)
        if not counts:
            return None

        top_gesture, top_count = counts.most_common(1)[0]

        if top_count < self.required_votes:
            return None   # no clear majority

        now = time.monotonic()
        # Suppress if the same gesture was just confirmed (cooldown)
        if (
            top_gesture == self._last_confirmed
            and (now - self._last_confirmed_at) < self.cooldown_seconds
        ):
            return None

        # Confirm the gesture
        self._last_confirmed    = top_gesture
        self._last_confirmed_at = now
        self._window.clear()
        return top_gesture

    def reset(self) -> None:
        """Reset the window and confirmation state (e.g., on session start)."""
        self._window.clear()
        self._last_confirmed    = None
        self._last_confirmed_at = 0.0
