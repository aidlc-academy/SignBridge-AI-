"""
Unit tests for the prediction stabilization service.
"""
from __future__ import annotations

import time
import pytest
from app.services.stabilization_service import StabilizationService


def test_no_emission_below_window():
    svc = StabilizationService(window_size=10, required_votes=7, cooldown_seconds=0)
    for _ in range(9):
        result = svc.push("A")
    assert result is None   # haven't filled the window yet


def test_emits_on_majority():
    svc = StabilizationService(window_size=10, required_votes=7, cooldown_seconds=0)
    for _ in range(10):
        svc.push("A")
    # Should have emitted on the 10th push
    svc2 = StabilizationService(window_size=10, required_votes=7, cooldown_seconds=0)
    result = None
    for _ in range(10):
        result = svc2.push("A")
    assert result == "A"


def test_no_emission_without_majority():
    svc = StabilizationService(window_size=10, required_votes=7, cooldown_seconds=0)
    gestures = ["A"] * 5 + ["B"] * 5
    result = None
    for g in gestures:
        result = svc.push(g)
    assert result is None   # 5/10 < 7/10 required


def test_cooldown_suppresses_repeat():
    svc = StabilizationService(window_size=5, required_votes=4, cooldown_seconds=10)

    # First emission
    first = None
    for _ in range(5):
        first = svc.push("A")
    assert first == "A"

    # Immediately after — should be suppressed by cooldown
    second = None
    for _ in range(5):
        second = svc.push("A")
    assert second is None


def test_different_gesture_after_cooldown_emits():
    svc = StabilizationService(window_size=5, required_votes=4, cooldown_seconds=0)

    # First gesture
    for _ in range(5):
        svc.push("A")

    # Different gesture — should emit even with cooldown=0
    result = None
    for _ in range(5):
        result = svc.push("B")
    assert result == "B"


def test_reset_clears_state():
    svc = StabilizationService(window_size=5, required_votes=4, cooldown_seconds=0)
    for _ in range(5):
        svc.push("A")
    svc.reset()
    # After reset, should not emit without enough new data
    result = None
    for _ in range(4):
        result = svc.push("A")
    assert result is None


def test_none_push_does_not_emit():
    svc = StabilizationService(window_size=5, required_votes=4, cooldown_seconds=0)
    result = None
    for _ in range(10):
        result = svc.push(None)
    assert result is None
