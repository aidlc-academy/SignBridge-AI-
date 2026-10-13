"""
Pydantic v2 request/response schemas for the SignBridge AI API.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ── Health ─────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    timestamp: datetime


# ── Model status ───────────────────────────────────────────────────────────

class ModelStatusResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    loaded: bool
    model_type: Optional[str]
    class_labels: list[str]
    trained_at: Optional[datetime]
    message: str


# ── Gestures ───────────────────────────────────────────────────────────────

class GestureVocabItem(BaseModel):
    id: str
    label: str
    description: str
    hand_shape: str
    language: str
    category: str


# ── Recognition ───────────────────────────────────────────────────────────

class RecognizeResponse(BaseModel):
    gesture: Optional[str]
    confidence: float = Field(ge=0.0, le=1.0)
    status: Literal[
        "recognized", "no_hand", "low_confidence",
        "model_unavailable", "error"
    ]
    timestamp: datetime
    message: str


# ── Messages ──────────────────────────────────────────────────────────────

class MessageCreate(BaseModel):
    content: str = Field(min_length=1, max_length=4096)
    source: Literal["sign_recognition", "speech_recognition", "manual"]
    language: str = Field(default="en", max_length=16)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class MessageOut(BaseModel):
    id: str
    conversation_id: str
    content: str
    source: str
    language: str
    confidence: Optional[float]
    timestamp: datetime

    model_config = {"from_attributes": True}


# ── Conversations ─────────────────────────────────────────────────────────

class ConversationCreate(BaseModel):
    session_id: str
    messages: list[MessageCreate] = Field(default_factory=list)


class ConversationOut(BaseModel):
    id: str
    session_id: str
    created_at: datetime
    updated_at: datetime
    messages: list[MessageOut] = Field(default_factory=list)

    model_config = {"from_attributes": True}


# ── Analytics ─────────────────────────────────────────────────────────────

class AnalyticsSummary(BaseModel):
    total_attempts: int
    accepted: int
    rejected: int
    avg_latency_ms: float
    gesture_distribution: dict[str, int]
    practice_completed: int



# ── Phase 3: Hand landmark detection response ──────────────────────────────

class LandmarkPoint(BaseModel):
    x: float
    y: float
    z: float


class HandLandmarks(BaseModel):
    handedness: Optional[str]  # "Left" or "Right" when available
    landmarks: list[LandmarkPoint]  # 21 landmarks


class LandmarkResponse(BaseModel):
    hands_detected: int
    hands: list[HandLandmarks]
    status: Literal["hand_detected", "no_hand", "error"]
    message: str
    latency_ms: int



# ── LSTM word-sign recognition (additional, independent method) ───────────

class SequenceResponse(BaseModel):
    """Result of feeding one frame into a per-session 30-frame LSTM window."""
    status: Literal[
        "collecting", "no_hand", "low_confidence",
        "recognized", "model_unavailable", "error",
    ]
    word: Optional[str]                       # current top word (only when recognized)
    confidence: float = Field(ge=0.0, le=1.0)  # probability of the top class (0 if not computed)
    confirmed_word: Optional[str]             # set on the single frame where a word is confirmed
    probabilities: Optional[dict[str, float]]  # real model output, one entry per word
    frames_buffered: int
    sequence_length: int
    hands_in_window: int
    window_seconds: Optional[float]           # wall-clock span of the buffered frames
    message: str
    session_id: str
    latency_ms: int
    method: Literal["lstm"] = "lstm"


class LstmStatusResponse(BaseModel):
    loaded: bool
    enabled: bool
    labels: list[str]
    sequence_length: int
    feature_size: Optional[int]
    parameters: Optional[int]
    confidence_threshold: float
    active_sessions: int
    message: str
