"""
SQLAlchemy models for SignBridge AI.
Database: SQLite (file-backed via DATABASE_PATH in config).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column, String, Float, Boolean, Integer,
    ForeignKey, DateTime, Text, Index
)
from sqlalchemy.orm import DeclarativeBase, relationship


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class ConversationModel(Base):
    __tablename__ = "conversations"

    id         = Column(String(36), primary_key=True, default=_uuid)
    session_id = Column(String(36), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), default=_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=_now, onupdate=_now, nullable=False)

    messages = relationship(
        "MessageModel", back_populates="conversation",
        cascade="all, delete-orphan", order_by="MessageModel.timestamp"
    )


class MessageModel(Base):
    __tablename__ = "messages"

    id              = Column(String(36), primary_key=True, default=_uuid)
    conversation_id = Column(String(36), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True)
    content         = Column(Text, nullable=False)
    source          = Column(String(32), nullable=False)   # sign_recognition | speech_recognition | manual
    language        = Column(String(16), nullable=False, default="en")
    confidence      = Column(Float, nullable=True)
    timestamp       = Column(DateTime(timezone=True), default=_now, nullable=False)

    conversation = relationship("ConversationModel", back_populates="messages")


class RecognitionSessionModel(Base):
    __tablename__ = "recognition_sessions"

    id         = Column(String(36), primary_key=True, default=_uuid)
    started_at = Column(DateTime(timezone=True), default=_now, nullable=False)
    ended_at   = Column(DateTime(timezone=True), nullable=True)

    events = relationship(
        "RecognitionEventModel", back_populates="session",
        cascade="all, delete-orphan"
    )


class RecognitionEventModel(Base):
    __tablename__ = "recognition_events"

    id         = Column(String(36), primary_key=True, default=_uuid)
    session_id = Column(String(36), ForeignKey("recognition_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    gesture    = Column(String(32), nullable=False)
    confidence = Column(Float, nullable=False)
    accepted   = Column(Boolean, nullable=False)
    latency_ms = Column(Integer, nullable=False)
    timestamp  = Column(DateTime(timezone=True), default=_now, nullable=False)

    session = relationship("RecognitionSessionModel", back_populates="events")

    __table_args__ = (
        Index("ix_recognition_events_timestamp", "timestamp"),
    )


class PracticeProgressModel(Base):
    __tablename__ = "practice_progress"

    id         = Column(String(36), primary_key=True, default=_uuid)
    gesture_id = Column(String(32), nullable=False, index=True)
    attempts   = Column(Integer, default=0, nullable=False)
    successes  = Column(Integer, default=0, nullable=False)
    last_at    = Column(DateTime(timezone=True), nullable=True)
