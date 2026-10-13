"""
/api/analytics  — aggregated recognition statistics from the database.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.database import RecognitionEventModel, PracticeProgressModel
from app.models.schemas import AnalyticsSummary
from database.connection import get_db

router = APIRouter(prefix="/api", tags=["analytics"])


@router.get("/analytics", response_model=AnalyticsSummary)
def get_analytics(db: Session = Depends(get_db)):
    # Attempt counts
    total = db.query(func.count(RecognitionEventModel.id)).scalar() or 0
    accepted = (
        db.query(func.count(RecognitionEventModel.id))
        .filter(RecognitionEventModel.accepted.is_(True))
        .scalar() or 0
    )
    rejected = total - accepted

    # Average latency
    avg_latency = (
        db.query(func.avg(RecognitionEventModel.latency_ms)).scalar() or 0.0
    )

    # Gesture distribution (accepted only)
    distribution_rows = (
        db.query(RecognitionEventModel.gesture, func.count(RecognitionEventModel.id))
        .filter(RecognitionEventModel.accepted.is_(True))
        .group_by(RecognitionEventModel.gesture)
        .all()
    )
    gesture_distribution = {gesture: count for gesture, count in distribution_rows}

    # Practice completions (sessions where all successes ≥ 1)
    practice_completed = (
        db.query(func.count(PracticeProgressModel.id))
        .filter(PracticeProgressModel.successes >= 1)
        .scalar() or 0
    )

    return AnalyticsSummary(
        total_attempts=total,
        accepted=accepted,
        rejected=rejected,
        avg_latency_ms=float(avg_latency),
        gesture_distribution=gesture_distribution,
        practice_completed=practice_completed,
    )
