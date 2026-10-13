"""
/api/recognize  — accepts a JPEG frame and returns a gesture prediction.
/api/model/status — reports whether a trained model is loaded.
/api/gestures   — returns the supported vocabulary.
"""
from __future__ import annotations

import re
import time
from datetime import datetime, timezone

from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Depends
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from app.config import MAX_FRAME_SIZE_BYTES, DEFAULT_CONF_THRESHOLD, SUPPORTED_GESTURES
from app.models.schemas import (
    RecognizeResponse, ModelStatusResponse, GestureVocabItem, LandmarkResponse,
    HandLandmarks, LandmarkPoint, SequenceResponse, LstmStatusResponse,
)
from app.services.inference_service import get_inference_service
from app.services.lstm_service     import get_lstm_service
from app.services.vision_service    import get_vision_service

router = APIRouter(prefix="/api", tags=["recognition"])


@router.get("/model/status", response_model=ModelStatusResponse)
async def model_status():
    svc = get_inference_service()
    return ModelStatusResponse(
        loaded=svc.is_loaded,
        model_type=svc.model_type,
        class_labels=svc.class_labels,
        trained_at=svc.trained_at,
        message=(
            "Model loaded and ready."
            if svc.is_loaded
            else (
                "Recognition model not available. "
                "Please train or load a model (backend/app/ml/train_model.py)."
            )
        ),
    )


@router.get("/vision/status", tags=["recognition"])
async def vision_status():
    """
    Diagnostic endpoint: reports whether the OpenCV + MediaPipe vision stack
    initializes correctly (imports resolve and a MediaPipe Hands graph can be
    constructed). Does NOT require a trained model or a frame. Used by the
    frontend dev diagnostics and for integration health checks.
    """
    svc = get_vision_service()
    return svc.self_check()


@router.get("/gestures", response_model=list[GestureVocabItem])
async def get_gestures():
    return [
        GestureVocabItem(
            id=g["id"],
            label=g["label"],
            description=g["description"],
            hand_shape=g["handShape"],
            language=g["language"],
            category=g["category"],
        )
        for g in SUPPORTED_GESTURES
    ]


@router.post("/recognize", response_model=RecognizeResponse)
async def recognize(
    frame: UploadFile = File(..., description="JPEG frame from the browser webcam"),
):
    # Validate content type
    if frame.content_type not in ("image/jpeg", "image/jpg", "image/png"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type '{frame.content_type}'. Send JPEG or PNG.",
        )

    # Read and size-check
    data = await frame.read()
    if len(data) > MAX_FRAME_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Frame too large ({len(data)} bytes). Maximum is {MAX_FRAME_SIZE_BYTES} bytes.",
        )
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Empty frame received.")

    start = time.monotonic()

    # Vision pipeline
    vision_svc = get_vision_service()
    vision_result = vision_svc.process_frame(data)

    if vision_result["status"] == "no_hand":
        return RecognizeResponse(
            gesture=None,
            confidence=0.0,
            status="no_hand",
            timestamp=datetime.now(timezone.utc),
            message="No hand detected in frame.",
        )

    if vision_result["status"] == "error":
        return RecognizeResponse(
            gesture=None,
            confidence=0.0,
            status="error",
            timestamp=datetime.now(timezone.utc),
            message=vision_result["message"],
        )

    # Inference
    inference_svc = get_inference_service()
    result = inference_svc.predict(
        features=vision_result["features"],
        threshold=DEFAULT_CONF_THRESHOLD,
    )

    latency_ms = int((time.monotonic() - start) * 1000)

    return RecognizeResponse(
        gesture=result["gesture"],
        confidence=result["confidence"],
        status=result["status"],
        timestamp=result["timestamp"],
        message=f'{result["message"]} (latency: {latency_ms}ms)',
    )



@router.post("/recognize/landmarks", response_model=LandmarkResponse)
async def recognize_landmarks(
    frame: UploadFile = File(..., description="JPEG frame from the browser webcam"),
):
    """
    Phase 3 endpoint: returns raw hand landmark coordinates for frontend visualization.
    Does NOT require a trained model.
    """
    # Validate content type
    if frame.content_type not in ("image/jpeg", "image/jpg", "image/png"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type '{frame.content_type}'. Send JPEG or PNG.",
        )

    data = await frame.read()
    if len(data) > MAX_FRAME_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Frame too large ({len(data)} bytes). Maximum is {MAX_FRAME_SIZE_BYTES} bytes.",
        )
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Empty frame received.")

    start = time.monotonic()
    vision_svc = get_vision_service()
    result = await run_in_threadpool(vision_svc.process_frame_landmarks, data)
    latency_ms = int((time.monotonic() - start) * 1000)

    hands_out = [
        HandLandmarks(
            handedness=h["handedness"],
            landmarks=[LandmarkPoint(**lm) for lm in h["landmarks"]],
        )
        for h in result["hands"]
    ]

    return LandmarkResponse(
        hands_detected=result["hands_detected"],
        hands=hands_out,
        status=result["status"],
        message=result["message"],
        latency_ms=latency_ms,
    )



# ── LSTM word-sign recognition (additional method; independent of the above) ──

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@router.get("/lstm/status", response_model=LstmStatusResponse, tags=["lstm"])
async def lstm_status():
    """Whether the LSTM word model (hello / thanks / iloveyou) is loaded."""
    return LstmStatusResponse(**get_lstm_service().status())


@router.post("/recognize/sequence", response_model=SequenceResponse, tags=["lstm"])
async def recognize_sequence(
    frame: UploadFile = File(..., description="JPEG/PNG frame from the browser webcam"),
    session_id: str = Form("default", description="Identifies one camera session's 30-frame window"),
):
    """
    Feed ONE frame into this session's rolling 30-frame window and return the
    LSTM's current state. Call it repeatedly with consecutive frames; a word is
    only returned once 30 frames are buffered, a hand is visible, and the real
    model probability clears the confidence threshold.
    """
    if frame.content_type not in ("image/jpeg", "image/jpg", "image/png"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type '{frame.content_type}'. Send JPEG or PNG.",
        )
    if not _SESSION_ID_RE.match(session_id):
        raise HTTPException(status_code=400, detail="session_id must be 1-64 chars of [A-Za-z0-9_-].")

    data = await frame.read()
    if len(data) > MAX_FRAME_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Frame too large ({len(data)} bytes). Maximum is {MAX_FRAME_SIZE_BYTES} bytes.",
        )
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Empty frame received.")

    start = time.monotonic()
    result = await run_in_threadpool(get_lstm_service().process_frame, session_id, data)
    latency_ms = int((time.monotonic() - start) * 1000)
    return SequenceResponse(**result, latency_ms=latency_ms)


@router.post("/recognize/sequence/reset", tags=["lstm"])
async def reset_sequence(session_id: str = Form("default")):
    """Drop a session's buffered frames (e.g. when the camera is stopped)."""
    if not _SESSION_ID_RE.match(session_id):
        raise HTTPException(status_code=400, detail="session_id must be 1-64 chars of [A-Za-z0-9_-].")
    existed = await run_in_threadpool(get_lstm_service().reset_session, session_id)
    return {"session_id": session_id, "reset": existed}
