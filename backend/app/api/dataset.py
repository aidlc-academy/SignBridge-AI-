"""
Phase 5 — dataset collection endpoints.

POST /api/dataset/sample  — record ONE labeled landmark sample from a frame.
GET  /api/dataset/stats   — per-label sample counts for the collected dataset.

These build ``datasets/landmarks.csv``, the input consumed by
``python -m app.ml.train_model``. No sample is written unless MediaPipe
actually detects a hand, and labels are validated against the supported
vocabulary — so the dataset never contains fabricated or stray rows.
"""
from __future__ import annotations

from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.config import MAX_FRAME_SIZE_BYTES
from app.models.schemas import DatasetSampleResponse, DatasetStatsResponse
from app.services.dataset_service import get_dataset_service

router = APIRouter(prefix="/api/dataset", tags=["dataset"])


@router.post("/sample", response_model=DatasetSampleResponse)
async def add_sample(
    frame: UploadFile = File(..., description="JPEG/PNG frame from the browser webcam"),
    label: str = Form(..., description="Gesture label (e.g. 'A', 'V', '1')"),
):
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

    result = await run_in_threadpool(get_dataset_service().add_sample, label, data)
    return DatasetSampleResponse(**result)


@router.get("/stats", response_model=DatasetStatsResponse)
async def dataset_stats():
    return DatasetStatsResponse(**get_dataset_service().stats())
