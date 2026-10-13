"""
SignBridge AI — FastAPI backend entry point.

Start with:
    uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

from datetime import datetime, timezone

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import APP_TITLE, APP_DESCRIPTION, API_VERSION, CORS_ORIGINS
from app.api.recognition   import router as recognition_router
from app.api.conversations import router as conversations_router
from app.api.analytics     import router as analytics_router
from app.services.lstm_service import shutdown_lstm_service
from database.connection   import init_db


# ── Lifespan ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(application: FastAPI):
    init_db()
    yield
    shutdown_lstm_service()   # release per-session MediaPipe Holistic graphs


# ── App ────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ───────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,         # never "*" in production
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)


# ── Routers ────────────────────────────────────────────────────────────────

app.include_router(recognition_router)
app.include_router(conversations_router)
app.include_router(analytics_router)


# ── Health ─────────────────────────────────────────────────────────────────

@app.get("/api/health", tags=["health"])
async def health():
    return {
        "status": "ok",
        "version": API_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
