"""
backend/api/main.py
====================
FastAPI application entry point.

Phase 1: Minimal placeholder with health endpoint only.
Full implementation in Phase 6 (routes, SSE streaming, file upload, middleware).

The lifespan context manager is already wired — Phase 6 will add:
- Embedding model load
- BM25 index build
- ChromaDB client connect
- Agent registry initialization
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings
from backend.utils.log import configure_logging, get_logger

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan — runs startup logic before accepting requests.

    Phase 1: logging only.
    Phase 6: will load embedding model, build BM25 index, connect ChromaDB.
    """
    logger.info("hrmind_starting", phase=1)
    yield
    logger.info("hrmind_shutdown")


app = FastAPI(
    title="HrMind API",
    description="Multi-agent HR intelligence platform",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Phase 6: restrict to frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", tags=["System"])
async def health() -> dict:
    """
    Health check endpoint.

    Phase 1: returns basic status.
    Phase 6: will add ChromaDB + LLM connectivity checks.
    """
    return {
        "status": "ok",
        "service": "hrmind-backend",
        "version": "0.1.0",
        "phase": 1,
    }
