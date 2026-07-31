import asyncio
import time
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings
from backend.utils.log import configure_logging, get_logger
from backend.api.routes import auth, chat, upload, eval, health, sessions


settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)

async def cleanup_old_uploads_task():
    """Background task to delete uploaded files older than TTL."""
    settings = get_settings()
    upload_dir = Path(settings.upload_dir)
    ttl_seconds = settings.upload_ttl_hours * 3600
    
    while True:
        try:
            if upload_dir.exists():
                now = time.time()
                for file_path in upload_dir.iterdir():
                    if file_path.is_file():
                        mtime = file_path.stat().st_mtime
                        if now - mtime > ttl_seconds:
                            try:
                                file_path.unlink()
                                logger.info(f"Deleted old upload: {file_path}")
                            except Exception as e:
                                logger.error(f"Failed to delete {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")
            
        await asyncio.sleep(1800) # Check every 30 mins

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan — runs startup logic before accepting requests.
    """
    logger.info("hrmind_starting", phase=6)
    
    # Start the background cleanup task
    cleanup_task = asyncio.create_task(cleanup_old_uploads_task())
    
    # Initialize AgentFactory heavy dependencies (RAG embeddings, ChromaDB, BM25)
    from backend.orchestration.factory import AgentFactory
    await AgentFactory.initialize()
    
    yield
    
    cleanup_task.cancel()
    logger.info("hrmind_shutdown")

app = FastAPI(
    title="HrMind API",
    description="Multi-agent HR intelligence platform",
    version="0.6.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router)
app.include_router(health.router)
app.include_router(chat.router)
app.include_router(upload.router)
app.include_router(eval.router)
app.include_router(sessions.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api.main:app", host="0.0.0.0", port=8000, reload=True)
