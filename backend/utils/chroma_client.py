"""
backend/utils/chroma_client.py
================================
Factory for creating ChromaDB client instances.

Two modes controlled by CHROMA_MODE env var:

  "local"  → chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
              Data stored locally in ./chroma_data/
              No Docker, no server required — works out of the box for dev.

  "server" → chromadb.AsyncHttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
              Connects to a running ChromaDB HTTP server (Docker or remote).
              Used in production / docker-compose deployments.

Where is data stored?
----------------------
LOCAL mode:
    ./chroma_data/              ← project root (configurable via CHROMA_PERSIST_DIR)
    ├── chroma.sqlite3          ← ChromaDB metadata + collection index
    └── <uuid>/                 ← HNSW vector index segments (one dir per collection)
        ├── header.bin
        ├── data_level0.bin
        └── length.bin

SERVER mode (Docker):
    Inside the container: /chroma/chroma/
    On the host: Docker named volume "chroma_data"
    To find it:  docker volume inspect hrmind_chroma_data
"""

from __future__ import annotations

import logging

import chromadb

from backend.config import Settings, get_settings

logger = logging.getLogger(__name__)


def create_chroma_client(settings: Settings | None = None) -> chromadb.ClientAPI:
    """
    Create a synchronous ChromaDB client (for ingestion + local scripts).

    Returns PersistentClient in local mode, raises if server mode is requested
    (server mode requires async — use create_async_chroma_client instead).

    Parameters
    ----------
    settings : Settings | None
        Application settings. Defaults to get_settings().

    Returns
    -------
    chromadb.ClientAPI
        Synchronous ChromaDB client.
    """
    s = settings or get_settings()

    if s.chroma_mode == "local":
        import os
        persist_dir = os.path.abspath(s.chroma_persist_dir)
        logger.info("ChromaDB: local PersistentClient at %s", persist_dir)
        return chromadb.PersistentClient(path=persist_dir)
    else:
        # Synchronous HTTP client (for single-threaded scripts)
        logger.info("ChromaDB: HTTP client at %s:%s", s.chroma_host, s.chroma_port)
        return chromadb.HttpClient(host=s.chroma_host, port=s.chroma_port)


async def create_async_chroma_client(
    settings: Settings | None = None,
) -> chromadb.AsyncClientAPI:
    """
    Create an async ChromaDB client (for FastAPI / async contexts).

    In local mode, wraps PersistentClient in AsyncClientAPI.
    In server mode, uses AsyncHttpClient.

    Parameters
    ----------
    settings : Settings | None
        Application settings. Defaults to get_settings().

    Returns
    -------
    chromadb.AsyncClientAPI
        Async-compatible ChromaDB client.
    """
    s = settings or get_settings()

    if s.chroma_mode == "local":
        import os
        persist_dir = os.path.abspath(s.chroma_persist_dir)
        logger.info("ChromaDB: async local EphemeralClient (wrapped) at %s", persist_dir)
        # We return the synchronous PersistentClient. 
        # The ChromaVectorRepository uses _maybe_await to handle both sync and async clients gracefully.
        return chromadb.PersistentClient(path=persist_dir)
    else:
        logger.info(
            "ChromaDB: AsyncHttpClient at %s:%s", s.chroma_host, s.chroma_port
        )
        return await chromadb.AsyncHttpClient(
            host=s.chroma_host, port=s.chroma_port
        )
