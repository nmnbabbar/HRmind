"""
backend/agents/doc_parser_agent/cache.py
=========================================
In-memory extraction cache keyed by file SHA-256 hash.

Purpose:
    If the same file is uploaded again (same content, same hash), skip all
    LLM calls and return the cached ExtractionResult immediately.
    Zero cost, zero latency on repeat uploads.

Scope:
    In-memory dict — lives for the lifetime of the process.
    Suitable for single-worker Uvicorn (--workers 1, as in docker-compose).
    For multi-worker or multi-instance deployments, swap the _store dict
    for a Redis client without changing the public interface.

Cache key:
    SHA-256 hex digest of the raw file bytes (same hash as ingestion.py).
    Content-addressable: different files never collide, same file always hits.

TTL:
    No TTL on the cache itself — entries are cheap (small dict).
    Uploaded files are deleted after 1h, but the extraction result is kept
    in memory until process restart. This is intentional: follow-up uploads
    of the same document cost nothing.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def compute_file_hash(file_path: Path) -> str:
    """
    Return the SHA-256 hex digest of a file's raw bytes.

    Reads in 64KB chunks to avoid loading large files entirely into memory.
    Identical to the hash used in ingestion.py — consistent across the system.
    """
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class ExtractionCache:
    """
    Process-scoped in-memory cache for DocParser extraction results.

    Usage
    -----
        cache = ExtractionCache()                   # one instance per process
        file_hash = compute_file_hash(file_path)

        cached = cache.get(file_hash)
        if cached:
            return cached    # zero LLM calls

        result = await run_extraction(...)
        cache.set(file_hash, result)

    The cache stores the full structured_data dict (same format as
    AgentResult.structured_data), not the Pydantic model, for safe
    serialization and JSON compatibility.
    """

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}
        self._hits   = 0
        self._misses = 0

    def get(self, file_hash: str) -> dict[str, Any] | None:
        """
        Return cached extraction result for a file hash, or None on miss.
        """
        result = self._store.get(file_hash)
        if result is not None:
            self._hits += 1
            logger.info(
                "ExtractionCache: HIT (hash=%s...) — skipping LLM calls. "
                "Total hits: %d, misses: %d",
                file_hash[:12], self._hits, self._misses,
            )
        else:
            self._misses += 1
        return result

    def set(self, file_hash: str, result: dict[str, Any]) -> None:
        """
        Store an extraction result keyed by file hash.
        """
        self._store[file_hash] = result
        logger.debug(
            "ExtractionCache: stored result for hash=%s... (cache size: %d)",
            file_hash[:12], len(self._store),
        )

    def clear(self) -> None:
        """Clear the entire cache (useful for testing)."""
        self._store.clear()
        self._hits   = 0
        self._misses = 0

    @property
    def stats(self) -> dict[str, int]:
        return {
            "size":   len(self._store),
            "hits":   self._hits,
            "misses": self._misses,
        }


# ── Process-scoped singleton ───────────────────────────────────────────────────
# Instantiated once at module load. DocParserAgent imports this directly.
# Tests can call extraction_cache.clear() between runs for isolation.

extraction_cache = ExtractionCache()
