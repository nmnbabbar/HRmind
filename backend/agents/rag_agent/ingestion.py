"""
backend/agents/rag_agent/ingestion.py
======================================
Async document ingestion pipeline for HR policy documents.

Design decisions
----------------
- Supports PDF (.pdf) and Word (.docx) files only.
- RecursiveCharacterTextSplitter (chunk_size=512, overlap=128) for precise
  citation-level granularity.
- Embedding model: BAAI/bge-large-en-v1.5 (335M params — highest BGE accuracy).
  Model is loaded ONCE and reused; wrapped in run_in_executor for async safety.
- Run-once persistence: a separate ChromaDB collection ("ingestion_log") records
  each ingested file by its SHA-256 content hash. On subsequent runs, only files
  whose hash is not present in the log are ingested. Files are NEVER re-ingested
  unless their content changes (hash changes).
- Incremental: adding new documents to the folder re-runs ingestion; only the
  new docs are processed. The vector DB is updated via upsert semantics (IDs
  are deterministic: file_hash + chunk_index).

Metadata stored per chunk (for citations)
------------------------------------------
    source          : str   — file basename, e.g. "Maternity-Policy.docx"
    source_path     : str   — absolute path
    page            : int   — 1-indexed page / section number
    chunk_index     : int   — 0-indexed chunk within the document
    total_chunks    : int   — total chunks in this document
    file_hash       : str   — SHA-256 of the raw file content
    doc_type        : str   — "pdf" | "docx"
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

from backend.config import get_settings
from backend.utils.chroma_client import create_chroma_client

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}

# Ingestion log collection — tracks which file hashes have been ingested
INGESTION_LOG_COLLECTION = "ingestion_log"

# Splitter config — larger chunks for better context preservation (especially bulleted lists)
CHUNK_SIZE = 1536
CHUNK_OVERLAP = 256

# BGE instruction prefix — improves retrieval accuracy for bge-large
BGE_PASSAGE_PREFIX = "Represent this sentence for searching relevant passages: "


# ── File hashing ──────────────────────────────────────────────────────────────

def compute_file_hash(file_path: Path) -> str:
    """Return the SHA-256 hex digest of a file's raw bytes."""
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Document loading ──────────────────────────────────────────────────────────

def load_document(file_path: Path) -> list[Document]:
    """
    Load a PDF or DOCX file and return a list of LangChain Documents.

    For PDFs: PyPDFLoader returns one Document per page — page number is
    embedded in metadata["page"] (0-indexed). We normalise to 1-indexed.

    For DOCX: Docx2txtLoader returns the entire document as one Document.
    We attach page=1 since Word docs have no native page boundary.

    Parameters
    ----------
    file_path : Path
        Absolute path to the document.

    Returns
    -------
    list[Document]
        Raw documents before chunking. Each has .page_content and .metadata.
    """
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
        # Normalise page to 1-indexed
        for doc in docs:
            doc.metadata["page"] = doc.metadata.get("page", 0) + 1
        return docs
    elif suffix == ".docx":
        loader = Docx2txtLoader(str(file_path))
        docs = loader.load()
        for doc in docs:
            doc.metadata["page"] = 1  # Word has no page boundary at load time
        return docs
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Supported: .pdf, .docx")


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_documents(
    docs: list[Document],
    file_path: Path,
    file_hash: str,
) -> list[Document]:
    """
    Split raw documents into chunks using RecursiveCharacterTextSplitter.

    Each chunk inherits the parent's page metadata and receives additional
    chunk-level metadata (chunk_index, total_chunks, etc.).

    The splitter tries to split on ["\n\n", "\n", " ", ""] in order,
    preserving paragraph / sentence / word / character boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        add_start_index=True,  # records start_index in metadata
    )

    suffix = file_path.suffix.lower()
    source_name = file_path.name

    all_chunks: list[Document] = []
    for doc in docs:
        page_num = doc.metadata.get("page", 1)
        raw_chunks = splitter.split_documents([doc])
        for idx, chunk in enumerate(raw_chunks):
            chunk.metadata.update(
                {
                    "source": source_name,
                    "source_path": str(file_path),
                    "page": page_num,
                    "chunk_index": idx,
                    "file_hash": file_hash,
                    "doc_type": suffix.lstrip("."),
                }
            )
            all_chunks.append(chunk)

    # Back-fill total_chunks now that we know the full count
    total = len(all_chunks)
    for chunk in all_chunks:
        chunk.metadata["total_chunks"] = total

    return all_chunks


# ── Deterministic chunk IDs ────────────────────────────────────────────────────

def make_chunk_id(file_hash: str, chunk_index: int) -> str:
    """
    Return a deterministic, stable ID for a chunk.

    Using file_hash + chunk_index means:
    - Same file content always produces the same IDs → upsert semantics
    - Different file content (modified doc) produces different IDs → new vectors
    """
    return f"{file_hash}_{chunk_index:05d}"


# ── Embedding ─────────────────────────────────────────────────────────────────

class EmbeddingService:
    """
    Thin wrapper around SentenceTransformer for async-safe usage.

    The SentenceTransformer.encode() call is CPU/GPU-bound and blocking.
    All public methods run it via asyncio.run_in_executor to avoid blocking
    the event loop.

    Model: BAAI/bge-large-en-v1.5
        - 335M parameters
        - MTEB leaderboard top-tier for retrieval tasks
        - Requires BGE-specific prefix for passage encoding
    """

    def __init__(self, model_name: str) -> None:
        logger.info("Loading embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name
        logger.info("Embedding model loaded.")

    def _encode_sync(
        self, texts: list[str], is_query: bool = False
    ) -> list[list[float]]:
        """Synchronous encode — runs in thread pool executor."""
        # BGE models benefit from instruction prefix for queries only
        if is_query and "bge" in self._model_name.lower():
            texts = [BGE_PASSAGE_PREFIX + t for t in texts]
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,   # cosine similarity via dot product
            batch_size=16,               # safe for CPU
            show_progress_bar=False,
        )
        return [emb.tolist() for emb in embeddings]

    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Embed document passages (no instruction prefix) — async wrapper."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._encode_sync, texts, False
        )

    def embed_passages_sync(self, texts: list[str]) -> list[list[float]]:
        """Synchronous embed for ingestion pipeline (no event loop needed)."""
        return self._encode_sync(texts, is_query=False)

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query with BGE instruction prefix — async wrapper."""
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, self._encode_sync, [text], True
        )
        return results[0]

    def embed_query_sync(self, text: str) -> list[float]:
        """Synchronous query embed for CLI tools."""
        return self._encode_sync([text], is_query=True)[0]


# ── Ingestion log (ChromaDB-backed) ───────────────────────────────────────────

class IngestionLog:
    """
    Tracks which files have been ingested using a ChromaDB collection.

    Each entry is stored as a ChromaDB document with:
        id          = file_hash
        document    = JSON-encoded metadata (filename, chunk_count, ingested_at)
        metadata    = {"source": filename}

    This lives in a SEPARATE collection ("ingestion_log") so it never
    interferes with the main vector search collection.

    Uses synchronous ChromaDB API (works with both PersistentClient and HttpClient).
    """

    def __init__(self, client: chromadb.ClientAPI) -> None:
        self._client = client
        self._collection: chromadb.Collection | None = None

    def _get_collection(self) -> chromadb.Collection:
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(
                name=INGESTION_LOG_COLLECTION,
                metadata={"purpose": "ingestion_tracking"},
            )
        return self._collection

    def is_ingested(self, file_hash: str) -> bool:
        """Return True if this file hash is already in the ingestion log."""
        col = self._get_collection()
        result = col.get(ids=[file_hash])
        return len(result["ids"]) > 0

    def mark_ingested(
        self, file_hash: str, filename: str, chunk_count: int
    ) -> None:
        """Record a file as successfully ingested."""
        import datetime

        col = self._get_collection()
        entry = json.dumps(
            {
                "source": filename,
                "chunk_count": chunk_count,
                "ingested_at": datetime.datetime.utcnow().isoformat(),
            }
        )
        col.upsert(
            ids=[file_hash],
            documents=[entry],
            metadatas=[{"source": filename}],
        )
        logger.info(
            "Ingestion log updated: %s (%d chunks)", filename, chunk_count
        )

    def get_all_ingested(self) -> dict[str, Any]:
        """Return all ingested file hashes → metadata."""
        col = self._get_collection()
        result = col.get()
        return {
            file_id: json.loads(doc)
            for file_id, doc in zip(result["ids"], result["documents"] or [])
        }


# ── Main ingestion pipeline ────────────────────────────────────────────────────

def ingest_documents(
    docs_path: Path | None = None,
    embedding_service: EmbeddingService | None = None,
) -> dict[str, int]:
    """
    Main entry point for the ingestion pipeline.

    Behaviour
    ----------
    1. Scan docs_path for .pdf and .docx files.
    2. For each file, compute SHA-256 hash.
    3. Check IngestionLog — skip if already ingested with the same hash.
    4. Load, chunk, embed, and upsert new/modified documents.
    5. Record completion in IngestionLog.

    This function is idempotent: running it multiple times is safe.
    Only genuinely new or modified files are processed.

    Returns
    -------
    dict[str, int]
        {"ingested": N, "skipped": M, "errors": K}
    """
    settings = get_settings()

    # Resolve defaults from settings
    docs_path = docs_path or Path(settings.hr_docs_path)
    collection_name = settings.chroma_collection_name

    logger.info(
        "Starting ingestion. docs_path=%s, chroma_mode=%s, collection=%s",
        docs_path,
        settings.chroma_mode,
        collection_name,
    )

    # ── Connect to ChromaDB (local or server) ────────────────────────────
    chroma_client: chromadb.ClientAPI = create_chroma_client(settings)
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    ingestion_log = IngestionLog(chroma_client)

    # ── Embedding service ──────────────────────────────────────────────────
    if embedding_service is None:
        embedding_service = EmbeddingService(settings.embedding_model)

    # ── Discover documents ─────────────────────────────────────────────────
    if not docs_path.exists():
        raise FileNotFoundError(f"docs_path does not exist: {docs_path}")

    all_files = [
        f
        for f in docs_path.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not all_files:
        logger.warning("No .pdf or .docx files found in %s", docs_path)
        return {"ingested": 0, "skipped": 0, "errors": 0}

    logger.info("Found %d documents to evaluate.", len(all_files))

    # ── Process each document ──────────────────────────────────────────────
    stats = {"ingested": 0, "skipped": 0, "errors": 0}

    for file_path in sorted(all_files):
        try:
            file_hash = compute_file_hash(file_path)

            # Check if already ingested
            if ingestion_log.is_ingested(file_hash):
                logger.debug("Skipping (already ingested): %s", file_path.name)
                stats["skipped"] += 1
                continue

            logger.info("Ingesting: %s", file_path.name)

            # Load raw pages/document
            raw_docs = load_document(file_path)

            # Chunk
            chunks = chunk_documents(raw_docs, file_path, file_hash)

            if not chunks:
                logger.warning("No chunks produced for %s — skipping.", file_path.name)
                stats["skipped"] += 1
                continue

            # Build chunk texts (content only — metadata is stored separately)
            texts = [c.page_content for c in chunks]
            metadatas = [c.metadata for c in chunks]
            ids = [make_chunk_id(file_hash, i) for i in range(len(chunks))]

            # Embed all chunks (synchronous — ingestion is a CLI script)
            embeddings = embedding_service.embed_passages_sync(texts)

            # Upsert into ChromaDB — sync collection
            batch_size = 100
            for start in range(0, len(chunks), batch_size):
                end = start + batch_size
                collection.upsert(
                    ids=ids[start:end],
                    documents=texts[start:end],
                    embeddings=embeddings[start:end],
                    metadatas=metadatas[start:end],
                )

            # Mark as ingested
            ingestion_log.mark_ingested(file_hash, file_path.name, len(chunks))

            stats["ingested"] += 1
            logger.info(
                "Ingested %s: %d chunks", file_path.name, len(chunks)
            )

        except Exception as exc:
            logger.error(
                "Error ingesting %s: %s", file_path.name, exc, exc_info=True
            )
            stats["errors"] += 1

    logger.info(
        "Ingestion complete. ingested=%d, skipped=%d, errors=%d",
        stats["ingested"],
        stats["skipped"],
        stats["errors"],
    )
    return stats


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    result = ingest_documents()
    print(f"\nIngestion result: {result}")
    sys.exit(0 if result["errors"] == 0 else 1)
