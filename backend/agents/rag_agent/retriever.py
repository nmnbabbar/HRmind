"""
backend/agents/rag_agent/retriever.py
=======================================
ChromaDB vector repository implementation + BM25 sparse index builder.

ChromaVectorRepository
-----------------------
Implements VectorRepository protocol. Uses ChromaDB's async HTTP client.
Accepts pre-computed embeddings (caller is responsible for embedding the query).

BM25IndexBuilder
-----------------
Loads ALL chunk texts from ChromaDB at startup and builds a BM25Okapi index
in memory. The BM25 index is rebuilt on startup — no persistence needed since
ChromaDB is the source of truth. Rebuild is fast (< 2s for ~1000 chunks).

Both objects are created ONCE at application startup and held in app.state.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

import chromadb
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from backend.base.repository import VectorRepository
from backend.config import get_settings

logger = logging.getLogger(__name__)


# ── ChromaDB vector repository ────────────────────────────────────────────────

class ChromaVectorRepository(VectorRepository):
    """
    Async vector repository backed by ChromaDB HTTP client.

    Implements VectorRepository protocol:
        - similarity_search(query_embedding, k) → list[Document]
        - add_documents(documents, embeddings, ids) → None
        - collection_count() → int

    The repository is responsible ONLY for storage and retrieval.
    Embedding computation happens in the caller (EmbeddingService / RAGAgent).
    """

    def __init__(
        self,
        client: chromadb.AsyncClientAPI,
        collection_name: str,
    ) -> None:
        self._client = client
        self._collection_name = collection_name
        self._collection: chromadb.AsyncCollection | None = None

    async def _get_collection(self) -> chromadb.AsyncCollection:
        """Lazily fetch the collection (created during ingestion)."""
        if self._collection is None:
            self._collection = await self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    async def similarity_search(
        self,
        query_embedding: list[float],
        k: int,
    ) -> list[Document]:
        """
        Return top-k most similar documents using cosine distance.

        Parameters
        ----------
        query_embedding : list[float]
            Pre-computed query embedding (normalized, for cosine similarity).
        k : int
            Number of results to return.

        Returns
        -------
        list[Document]
            Each Document has .page_content (chunk text) and .metadata dict.
            Metadata includes: source, page, chunk_index, file_hash, doc_type.
            An extra key "distance" is added (lower = more similar for cosine).
        """
        collection = await self._get_collection()

        results = await collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, await self.collection_count()),
            include=["documents", "metadatas", "distances"],
        )

        docs: list[Document] = []
        if not results["documents"] or not results["documents"][0]:
            return docs

        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            doc = Document(
                page_content=text,
                metadata={
                    **(meta or {}),
                    "distance": dist,
                    "score": 1.0 - dist,  # convert cosine distance to similarity
                },
            )
            docs.append(doc)

        return docs

    async def add_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
        ids: list[str] | None = None,
    ) -> None:
        """Upsert documents with pre-computed embeddings into ChromaDB."""
        collection = await self._get_collection()
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]

        if ids is None:
            import hashlib
            ids = [
                hashlib.md5(t.encode()).hexdigest()[:16] for t in texts
            ]

        await collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    async def collection_count(self) -> int:
        """Return total number of documents in the collection."""
        collection = await self._get_collection()
        return await collection.count()

    async def get_all_chunks(self) -> list[Document]:
        """
        Return ALL stored chunks — used to build the BM25 index at startup.

        Warning: loads everything into memory. For HR docs this is fine
        (< 5000 chunks typically), but would need pagination for very large corpora.
        """
        collection = await self._get_collection()
        count = await collection.count()
        if count == 0:
            return []

        results = await collection.get(
            limit=count,
            include=["documents", "metadatas"],
        )

        docs: list[Document] = []
        for doc_id, text, meta in zip(
            results["ids"],
            results["documents"] or [],
            results["metadatas"] or [],
        ):
            docs.append(
                Document(
                    page_content=text,
                    metadata={**(meta or {}), "chroma_id": doc_id},
                )
            )
        return docs


# ── BM25 sparse index ──────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """
    Simple whitespace + lowercase tokenizer for BM25.

    Splits on non-alphanumeric characters, lowercases, removes empty tokens.
    This is intentionally simple — BM25 performs well with basic tokenization
    and adding stop-word removal risks losing policy-critical terms.
    """
    tokens = re.split(r"[^a-zA-Z0-9']+", text.lower())
    return [t for t in tokens if t]


class BM25Index:
    """
    In-memory BM25Okapi index over all ChromaDB chunks.

    Built once at startup from the full set of stored chunks.
    The index maps BM25 rank positions back to Document objects so the
    hybrid search function can return rich Documents (with metadata)
    rather than just scores.

    Usage
    -----
        index = BM25Index(all_chunks)
        top_docs = index.get_top_n(query, n=20)
    """

    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents
        if documents:
            tokenized = [_tokenize(doc.page_content) for doc in documents]
            self._bm25: BM25Okapi | None = BM25Okapi(tokenized)
        else:
            self._bm25 = None
        logger.info("BM25 index built with %d documents.", len(documents))

    def get_top_n(self, query: str, n: int = 20) -> list[Document]:
        """
        Return the top-n documents scored by BM25.

        Parameters
        ----------
        query : str
            Raw natural-language query (will be tokenized).
        n : int
            Number of results to return.

        Returns
        -------
        list[Document]
            Sorted by BM25 score descending. Each Document has .metadata["bm25_score"].
        """
        if not self._documents:
            return []

        query_tokens = _tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        # Sort by score descending, take top-n
        top_n = min(n, len(self._documents))
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]

        results: list[Document] = []
        for idx in ranked_indices:
            doc = self._documents[idx]
            doc_copy = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "bm25_score": float(scores[idx])},
            )
            results.append(doc_copy)
        return results

    @property
    def document_count(self) -> int:
        return len(self._documents)


async def build_bm25_index(vector_repo: ChromaVectorRepository) -> BM25Index:
    """
    Load all chunks from ChromaDB and build a BM25Index.

    Called once at application startup (FastAPI lifespan).
    The BM25 index is held in memory and serves all incoming queries.

    Returns
    -------
    BM25Index
        Ready-to-query BM25 index.
    """
    logger.info("Loading all chunks from ChromaDB for BM25 index...")
    all_chunks = await vector_repo.get_all_chunks()

    if not all_chunks:
        logger.warning(
            "ChromaDB collection is empty — BM25 index will have 0 documents. "
            "Run ingestion first: python -m backend.agents.rag_agent.ingestion"
        )

    return BM25Index(all_chunks)


# ── Factory ────────────────────────────────────────────────────────────────────

async def create_chroma_repository(
    host: str | None = None,
    port: int | None = None,
    collection_name: str | None = None,
) -> ChromaVectorRepository:
    """
    Factory: create a ChromaVectorRepository from settings.

    Resolves defaults from get_settings() if not provided.
    """
    settings = get_settings()
    client = await chromadb.AsyncHttpClient(
        host=host or settings.chroma_host,
        port=port or settings.chroma_port,
    )
    return ChromaVectorRepository(
        client=client,
        collection_name=collection_name or settings.chroma_collection_name,
    )
