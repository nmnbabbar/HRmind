"""
backend/base/repository.py
==========================
VectorRepository Protocol — abstraction over vector store implementations.

Why a Protocol?
---------------
Dependency Inversion Principle: agents depend on this abstraction, not on
ChromaDB directly. Swap the vector store (Qdrant, Pinecone, Weaviate) by
implementing a new class that satisfies this Protocol — zero agent code changes.

Note on embeddings
------------------
This repository accepts PRE-COMPUTED embeddings (list[float]).
The agent is responsible for running the embedding model and passing vectors.
This separates the embedding concern from the storage concern cleanly.

Current implementation: ChromaVectorRepository (Phase 2)
Future-proof: any class with these three async methods satisfies the Protocol.
"""

from typing import Protocol, runtime_checkable

from langchain_core.documents import Document


@runtime_checkable
class VectorRepository(Protocol):
    """
    Async interface for vector store operations.

    All methods are async — implementations must use an async client
    (e.g. chromadb.AsyncHttpClient, not the synchronous client).
    """

    async def similarity_search(
        self,
        query_embedding: list[float],
        k: int,
    ) -> list[Document]:
        """
        Return the top-k most similar documents to the query embedding.

        Parameters
        ----------
        query_embedding : list[float]
            The embedding vector for the query (pre-computed by the caller).
        k : int
            Maximum number of documents to return.

        Returns
        -------
        list[Document]
            Documents in descending similarity order.
            Each Document has .page_content (str) and .metadata (dict).
        """
        ...

    async def add_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
        ids: list[str] | None = None,
    ) -> None:
        """
        Upsert documents with their pre-computed embeddings.

        Parameters
        ----------
        documents : list[Document]
            Documents to store.
        embeddings : list[list[float]]
            Embedding vectors aligned 1:1 with documents.
        ids : list[str] | None
            Optional stable IDs for upsert semantics.
            If None, IDs are generated from document content hash.
        """
        ...

    async def collection_count(self) -> int:
        """
        Return the total number of documents in the collection.

        Used to check whether ingestion has been run and to
        validate the index is non-empty before serving queries.
        """
        ...
