"""
backend/agents/rag_agent/hybrid_search.py
==========================================
Reciprocal Rank Fusion (RRF) for combining dense + sparse search results.

Why RRF?
---------
RRF is a well-established rank fusion algorithm that:
- Requires no score normalization (ranks are unit-free)
- Handles complementary signals gracefully (dense finds semantic matches,
  BM25 finds keyword/exact matches)
- Is O(N) and takes < 1ms for typical result sets
- Outperforms naive score-averaging for multi-modal retrieval

Formula: RRF_score(d) = Σ 1 / (k + rank(d, list_i))
    where k=60 (default) prevents high-ranked items from dominating.

Deduplication
-------------
The same chunk can appear in both dense and sparse results. Deduplication is
done by (source, page, chunk_index) — the chunk's identity in the corpus.
A chunk found in both lists gets a combined score (sum of both RRF contributions).

Citations
---------
Each returned Document's metadata contains:
    source       : str  — file basename (e.g. "Maternity-Policy.docx")
    page         : int  — 1-indexed page number
    chunk_index  : int  — position within the document
    rrf_score    : float — combined RRF score (for debugging)
    dense_rank   : int | None
    sparse_rank  : int | None
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ── RRF constant ──────────────────────────────────────────────────────────────
# k=60: standard default from the original RRF paper (Cormack et al., 2009)
# Higher k = smoother ranking; lower k = higher rank items dominate more
DEFAULT_RRF_K = 60


def _chunk_key(doc: Document) -> str:
    """
    Return a stable identity key for a chunk.

    Priority: use the stored chroma_id if present (most reliable).
    Fallback: source + page + chunk_index composite key.
    """
    meta = doc.metadata
    if "chroma_id" in meta:
        return meta["chroma_id"]
    # Fallback composite key
    source = meta.get("source", "unknown")
    page = meta.get("page", 0)
    chunk_index = meta.get("chunk_index", 0)
    return f"{source}::p{page}::c{chunk_index}"


def reciprocal_rank_fusion(
    dense_results: list[Document],
    sparse_results: list[Document],
    k: int = DEFAULT_RRF_K,
    final_top_k: int = 8,
) -> list[Document]:
    """
    Combine dense (semantic) and sparse (BM25) retrieval results using RRF.

    Parameters
    ----------
    dense_results : list[Document]
        Documents from ChromaDB semantic search, ordered by similarity (best first).
    sparse_results : list[Document]
        Documents from BM25 index, ordered by BM25 score (best first).
    k : int
        RRF smoothing constant. Default: 60 (standard).
    final_top_k : int
        Number of documents to return after fusion.

    Returns
    -------
    list[Document]
        Top final_top_k documents sorted by combined RRF score descending.
        Each Document has additional metadata:
            rrf_score  : float
            dense_rank : int | None  (1-indexed, None if not in dense results)
            sparse_rank: int | None  (1-indexed, None if not in sparse results)
    """
    # Map chunk_key → cumulative RRF score + rank tracking
    scores: dict[str, dict[str, Any]] = {}
    # Map chunk_key → Document object (for final assembly)
    doc_map: dict[str, Document] = {}

    # ── Score dense results ───────────────────────────────────────────────
    for rank_0, doc in enumerate(dense_results):
        rank = rank_0 + 1  # 1-indexed
        key = _chunk_key(doc)
        rrf_contribution = 1.0 / (k + rank)

        if key not in scores:
            scores[key] = {
                "rrf_score": 0.0,
                "dense_rank": None,
                "sparse_rank": None,
            }
            doc_map[key] = doc

        scores[key]["rrf_score"] += rrf_contribution
        scores[key]["dense_rank"] = rank

    # ── Score sparse results ──────────────────────────────────────────────
    for rank_0, doc in enumerate(sparse_results):
        rank = rank_0 + 1  # 1-indexed
        key = _chunk_key(doc)
        rrf_contribution = 1.0 / (k + rank)

        if key not in scores:
            scores[key] = {
                "rrf_score": 0.0,
                "dense_rank": None,
                "sparse_rank": None,
            }
            doc_map[key] = doc

        scores[key]["rrf_score"] += rrf_contribution
        scores[key]["sparse_rank"] = rank

    # ── Sort by combined RRF score ────────────────────────────────────────
    ranked_keys = sorted(
        scores.keys(),
        key=lambda k_: scores[k_]["rrf_score"],
        reverse=True,
    )

    # ── Build output documents ────────────────────────────────────────────
    final_docs: list[Document] = []
    for key in ranked_keys[:final_top_k]:
        base_doc = doc_map[key]
        score_info = scores[key]

        merged_doc = Document(
            page_content=base_doc.page_content,
            metadata={
                **base_doc.metadata,
                "rrf_score": round(score_info["rrf_score"], 6),
                "dense_rank": score_info["dense_rank"],
                "sparse_rank": score_info["sparse_rank"],
            },
        )
        final_docs.append(merged_doc)

    logger.debug(
        "RRF fusion: %d dense + %d sparse → %d unique → top %d returned",
        len(dense_results),
        len(sparse_results),
        len(scores),
        len(final_docs),
    )

    return final_docs


def format_citation(doc: Document) -> str:
    """
    Format a retrieved document as a citation string.

    Format: [Filename, page N]

    Parameters
    ----------
    doc : Document
        A chunk returned by RRF fusion (must have source + page in metadata).

    Returns
    -------
    str
        e.g. "[Maternity-Policy.docx, page 3]"
    """
    meta = doc.metadata
    source = meta.get("source", "Unknown document")
    page = meta.get("page", "?")
    return f"[{source}, page {page}]"


def format_context_with_citations(docs: list[Document]) -> str:
    """
    Build the context block injected into the RAG LLM prompt.

    Each chunk is presented with its citation reference so the LLM
    can include inline citations in its answer.

    Format:
        --- [Maternity-Policy.docx, page 3] ---
        <chunk text>

        --- [Notice-Periods-Policy.pdf, page 1] ---
        <chunk text>
        ...

    Parameters
    ----------
    docs : list[Document]
        Retrieved and fused chunks (RRF output).

    Returns
    -------
    str
        Formatted context string ready for LLM prompt injection.
    """
    parts: list[str] = []
    for doc in docs:
        citation = format_citation(doc)
        parts.append(f"--- {citation} ---\n{doc.page_content.strip()}")
    return "\n\n".join(parts)
