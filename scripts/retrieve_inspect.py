"""
scripts/retrieve_inspect.py
============================
Interactive CLI tool to inspect the RAG retrieval pipeline.

Ask a question and see exactly what context would be injected into
the LLM prompt — without making any LLM calls.

This is useful for:
- Verifying that ingestion worked correctly
- Debugging retrieval quality (wrong chunks? missing docs?)
- Manually calibrating top_k and RRF settings
- Understanding citations before running full evaluations

Usage
-----
    uv run python scripts/retrieve_inspect.py

    # One-shot (non-interactive):
    uv run python scripts/retrieve_inspect.py --query "What is the maternity leave policy?"

    # Show more chunks:
    uv run python scripts/retrieve_inspect.py --top-k 12

    # Show raw chunk content + scores only (no prompt wrapper):
    uv run python scripts/retrieve_inspect.py --raw

Requires
--------
- ChromaDB running with ingested documents
  (Run: uv run python -m backend.agents.rag_agent.ingestion)
- BAAI/bge-large-en-v1.5 will be downloaded on first run (~1.3GB)
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

# Ensure project root is on the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def _separator(char: str = "─", width: int = 72) -> str:
    return char * width


def _print_header(text: str) -> None:
    print(f"\n{_separator('═')}")
    print(f"  {text}")
    print(_separator('═'))


def _print_chunk(idx: int, doc, show_scores: bool = True) -> None:
    """Pretty-print a single retrieved chunk with its metadata and scores."""
    meta = doc.metadata
    source = meta.get("source", "unknown")
    page = meta.get("page", "?")
    chunk_index = meta.get("chunk_index", "?")
    total_chunks = meta.get("total_chunks", "?")

    # Scores
    rrf_score = meta.get("rrf_score", None)
    dense_rank = meta.get("dense_rank", None)
    sparse_rank = meta.get("sparse_rank", None)

    print(f"\n{'─' * 72}")
    print(f"  Chunk {idx + 1}  │  {source}, page {page}  │  chunk {chunk_index}/{total_chunks}")
    if show_scores and rrf_score is not None:
        dense_str = f"dense rank #{dense_rank}" if dense_rank else "not in dense"
        sparse_str = f"BM25 rank #{sparse_rank}" if sparse_rank else "not in BM25"
        print(f"           │  RRF score: {rrf_score:.6f}  │  {dense_str}  │  {sparse_str}")
    print(f"{'─' * 72}")

    # Word-wrap the chunk content
    wrapped = textwrap.fill(
        doc.page_content.strip(),
        width=70,
        initial_indent="  ",
        subsequent_indent="  ",
    )
    print(wrapped)


def _print_prompt_context(docs) -> None:
    """Print the exact context block that would be sent to the LLM."""
    from backend.agents.rag_agent.hybrid_search import format_context_with_citations
    from backend.agents.rag_agent.context_builder import RAG_SYSTEM_PROMPT

    context_str = format_context_with_citations(docs)
    system_prompt = RAG_SYSTEM_PROMPT.format(context=context_str)

    print(f"\n{'═' * 72}")
    print("  EXACT PROMPT CONTEXT (what the LLM would see)")
    print(f"{'═' * 72}")
    print()

    # Print with line wrapping
    for line in system_prompt.split("\n"):
        if line.strip():
            print(textwrap.fill(line, width=70, subsequent_indent="    "))
        else:
            print()


def run_retrieval(
    query: str,
    repo,
    embedder,
    bm25_index,
    settings,
    top_k: int,
    show_scores: bool,
    show_prompt: bool,
) -> None:
    """Execute hybrid retrieval and display results."""
    from backend.agents.rag_agent.hybrid_search import (
        format_citation,
        reciprocal_rank_fusion,
    )

    print(f"\n{'─' * 72}")
    print(f"  Query: {query}")
    print(f"  Mode: {settings.chroma_mode.upper()} ChromaDB", end="")
    if settings.chroma_mode == "local":
        print(f" ({settings.chroma_persist_dir})")
    else:
        print(f" ({settings.chroma_host}:{settings.chroma_port})")
    print(f"  Embedding: {settings.embedding_model}")
    print(f"  top_k: dense={settings.dense_top_k}, sparse={settings.sparse_top_k}, final={top_k}")
    print(f"{'─' * 72}")

    # ── Hybrid retrieval ───────────────────────────────────────────────────
    print("\n[Running hybrid retrieval...]")

    # Embed query (synchronous)
    query_embedding = embedder.embed_query_sync(query)

    # Dense search
    dense_results = repo.similarity_search(query_embedding, k=settings.dense_top_k)
    print(f"      Dense  : {len(dense_results)} candidates")

    # Sparse BM25
    sparse_results = bm25_index.get_top_n(query, n=settings.sparse_top_k)
    print(f"      Sparse : {len(sparse_results)} candidates")

    # RRF fusion
    fused = reciprocal_rank_fusion(
        dense_results=dense_results,
        sparse_results=sparse_results,
        k=settings.rrf_k,
        final_top_k=top_k,
    )
    print(f"      Fused  : {len(fused)} chunks after RRF (k={settings.rrf_k})")

    # ── Display results ────────────────────────────────────────────────────
    _print_header(f"RETRIEVED CONTEXT — {len(fused)} chunks")

    for i, doc in enumerate(fused):
        _print_chunk(i, doc, show_scores=show_scores)

    # Citations summary
    print(f"\n{'─' * 72}")
    print("  CITATIONS SUMMARY")
    print(f"{'─' * 72}")
    seen = set()
    for doc in fused:
        citation = format_citation(doc)
        if citation not in seen:
            seen.add(citation)
            print(f"  → {citation}")

    # Full prompt preview
    if show_prompt:
        _print_prompt_context(fused)
    else:
        print(f"\n  Tip: run with --prompt to see the full system prompt injected into the LLM.")


def setup_pipeline(settings):
    """Initialize ChromaDB, load embedding model, and build BM25 index."""
    from backend.agents.rag_agent.ingestion import EmbeddingService
    from backend.agents.rag_agent.retriever import BM25Index
    from backend.utils.chroma_client import create_chroma_client

    print("\n[1/3] Connecting to ChromaDB...")
    try:
        client = create_chroma_client(settings)
        collection = client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        count = collection.count()
        print(f"      ✓ Connected. Collection '{settings.chroma_collection_name}' has {count:,} chunks.")
    except Exception as e:
        print(f"\n  ✗ ChromaDB connection failed: {e}")
        print(f"\n  If using local mode, run ingestion first:")
        print(f"    uv run python -m backend.agents.rag_agent.ingestion")
        sys.exit(1)

    if count == 0:
        print("\n  ✗ Collection is empty. Run ingestion first:")
        print("    uv run python -m backend.agents.rag_agent.ingestion")
        sys.exit(1)

    class SyncChromaRepo:
        def similarity_search(self, embedding: list[float], k: int):
            from langchain_core.documents import Document
            results = collection.query(
                query_embeddings=[embedding],
                n_results=min(k, count),
                include=["documents", "metadatas", "distances"],
            )
            docs = []
            if results["documents"] and results["documents"][0]:
                for text, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                ):
                    docs.append(Document(
                        page_content=text,
                        metadata={**(meta or {}), "distance": dist, "score": 1.0 - dist},
                    ))
            return docs
    repo = SyncChromaRepo()

    print("\n[2/3] Loading embedding model (first run downloads ~1.3GB)...")
    embedder = EmbeddingService(settings.embedding_model)
    print(f"      ✓ {settings.embedding_model} ready.")

    print("\n[3/3] Building BM25 index...")
    from langchain_core.documents import Document as LC_Document
    all_results = collection.get(
        limit=count,
        include=["documents", "metadatas"],
    )
    all_docs = [
        LC_Document(
            page_content=text,
            metadata={**(meta or {}), "chroma_id": doc_id},
        )
        for doc_id, text, meta in zip(
            all_results["ids"],
            all_results["documents"] or [],
            all_results["metadatas"] or [],
        )
    ]
    bm25_index = BM25Index(all_docs)
    print(f"      ✓ BM25 index built with {bm25_index.document_count:,} chunks.")

    return repo, embedder, bm25_index


def interactive_loop(repo, embedder, bm25_index, settings, top_k: int, show_scores: bool, show_prompt: bool) -> None:
    """REPL loop — ask questions until the user quits."""
    _print_header("HrMind Retrieval Inspector  (type 'quit' or Ctrl+C to exit)")
    print("  This tool shows the exact context chunks that would be injected")
    print("  into the LLM prompt — no LLM calls are made.\n")

    while True:
        try:
            query = input("  ❓ Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Goodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("\n  Goodbye!")
            break

        try:
            run_retrieval(
                query,
                repo=repo,
                embedder=embedder,
                bm25_index=bm25_index,
                settings=settings,
                top_k=top_k,
                show_scores=show_scores,
                show_prompt=show_prompt
            )
        except Exception as e:
            print(f"\n  ✗ Error: {e}")

        print()


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HrMind RAG Retrieval Inspector — see exactly what context the LLM would receive.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          # Interactive REPL
          uv run python scripts/retrieve_inspect.py

          # One-shot query
          uv run python scripts/retrieve_inspect.py --query "What is the maternity leave policy?"

          # Show more chunks and the full prompt
          uv run python scripts/retrieve_inspect.py --top-k 12 --prompt

          # Hide retrieval scores (cleaner output)
          uv run python scripts/retrieve_inspect.py --no-scores
        """),
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Run a single query and exit (non-interactive mode).",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=None,
        help="Number of final chunks to return after RRF fusion. Default: from settings (8).",
    )
    parser.add_argument(
        "--prompt",
        action="store_true",
        default=False,
        help="Print the full system prompt exactly as the LLM would receive it.",
    )
    parser.add_argument(
        "--no-scores",
        action="store_true",
        default=False,
        help="Hide RRF/BM25/dense rank scores (cleaner output).",
    )
    args = parser.parse_args()

    # Resolve top_k: CLI arg → settings default
    from backend.config import get_settings
    _settings = get_settings()
    top_k = args.top_k or _settings.final_top_k
    show_scores = not args.no_scores

    if args.query:
        repo, embedder, bm25_index = setup_pipeline(_settings)
        # One-shot mode
        run_retrieval(
            query=args.query,
            repo=repo,
            embedder=embedder,
            bm25_index=bm25_index,
            settings=_settings,
            top_k=top_k,
            show_scores=show_scores,
            show_prompt=args.prompt,
        )
    else:
        # Initialise once before loop
        repo, embedder, bm25_index = setup_pipeline(_settings)
        # Interactive REPL
        interactive_loop(
            repo=repo,
            embedder=embedder,
            bm25_index=bm25_index,
            settings=_settings,
            top_k=top_k,
            show_scores=show_scores,
            show_prompt=args.prompt,
        )
