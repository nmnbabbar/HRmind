"""
backend/agents/rag_agent/evals/eval_runner.py
==============================================
RAGAS evaluation runner for the RAG agent pipeline.

⚠️  THIS SCRIPT MAKES LLM CALLS AND CHROMADB CALLS.
    Do not run until:
    1. ChromaDB is running and ingestion has been completed
    2. LLM credentials are configured in .env
    3. You have explicitly decided to evaluate

How to run
----------
    # From the project root:
    uv run python -m backend.agents.rag_agent.evals.eval_runner

    # Or with a custom output path:
    uv run python -m backend.agents.rag_agent.evals.eval_runner --output ./my_results.json

    # Dry-run (retrieval only, no LLM synthesis or RAGAS scoring):
    uv run python -m backend.agents.rag_agent.evals.eval_runner --dry-run

What it does
-------------
For each sample in GOLDEN_DATASET:
    1. Embed the question → ChromaDB similarity search (top-20)
    2. BM25 index lookup (top-20)
    3. RRF fusion → top-8 chunks
    4. LLM synthesis → generated answer
    5. Collect: {question, answer, contexts, ground_truth}

Then:
    - Feeds all samples to RAGAS evaluate()
    - Computes: Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
    - Saves full report to eval_results/ragas_results_<timestamp>.json

Output JSON schema
------------------
{
  "run_metadata": {
    "timestamp": "2026-07-30T16:00:00Z",
    "model": "deepseek-ai/deepseek-r1-0528-qwen3-8b",
    "embedding_model": "BAAI/bge-large-en-v1.5",
    "total_samples": 33,
    "retrieval_config": { "dense_top_k": 20, "sparse_top_k": 20, "final_top_k": 8, "rrf_k": 60 }
  },
  "aggregate_scores": {
    "faithfulness": 0.87,
    "answer_relevancy": 0.91,
    "context_recall": 0.83,
    "context_precision": 0.79
  },
  "metric_definitions": { ... },
  "per_question_results": [
    {
      "index": 0,
      "question": "...",
      "ground_truth": "...",
      "source_document": "...",
      "generated_answer": "...",
      "retrieved_contexts": ["...", "..."],
      "citations": ["[Maternity-Policy.docx, page 2]", ...],
      "scores": {
        "faithfulness": 0.9,
        "answer_relevancy": 0.88,
        "context_recall": 0.75,
        "context_precision": 0.83
      },
      "retrieval_stats": {
        "dense_results_count": 20,
        "sparse_results_count": 20,
        "fused_chunks_count": 8,
        "duration_ms": 312
      }
    },
    ...
  ]
}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import chromadb

from backend.agents.rag_agent.evals.eval_dataset import GOLDEN_DATASET
from backend.agents.rag_agent.evals.metrics import METRIC_DEFINITIONS, build_ragas_metrics
from backend.agents.rag_agent.hybrid_search import (
    format_citation,
    format_context_with_citations,
    reciprocal_rank_fusion,
)
from backend.agents.rag_agent.ingestion import EmbeddingService
from backend.agents.rag_agent.retriever import (
    BM25Index,
    ChromaVectorRepository,
    build_bm25_index,
)
from backend.config import get_settings

logger = logging.getLogger(__name__)

# Output directory relative to project root
RESULTS_DIR = Path("eval_results")


# ── Result data structures ─────────────────────────────────────────────────────

def build_run_metadata(settings, total_samples: int) -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": settings.agent_model,
        "embedding_model": settings.embedding_model,
        "llm_base_url": settings.llm_base_url,
        "total_samples": total_samples,
        "retrieval_config": {
            "dense_top_k": settings.dense_top_k,
            "sparse_top_k": settings.sparse_top_k,
            "final_top_k": settings.final_top_k,
            "rrf_k": settings.rrf_k,
        },
    }


def build_metric_definitions_block() -> dict:
    """Serialise MetricDefinition objects for the JSON report."""
    return {
        key: {
            "name": md.name,
            "short_description": md.short_description,
            "what_it_measures": md.what_it_measures,
            "range": md.range,
            "interpretation": md.interpretation,
            "common_failure_causes": md.common_failure_causes,
        }
        for key, md in METRIC_DEFINITIONS.items()
    }


# ── Retrieval + synthesis per question ────────────────────────────────────────

async def retrieve_and_answer(
    question: str,
    embedding_svc: EmbeddingService,
    vector_repo: ChromaVectorRepository,
    bm25_index: BM25Index,
    llm,
    settings,
    dry_run: bool = False,
) -> dict:
    """
    Run full retrieval + LLM synthesis for a single evaluation question.

    Parameters
    ----------
    question : str
        The evaluation question.
    embedding_svc : EmbeddingService
        Pre-loaded embedding model.
    vector_repo : ChromaVectorRepository
        ChromaDB repository.
    bm25_index : BM25Index
        In-memory BM25 index.
    llm : BaseChatModel
        Synthesis LLM.
    settings : Settings
        Application settings.
    dry_run : bool
        If True, skip LLM synthesis — retrieval only.

    Returns
    -------
    dict
        {generated_answer, retrieved_contexts, citations, retrieval_stats}
    """
    t0 = time.monotonic()

    # 1. Embed query
    query_embedding = await embedding_svc.embed_query(question)

    # 2. Dense + sparse search concurrently
    dense_results, sparse_results = await asyncio.gather(
        vector_repo.similarity_search(query_embedding, k=settings.dense_top_k),
        asyncio.get_event_loop().run_in_executor(
            None, bm25_index.get_top_n, question, settings.sparse_top_k
        ),
    )

    # 3. RRF fusion
    fused = reciprocal_rank_fusion(
        dense_results=dense_results,
        sparse_results=sparse_results,
        k=settings.rrf_k,
        final_top_k=settings.final_top_k,
    )

    retrieved_contexts = [chunk.page_content for chunk in fused]
    citations = [format_citation(chunk) for chunk in fused]

    retrieval_ms = int((time.monotonic() - t0) * 1000)

    # 4. LLM synthesis (skipped in dry-run)
    generated_answer = "[dry-run — no LLM call made]"
    if not dry_run and fused:
        from langchain_core.messages import HumanMessage, SystemMessage

        context_str = format_context_with_citations(fused)
        messages = [
            SystemMessage(
                content=(
                    "You are an expert HR assistant. Answer the question using ONLY "
                    "the provided document excerpts. Include inline citations in the "
                    "format [Filename, page N] for every factual claim. "
                    "If the information is not in the excerpts, say so explicitly.\n\n"
                    f"DOCUMENT EXCERPTS:\n{context_str}"
                )
            ),
            HumanMessage(content=question),
        ]
        response = await llm.ainvoke(messages)
        generated_answer = response.content

    total_ms = int((time.monotonic() - t0) * 1000)

    return {
        "generated_answer": generated_answer,
        "retrieved_contexts": retrieved_contexts,
        "citations": citations,
        "retrieval_stats": {
            "dense_results_count": len(dense_results),
            "sparse_results_count": len(sparse_results),
            "fused_chunks_count": len(fused),
            "retrieval_ms": retrieval_ms,
            "total_ms": total_ms,
        },
    }


# ── RAGAS scoring ──────────────────────────────────────────────────────────────

def run_ragas_scoring(
    eval_samples: list[dict],
    llm=None,
    embeddings=None,
) -> dict:
    """
    Run RAGAS evaluation on collected samples.

    Parameters
    ----------
    eval_samples : list[dict]
        Each dict: {question, answer, contexts, ground_truth}
    llm : optional
        Judge LLM for RAGAS (can differ from agent LLM).
    embeddings : optional
        Embeddings for answer relevancy metric.

    Returns
    -------
    dict
        {"aggregate": {metric: score}, "per_sample": [{"metric": score, ...}, ...]}
    """
    from datasets import Dataset
    from ragas import evaluate

    metrics = build_ragas_metrics(llm=llm, embeddings=embeddings)

    dataset = Dataset.from_list(eval_samples)
    result = evaluate(dataset, metrics=metrics)

    # Aggregate scores
    aggregate = {
        "faithfulness": round(float(result["faithfulness"]), 4),
        "answer_relevancy": round(float(result["answer_relevancy"]), 4),
        "context_recall": round(float(result["context_recall"]), 4),
        "context_precision": round(float(result["context_precision"]), 4),
    }

    # Per-sample scores
    df = result.to_pandas()
    per_sample = []
    for _, row in df.iterrows():
        per_sample.append({
            "faithfulness": round(float(row.get("faithfulness", 0)), 4),
            "answer_relevancy": round(float(row.get("answer_relevancy", 0)), 4),
            "context_recall": round(float(row.get("context_recall", 0)), 4),
            "context_precision": round(float(row.get("context_precision", 0)), 4),
        })

    return {"aggregate": aggregate, "per_sample": per_sample}


# ── Main eval loop ─────────────────────────────────────────────────────────────

async def run_evaluation(
    output_path: Path | None = None,
    dry_run: bool = False,
) -> dict:
    """
    Full evaluation pipeline.

    1. Set up infrastructure (ChromaDB, BM25, embeddings, LLM)
    2. For each sample: retrieve + synthesize
    3. Run RAGAS scoring (unless dry_run)
    4. Save results to JSON
    5. Return the full report dict

    Parameters
    ----------
    output_path : Path | None
        Where to save the JSON report. Defaults to
        eval_results/ragas_results_<timestamp>.json
    dry_run : bool
        If True, run retrieval only — no LLM calls, no RAGAS scoring.
        Useful for testing that the retrieval pipeline is working.
    """
    settings = get_settings()

    logger.info("=== HrMind RAG Evaluation Starting ===")
    logger.info("Model: %s", settings.agent_model)
    logger.info("Embedding: %s", settings.embedding_model)
    logger.info("Samples: %d", len(GOLDEN_DATASET))
    logger.info("Dry run: %s", dry_run)

    # ── Infrastructure setup ───────────────────────────────────────────────
    logger.info("Connecting to ChromaDB at %s:%s...", settings.chroma_host, settings.chroma_port)
    chroma_client = await chromadb.AsyncHttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port,
    )

    vector_repo = ChromaVectorRepository(
        client=chroma_client,
        collection_name=settings.chroma_collection_name,
    )

    # Verify collection is populated
    count = await vector_repo.collection_count()
    if count == 0:
        raise RuntimeError(
            "ChromaDB collection is empty. Run ingestion first:\n"
            "  uv run python -m backend.agents.rag_agent.ingestion"
        )
    logger.info("ChromaDB collection has %d chunks.", count)

    # Load embedding model
    logger.info("Loading embedding model: %s (this may take 30s on first run)...", settings.embedding_model)
    embedding_svc = EmbeddingService(settings.embedding_model)

    # Build BM25 index
    logger.info("Building BM25 index...")
    bm25_index = await build_bm25_index(vector_repo)

    # Set up LLM (only needed if not dry-run)
    llm = None
    if not dry_run:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=settings.agent_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            temperature=0,
        )
        logger.info("LLM ready: %s @ %s", settings.agent_model, settings.llm_base_url)

    # ── Per-question retrieval + synthesis ────────────────────────────────
    logger.info("Running retrieval + synthesis for %d questions...", len(GOLDEN_DATASET))

    per_question_results = []
    ragas_input_samples = []

    for idx, sample in enumerate(GOLDEN_DATASET):
        question = sample["question"]
        ground_truth = sample["ground_truth"]
        source_doc = sample["source_document"]

        logger.info("[%d/%d] %s", idx + 1, len(GOLDEN_DATASET), question[:80])

        try:
            result = await retrieve_and_answer(
                question=question,
                embedding_svc=embedding_svc,
                vector_repo=vector_repo,
                bm25_index=bm25_index,
                llm=llm,
                settings=settings,
                dry_run=dry_run,
            )

            per_question_results.append({
                "index": idx,
                "question": question,
                "ground_truth": ground_truth,
                "source_document": source_doc,
                "generated_answer": result["generated_answer"],
                "retrieved_contexts": result["retrieved_contexts"],
                "citations": result["citations"],
                "retrieval_stats": result["retrieval_stats"],
                "scores": {},  # filled in after RAGAS scoring
            })

            # Collect RAGAS input format
            ragas_input_samples.append({
                "question": question,
                "answer": result["generated_answer"],
                "contexts": result["retrieved_contexts"],
                "ground_truth": ground_truth,
            })

        except Exception as exc:
            logger.error("[%d/%d] FAILED: %s — %s", idx + 1, len(GOLDEN_DATASET), question[:60], exc)
            per_question_results.append({
                "index": idx,
                "question": question,
                "ground_truth": ground_truth,
                "source_document": source_doc,
                "error": str(exc),
                "scores": {},
            })
            ragas_input_samples.append({
                "question": question,
                "answer": f"ERROR: {exc}",
                "contexts": [],
                "ground_truth": ground_truth,
            })

    # ── RAGAS scoring ──────────────────────────────────────────────────────
    aggregate_scores = {}
    if not dry_run:
        logger.info("Running RAGAS evaluation (this makes LLM calls for each sample)...")
        try:
            scoring = run_ragas_scoring(ragas_input_samples)
            aggregate_scores = scoring["aggregate"]

            # Attach per-sample scores to results
            for i, scores in enumerate(scoring["per_sample"]):
                if i < len(per_question_results):
                    per_question_results[i]["scores"] = scores

            logger.info("RAGAS Scores:")
            for metric, score in aggregate_scores.items():
                logger.info("  %-25s %.4f", metric, score)

        except Exception as exc:
            logger.error("RAGAS scoring failed: %s", exc, exc_info=True)
            aggregate_scores = {"error": str(exc)}
    else:
        logger.info("Dry run — skipping RAGAS scoring.")
        aggregate_scores = {"note": "dry_run — no LLM calls made, no RAGAS scores computed"}

    # ── Assemble report ────────────────────────────────────────────────────
    report = {
        "run_metadata": build_run_metadata(settings, len(GOLDEN_DATASET)),
        "aggregate_scores": aggregate_scores,
        "metric_definitions": build_metric_definitions_block(),
        "per_question_results": per_question_results,
    }

    # ── Save JSON ─────────────────────────────────────────────────────────
    if output_path is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        suffix = "_dryrun" if dry_run else ""
        output_path = RESULTS_DIR / f"ragas_results_{ts}{suffix}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("Results saved to: %s", output_path.resolve())

    # ── Print summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("HrMind RAG Evaluation Summary")
    print("=" * 60)
    print(f"Samples evaluated : {len(GOLDEN_DATASET)}")
    print(f"Results saved to  : {output_path.resolve()}")
    print()
    if not dry_run and isinstance(aggregate_scores, dict) and "error" not in aggregate_scores:
        print("RAGAS Scores:")
        print(f"  Faithfulness     : {aggregate_scores.get('faithfulness', 'N/A'):.4f}")
        print(f"  Answer Relevancy : {aggregate_scores.get('answer_relevancy', 'N/A'):.4f}")
        print(f"  Context Recall   : {aggregate_scores.get('context_recall', 'N/A'):.4f}")
        print(f"  Context Precision: {aggregate_scores.get('context_precision', 'N/A'):.4f}")
    else:
        print(f"Scores: {aggregate_scores}")
    print("=" * 60)

    return report


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation for the HrMind RAG agent."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save the JSON results. Default: eval_results/ragas_results_<timestamp>.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Run retrieval only — no LLM calls, no RAGAS scoring. Tests the pipeline without API usage.",
    )
    args = parser.parse_args()

    asyncio.run(run_evaluation(output_path=args.output, dry_run=args.dry_run))
