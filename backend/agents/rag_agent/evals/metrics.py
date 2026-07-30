"""
backend/agents/rag_agent/evals/metrics.py
==========================================
RAGAS metric definitions and configuration.

The four metrics used to evaluate the RAG pipeline:

1. Faithfulness    — answer contains only facts from the retrieved context
2. AnswerRelevancy — answer actually addresses the question
3. ContextRecall   — retrieved context covers all ground truth claims
4. ContextPrecision — retrieved context is mostly relevant (low noise)

All four metrics use LLM calls internally (via RAGAS framework).
The LLM used for evaluation is configured separately from the agent LLM
so you can use a stronger judge model (e.g. gpt-4o) even if agents use a
cheaper model.

Usage
-----
    from backend.agents.rag_agent.evals.metrics import build_ragas_metrics
    metrics = build_ragas_metrics(llm, embeddings)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricDefinition:
    """
    Human-readable definition of a RAGAS metric.

    Used to annotate results in the saved JSON report so the
    output is self-documenting without needing the RAGAS docs.
    """
    name: str
    short_description: str
    what_it_measures: str
    range: str
    interpretation: str
    common_failure_causes: list[str]


# ── Metric catalogue ────────────────────────────────────────────────────────────

METRIC_DEFINITIONS: dict[str, MetricDefinition] = {
    "faithfulness": MetricDefinition(
        name="Faithfulness",
        short_description="Fraction of answer claims supported by retrieved context",
        what_it_measures=(
            "Hallucination detection. Each factual claim in the generated answer "
            "is checked against the retrieved chunks. Claims not traceable to context "
            "reduce this score."
        ),
        range="0.0 – 1.0 (higher is better)",
        interpretation=(
            "1.0 = every fact in the answer is grounded in the retrieved chunks. "
            "0.5 = half the claims are hallucinated or unverifiable from context."
        ),
        common_failure_causes=[
            "LLM adds general knowledge not in the documents",
            "Chunks retrieved don't contain enough detail, LLM fills in gaps",
            "System prompt not strict enough about using only context",
        ],
    ),
    "answer_relevancy": MetricDefinition(
        name="Answer Relevancy",
        short_description="How well the answer addresses the original question",
        what_it_measures=(
            "Answer completeness and focus. The LLM generates reverse-questions "
            "from the answer and measures their embedding similarity to the original "
            "question. High score = answer is on-topic and complete."
        ),
        range="0.0 – 1.0 (higher is better)",
        interpretation=(
            "1.0 = answer directly and completely addresses the question. "
            "Low scores indicate vague, off-topic, or incomplete answers."
        ),
        common_failure_causes=[
            "RAG answered about a related but different topic",
            "Answer too generic ('Please consult HR') without actual policy content",
            "Question rewriting in Planner changed the intent",
        ],
    ),
    "context_recall": MetricDefinition(
        name="Context Recall",
        short_description="Fraction of ground truth claims found in retrieved context",
        what_it_measures=(
            "Retrieval completeness. Each sentence in the ground truth is checked "
            "against the retrieved chunks. If the retrieval missed key chunks, "
            "the ground truth facts won't be attributable to context."
        ),
        range="0.0 – 1.0 (higher is better)",
        interpretation=(
            "1.0 = all ground truth information was present in retrieved chunks. "
            "0.5 = only half the relevant policy content was retrieved."
        ),
        common_failure_causes=[
            "chunk_size too small — key information split across unchosen chunks",
            "final_top_k too low — relevant chunks ranked below the cutoff",
            "BM25 or dense search failed to rank the relevant doc highly",
            "Ground truth references a section not well-indexed",
        ],
    ),
    "context_precision": MetricDefinition(
        name="Context Precision",
        short_description="Fraction of retrieved context that is actually relevant",
        what_it_measures=(
            "Retrieval noise. Each retrieved chunk is classified as relevant or not. "
            "Relevant chunks ranked higher = better score. "
            "High precision = tight, focused retrieval with little noise."
        ),
        range="0.0 – 1.0 (higher is better)",
        interpretation=(
            "1.0 = all retrieved chunks were relevant to answering the question. "
            "Low scores indicate the retrieval is pulling in off-topic chunks "
            "that dilute the LLM's context window."
        ),
        common_failure_causes=[
            "RRF k too low — low-ranked but irrelevant chunks get boosted",
            "final_top_k too high — pulling in marginal chunks",
            "Keyword overlap causing BM25 to retrieve wrong-topic documents",
        ],
    ),
}


# ── RAGAS metric objects ─────────────────────────────────────────────────────

def build_ragas_metrics(llm=None, embeddings=None) -> list:
    """
    Build the list of RAGAS metric objects for evaluation.

    Parameters
    ----------
    llm : LangchainLLMWrapper | None
        LLM for RAGAS metric computation. If None, RAGAS uses its default.
        For best results use a strong judge model (e.g. gpt-4o or deepseek-v3).
    embeddings : LangchainEmbeddingsWrapper | None
        Embeddings for answer relevancy metric. If None, RAGAS uses its default.

    Returns
    -------
    list
        List of initialized RAGAS metric objects ready for evaluate().
    """
    # Import here so RAGAS (heavy dep) is only loaded when actually evaluating
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]

    # Inject custom LLM/embeddings if provided
    if llm is not None or embeddings is not None:
        for metric in metrics:
            if llm is not None and hasattr(metric, "llm"):
                metric.llm = llm
            if embeddings is not None and hasattr(metric, "embeddings"):
                metric.embeddings = embeddings

    return metrics
