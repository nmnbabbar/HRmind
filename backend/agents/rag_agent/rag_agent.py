"""
backend/agents/rag_agent/rag_agent.py
=======================================
RAGAgent — the concrete HR policy retrieval agent.

Execution flow (execute-in-code pattern)
-----------------------------------------
1. Extract query from state (uses plan's per-agent rewrite if available)
2. Run TopicGuardrail — rejects non-HR queries early
3. Execute hybrid retrieval IN CODE (no LLM tool calls):
   a. Dense: embed query → ChromaDB similarity_search → top-20 chunks
   b. Sparse: BM25 index lookup → top-20 chunks
   c. RRF fusion → top-8 chunks
4. Build prompt with retrieved context + citations
5. Single LLM call for synthesis
6. Post-generation GroundingGuardrail (NLI check)
7. Return AgentResult with answer + sources list (for Combiner)

Dependencies (injected by AgentFactory)
----------------------------------------
    llm               : BaseChatModel — synthesis LLM
    embedding_service : EmbeddingService — bge-large-en-v1.5
    vector_repo       : ChromaVectorRepository — ChromaDB HTTP client
    bm25_index        : BM25Index — in-memory sparse index
    settings          : Settings — retrieval k values, token budgets
"""

from __future__ import annotations

import asyncio
import logging
import time

from langchain_core.language_models import BaseChatModel

from backend.agents.rag_agent.context_builder import (
    NO_CONTEXT_ANSWER,
    RAGContextBuilder,
)
from backend.agents.rag_agent.guardrails import GroundingGuardrail, TopicGuardrail
from backend.agents.rag_agent.hybrid_search import (
    format_citation,
    format_context_with_citations,
    reciprocal_rank_fusion,
)
from backend.agents.rag_agent.ingestion import EmbeddingService
from backend.agents.rag_agent.retriever import BM25Index, ChromaVectorRepository
from backend.base.agent import BaseAgent
from backend.config import Settings, get_settings
from backend.state import AgentResult, GraphState

logger = logging.getLogger(__name__)


class RAGAgent(BaseAgent):
    """
    Hybrid-search RAG agent for HR policy questions.

    Inherits from BaseAgent (SOLID: Liskov + Open/Closed).
    All retrieval logic is executed in Python — the LLM only sees
    the pre-retrieved context and synthesizes a grounded answer.

    Constructor Parameters
    ----------------------
    llm : BaseChatModel
        Language model for synthesis (e.g. gpt-4o-mini).
    embedding_service : EmbeddingService
        Pre-loaded bge-large-en-v1.5 embedding model.
    vector_repo : ChromaVectorRepository
        ChromaDB HTTP async client for dense search.
    bm25_index : BM25Index
        In-memory BM25 index for sparse search.
    settings : Settings
        Application settings (k values, token budgets).
    """

    def __init__(
        self,
        llm: BaseChatModel,
        embedding_service: EmbeddingService,
        vector_repo: ChromaVectorRepository,
        bm25_index: BM25Index,
        settings: Settings | None = None,
    ) -> None:
        _settings = settings or get_settings()
        # Inject TopicGuardrail as the pre-query guardrail
        super().__init__(llm=llm, guardrails=[TopicGuardrail(llm)])

        self._embedding = embedding_service
        self._vector_repo = vector_repo
        self._bm25 = bm25_index
        self._settings = _settings
        self._context_builder = RAGContextBuilder(
            max_summary_tokens=_settings.max_summary_tokens,
            max_recent_tokens=_settings.max_recent_turns_tokens,
        )
        self._grounding_guardrail = GroundingGuardrail(llm)

    @property
    def name(self) -> str:
        return "rag"

    async def run(self, state: GraphState) -> AgentResult:
        """
        Execute the full RAG pipeline for a given LangGraph state.

        Returns
        -------
        AgentResult
            Always returned (never raises). On failure, success=False.
        """
        t0 = time.monotonic()

        # ── 1. Extract query ───────────────────────────────────────────────
        # Use the Planner's per-agent rewritten query if available
        plan = state.get("plan")
        if plan and isinstance(plan, dict):
            query = plan.get("queries", {}).get("rag") or state.get("query", "")
        else:
            query = state.get("query", "")

        if not query:
            return AgentResult.failure(
                agent_name=self.name,
                error="No query provided to RAG agent.",
                metadata=self._timed_result(t0),
            )

        logger.info("RAGAgent: processing query: %r", query[:100])

        # ── 2. Pre-query guardrail (TopicGuardrail) ───────────────────────
        guard_block = await self._run_with_guardrails(query)
        if guard_block:
            return guard_block  # non-HR query blocked

        # ── 3. Hybrid retrieval (no LLM calls) ────────────────────────────
        try:
            dense_k = self._settings.dense_top_k
            sparse_k = self._settings.sparse_top_k
            final_k = self._settings.final_top_k
            rrf_k = self._settings.rrf_k

            # Embed query (blocking → thread pool)
            query_embedding = await self._embedding.embed_query(query)

            # Run dense + sparse search concurrently
            dense_results, sparse_results = await asyncio.gather(
                self._vector_repo.similarity_search(query_embedding, k=dense_k),
                asyncio.get_event_loop().run_in_executor(
                    None, self._bm25.get_top_n, query, sparse_k
                ),
            )

            logger.debug(
                "Retrieval: dense=%d, sparse=%d",
                len(dense_results),
                len(sparse_results),
            )

            # RRF fusion → top final_k chunks
            fused_chunks = reciprocal_rank_fusion(
                dense_results=dense_results,
                sparse_results=sparse_results,
                k=rrf_k,
                final_top_k=final_k,
            )

            if not fused_chunks:
                logger.warning("RAGAgent: no chunks retrieved for query: %r", query[:100])
                return AgentResult(
                    agent_name=self.name,
                    success=True,
                    answer=NO_CONTEXT_ANSWER,
                    sources=[],
                    metadata=self._timed_result(t0, {"retrieved_chunks": 0}),
                )

        except Exception as exc:
            logger.error("RAGAgent: retrieval error: %s", exc, exc_info=True)
            return AgentResult.failure(
                agent_name=self.name,
                error=f"Retrieval failed: {exc}",
                metadata=self._timed_result(t0),
            )

        # ── 4. Build prompt with context + citations ───────────────────────
        context_str = format_context_with_citations(fused_chunks)
        messages = self._context_builder.build(state, query, context_str)

        # ── 5. Single LLM call for synthesis ──────────────────────────────
        try:
            response = await self._llm.ainvoke(messages)
            answer = response.content

        except Exception as exc:
            logger.error("RAGAgent: LLM synthesis error: %s", exc, exc_info=True)
            return AgentResult.failure(
                agent_name=self.name,
                error=f"LLM synthesis failed: {exc}",
                metadata=self._timed_result(t0),
            )

        # ── 6. Post-generation grounding check ────────────────────────────
        grounding = await self._grounding_guardrail.check_grounding(
            query=query,
            context=context_str,
            answer=answer,
        )
        if not grounding.passed:
            logger.warning(
                "RAGAgent: grounding check failed — using fallback answer."
            )
            answer = (
                "I found some information but could not fully verify its accuracy "
                "against the source documents. Please consult your HR department directly.\n\n"
                f"Relevant documents: {', '.join(format_citation(c) for c in fused_chunks)}"
            )

        # ── 7. Build sources list ──────────────────────────────────────────
        sources = [format_citation(chunk) for chunk in fused_chunks]
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_sources: list[str] = []
        for s in sources:
            if s not in seen:
                seen.add(s)
                unique_sources.append(s)

        # Rich structured data for Combiner + frontend
        structured_data = {
            "chunks": [
                {
                    "content": c.page_content,
                    "source": c.metadata.get("source", ""),
                    "page": c.metadata.get("page", "?"),
                    "chunk_index": c.metadata.get("chunk_index", 0),
                    "rrf_score": c.metadata.get("rrf_score", 0.0),
                    "dense_rank": c.metadata.get("dense_rank"),
                    "sparse_rank": c.metadata.get("sparse_rank"),
                }
                for c in fused_chunks
            ],
            "sources": unique_sources,
            "query_used": query,
        }

        metadata = self._timed_result(
            t0,
            {
                "model": getattr(self._llm, "model_name", "unknown"),
                "retrieved_chunks": len(fused_chunks),
                "dense_results": len(dense_results),
                "sparse_results": len(sparse_results),
                "grounding_passed": grounding.passed,
            },
        )

        logger.info(
            "RAGAgent: completed in %dms, %d chunks, grounding=%s",
            metadata["duration_ms"],
            len(fused_chunks),
            grounding.passed,
        )

        return AgentResult(
            agent_name=self.name,
            success=True,
            answer=answer,
            sources=unique_sources,
            structured_data=structured_data,
            metadata=metadata,
        )
