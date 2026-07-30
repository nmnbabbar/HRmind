"""
tests/test_rag_agent.py
========================
Tests for the RAG agent pipeline (Phase 2).

Test scope
----------
1. Ingestion pipeline — hash-based deduplication, chunking, metadata
2. BM25Index — tokenization, scoring, top-n ordering
3. Reciprocal Rank Fusion — deduplication, score ordering, citation format
4. RAGContextBuilder — message structure, context injection
5. RAGAgent.run() — mocked retrieval + LLM, success path + guardrail block
6. ChromaVectorRepository — unit-testable without a live ChromaDB instance

All ChromaDB and LLM calls are mocked. No network access required.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from backend.agents.rag_agent.context_builder import RAGContextBuilder
from backend.agents.rag_agent.hybrid_search import (
    format_citation,
    format_context_with_citations,
    reciprocal_rank_fusion,
)
from backend.agents.rag_agent.ingestion import (
    chunk_documents,
    compute_file_hash,
    make_chunk_id,
)
from backend.agents.rag_agent.retriever import BM25Index
from backend.state import GraphState


# ── Fixtures ───────────────────────────────────────────────────────────────────

def make_doc(content: str, source: str = "test.pdf", page: int = 1, idx: int = 0) -> Document:
    return Document(
        page_content=content,
        metadata={
            "source": source,
            "page": page,
            "chunk_index": idx,
            "file_hash": "abc123",
            "doc_type": "pdf",
        },
    )


def make_state(query: str = "test query") -> GraphState:
    return GraphState(
        query=query,
        session_id="test-session",
        conversation_summary="",
        recent_turns=[],
        entity_store={},
        plan={"queries": {"rag": query}, "agents": ["rag"], "parallel": False, "reasoning": ""},
        agent_results=[],
        final_answer="",
        uploaded_file_path=None,
        error=None,
    )


# ── Hybrid search tests ────────────────────────────────────────────────────────

class TestReciprocalRankFusion:
    def _make_docs(self, n: int, source_prefix: str = "doc") -> list[Document]:
        return [
            Document(
                page_content=f"Content {i}",
                metadata={
                    "source": f"{source_prefix}_{i}.pdf",
                    "page": i + 1,
                    "chunk_index": 0,
                    "chroma_id": f"{source_prefix}_{i}",
                },
            )
            for i in range(n)
        ]

    def test_rrf_returns_correct_count(self):
        dense = self._make_docs(10, "dense")
        sparse = self._make_docs(10, "sparse")
        result = reciprocal_rank_fusion(dense, sparse, k=60, final_top_k=5)
        assert len(result) == 5

    def test_rrf_deduplicates_same_doc(self):
        """A doc in both dense and sparse should appear once with combined score."""
        shared = Document(
            page_content="Shared content",
            metadata={
                "source": "shared.pdf",
                "page": 1,
                "chunk_index": 0,
                "chroma_id": "shared_0",
            },
        )
        dense = [shared] + self._make_docs(5, "dense")
        sparse = [shared] + self._make_docs(5, "sparse")

        result = reciprocal_rank_fusion(dense, sparse, k=60, final_top_k=10)
        # Shared doc should appear exactly once
        shared_results = [d for d in result if d.metadata.get("chroma_id") == "shared_0"]
        assert len(shared_results) == 1

    def test_rrf_scores_combined_doc_higher(self):
        """Doc appearing in both lists should rank above doc appearing in one list only."""
        shared = Document(
            page_content="Shared at top of both",
            metadata={"source": "both.pdf", "page": 1, "chunk_index": 0, "chroma_id": "shared"},
        )
        unique_dense = Document(
            page_content="Only in dense",
            metadata={"source": "dense_only.pdf", "page": 1, "chunk_index": 0, "chroma_id": "dense_only"},
        )

        result = reciprocal_rank_fusion(
            dense_results=[shared, unique_dense],
            sparse_results=[shared],
            k=60,
            final_top_k=5,
        )
        # shared appears in both lists, should have higher score
        result_ids = [d.metadata["chroma_id"] for d in result]
        assert result_ids[0] == "shared", "Shared doc should rank first"

    def test_rrf_attaches_rank_metadata(self):
        dense = self._make_docs(3, "d")
        sparse = self._make_docs(3, "s")
        result = reciprocal_rank_fusion(dense, sparse, k=60, final_top_k=6)
        for doc in result:
            assert "rrf_score" in doc.metadata
            assert "dense_rank" in doc.metadata
            assert "sparse_rank" in doc.metadata

    def test_rrf_empty_inputs(self):
        result = reciprocal_rank_fusion([], [], k=60, final_top_k=8)
        assert result == []

    def test_rrf_only_dense(self):
        dense = self._make_docs(5, "d")
        result = reciprocal_rank_fusion(dense, [], k=60, final_top_k=5)
        assert len(result) == 5
        for doc in result:
            assert doc.metadata["sparse_rank"] is None


class TestCitationFormatting:
    def test_format_citation_standard(self):
        doc = make_doc("content", "Maternity-Policy.docx", page=3)
        assert format_citation(doc) == "[Maternity-Policy.docx, page 3]"

    def test_format_citation_missing_page(self):
        doc = Document(page_content="content", metadata={"source": "Policy.pdf"})
        citation = format_citation(doc)
        assert "Policy.pdf" in citation
        assert "page" in citation

    def test_format_context_with_citations(self):
        docs = [
            make_doc("Policy text here.", "Alcohol-Policy.pdf", page=1),
            make_doc("Another policy section.", "Drugs-Policy.pdf", page=2),
        ]
        context = format_context_with_citations(docs)
        assert "Alcohol-Policy.pdf" in context
        assert "Drugs-Policy.pdf" in context
        assert "Policy text here." in context


# ── BM25 index tests ───────────────────────────────────────────────────────────

class TestBM25Index:
    def test_build_from_documents(self):
        docs = [
            make_doc("maternity leave policy 26 weeks", "Maternity-Policy.docx"),
            make_doc("drugs alcohol prohibited at work", "Alcohol-Policy.pdf"),
            make_doc("notice period one week per year", "Notice-Periods-Policy.pdf"),
        ]
        index = BM25Index(docs)
        assert index.document_count == 3

    def test_top_n_ordering(self):
        docs = [
            make_doc("maternity leave policy 26 weeks statutory", "Maternity-Policy.docx"),
            make_doc("drugs alcohol prohibited at work conduct", "Alcohol-Policy.pdf"),
            make_doc("notice period one week per year service", "Notice-Periods-Policy.pdf"),
        ]
        index = BM25Index(docs)
        results = index.get_top_n("maternity leave", n=3)
        # Maternity doc should rank first
        assert results[0].metadata["source"] == "Maternity-Policy.docx"

    def test_top_n_respects_limit(self):
        docs = [make_doc(f"content {i}", f"doc_{i}.pdf") for i in range(10)]
        index = BM25Index(docs)
        results = index.get_top_n("content", n=3)
        assert len(results) == 3

    def test_bm25_score_attached(self):
        docs = [make_doc("leave policy maternity", "test.pdf")]
        index = BM25Index(docs)
        results = index.get_top_n("maternity leave", n=1)
        assert "bm25_score" in results[0].metadata

    def test_empty_index(self):
        index = BM25Index([])
        results = index.get_top_n("any query", n=5)
        assert results == []


# ── Context builder tests ──────────────────────────────────────────────────────

class TestRAGContextBuilder:
    def test_builds_system_and_user_messages(self):
        builder = RAGContextBuilder()
        state = make_state("What is maternity leave?")
        context = "--- [Maternity-Policy.docx, page 1] ---\nEmployees get 26 weeks."
        messages = builder.build(state, "What is maternity leave?", context)

        from langchain_core.messages import HumanMessage, SystemMessage
        assert any(isinstance(m, SystemMessage) for m in messages)
        assert any(isinstance(m, HumanMessage) for m in messages)

    def test_context_in_system_message(self):
        builder = RAGContextBuilder()
        state = make_state()
        context = "--- [Policy.pdf, page 1] ---\nSpecific policy text."
        messages = builder.build(state, "query", context)
        system_content = messages[0].content
        assert "Policy.pdf" in system_content
        assert "Specific policy text." in system_content

    def test_summary_injected_when_present(self):
        builder = RAGContextBuilder()
        state = make_state()
        state["conversation_summary"] = "Previously asked about leave."
        messages = builder.build(state, "follow-up query", "some context")
        # Should contain a message with the summary
        all_content = " ".join(m.content for m in messages)
        assert "Previously asked about leave." in all_content

    def test_no_sql_or_ocr_in_rag_messages(self):
        """RAG messages must NOT contain SQL schema or OCR text."""
        builder = RAGContextBuilder()
        state = make_state()
        state["agent_results"] = [
            {"agent_name": "sql", "answer": "SELECT * FROM employees", "success": True}
        ]
        messages = builder.build(state, "policy question", "HR policy context")
        all_content = " ".join(m.content for m in messages)
        # SQL agent result should NOT bleed into RAG prompt
        assert "SELECT" not in all_content


# ── Ingestion unit tests ───────────────────────────────────────────────────────

class TestIngestionHelpers:
    def test_make_chunk_id_deterministic(self):
        id1 = make_chunk_id("hashxyz", 0)
        id2 = make_chunk_id("hashxyz", 0)
        assert id1 == id2

    def test_make_chunk_id_different_chunks(self):
        id1 = make_chunk_id("hashxyz", 0)
        id2 = make_chunk_id("hashxyz", 1)
        assert id1 != id2

    def test_make_chunk_id_different_files(self):
        id1 = make_chunk_id("hash_a", 0)
        id2 = make_chunk_id("hash_b", 0)
        assert id1 != id2

    def test_chunk_documents_metadata(self, tmp_path: Path):
        """Chunks must carry required citation metadata fields."""
        file_path = tmp_path / "test_policy.pdf"
        file_path.write_text("test content")
        raw_docs = [
            Document(
                page_content="This is a test policy document with enough content to be chunked properly. " * 10,
                metadata={"page": 1, "source": "test_policy.pdf"},
            )
        ]
        chunks = chunk_documents(raw_docs, file_path, "testhash123")
        assert len(chunks) > 0
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert "page" in chunk.metadata
            assert "chunk_index" in chunk.metadata
            assert "file_hash" in chunk.metadata
            assert "total_chunks" in chunk.metadata
            assert "doc_type" in chunk.metadata

    def test_chunk_documents_total_chunks_consistent(self, tmp_path: Path):
        """total_chunks must equal len(chunks) for all chunks."""
        file_path = tmp_path / "long_policy.pdf"
        long_text = "The employee is entitled to leave. " * 200  # enough for multiple chunks
        raw_docs = [Document(page_content=long_text, metadata={"page": 1})]
        chunks = chunk_documents(raw_docs, file_path, "longhash")
        total = len(chunks)
        for chunk in chunks:
            assert chunk.metadata["total_chunks"] == total


# ── RAGAgent integration tests (mocked) ───────────────────────────────────────

class TestRAGAgent:
    @pytest.fixture
    def mock_llm(self):
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(
            return_value=AIMessage(
                content="Employees are entitled to 26 weeks of maternity leave "
                        "[Maternity-Policy.docx, page 2].\n\n"
                        "Sources:\n[Maternity-Policy.docx, page 2]"
            )
        )
        llm.model_name = "gpt-4o-mini-mock"
        return llm

    @pytest.fixture
    def mock_embedding(self):
        svc = AsyncMock()
        svc.embed_query = AsyncMock(return_value=[0.1] * 1024)  # bge-large dim
        svc.embed_passages = AsyncMock(return_value=[[0.1] * 1024])
        return svc

    @pytest.fixture
    def mock_vector_repo(self):
        repo = AsyncMock()
        repo.similarity_search = AsyncMock(return_value=[
            Document(
                page_content="Employees get 26 weeks statutory maternity leave.",
                metadata={
                    "source": "Maternity-Policy.docx",
                    "page": 2,
                    "chunk_index": 0,
                    "chroma_id": "mat_0",
                    "file_hash": "abc",
                },
            )
        ])
        return repo

    @pytest.fixture
    def mock_bm25(self):
        index = MagicMock()
        index.get_top_n = MagicMock(return_value=[
            Document(
                page_content="Maternity leave entitlement 26 weeks ordinary.",
                metadata={
                    "source": "Maternity-Policy.docx",
                    "page": 1,
                    "chunk_index": 1,
                    "chroma_id": "mat_1",
                    "bm25_score": 4.5,
                },
            )
        ])
        return index

    @pytest.mark.asyncio
    async def test_successful_retrieval_and_answer(
        self, mock_llm, mock_embedding, mock_vector_repo, mock_bm25
    ):
        from backend.agents.rag_agent.rag_agent import RAGAgent

        # Patch TopicGuardrail to always pass
        with patch(
            "backend.agents.rag_agent.guardrails.TopicGuardrail.check",
            new_callable=AsyncMock,
        ) as mock_topic, patch(
            "backend.agents.rag_agent.guardrails.GroundingGuardrail.check_grounding",
            new_callable=AsyncMock,
        ) as mock_grounding:
            from backend.state import GuardrailResult
            mock_topic.return_value = GuardrailResult.ok()
            mock_grounding.return_value = GuardrailResult.ok()

            agent = RAGAgent(
                llm=mock_llm,
                embedding_service=mock_embedding,
                vector_repo=mock_vector_repo,
                bm25_index=mock_bm25,
            )

            state = make_state("What is the maternity leave entitlement?")
            result = await agent.run(state)

        assert result.success is True
        assert result.agent_name == "rag"
        assert "maternity" in result.answer.lower()
        assert len(result.sources) > 0
        assert "Maternity-Policy.docx" in result.sources[0]

    @pytest.mark.asyncio
    async def test_guardrail_blocks_non_hr_query(
        self, mock_llm, mock_embedding, mock_vector_repo, mock_bm25
    ):
        from backend.agents.rag_agent.rag_agent import RAGAgent
        from backend.state import GuardrailResult

        with patch(
            "backend.agents.rag_agent.guardrails.TopicGuardrail.check",
            new_callable=AsyncMock,
            return_value=GuardrailResult.fail(
                reason="Not HR-related", guardrail_name="TopicGuardrail"
            ),
        ):
            agent = RAGAgent(
                llm=mock_llm,
                embedding_service=mock_embedding,
                vector_repo=mock_vector_repo,
                bm25_index=mock_bm25,
            )
            state = make_state("What is the recipe for carbonara pasta?")
            result = await agent.run(state)

        assert result.success is False
        assert "TopicGuardrail" in result.error

    @pytest.mark.asyncio
    async def test_empty_query_returns_failure(
        self, mock_llm, mock_embedding, mock_vector_repo, mock_bm25
    ):
        from backend.agents.rag_agent.rag_agent import RAGAgent

        agent = RAGAgent(
            llm=mock_llm,
            embedding_service=mock_embedding,
            vector_repo=mock_vector_repo,
            bm25_index=mock_bm25,
        )
        state = make_state("")
        result = await agent.run(state)
        assert result.success is False

    @pytest.mark.asyncio
    async def test_structured_data_has_chunks(
        self, mock_llm, mock_embedding, mock_vector_repo, mock_bm25
    ):
        from backend.agents.rag_agent.rag_agent import RAGAgent

        with patch(
            "backend.agents.rag_agent.guardrails.TopicGuardrail.check",
            new_callable=AsyncMock,
        ) as mock_topic, patch(
            "backend.agents.rag_agent.guardrails.GroundingGuardrail.check_grounding",
            new_callable=AsyncMock,
        ) as mock_grounding:
            from backend.state import GuardrailResult
            mock_topic.return_value = GuardrailResult.ok()
            mock_grounding.return_value = GuardrailResult.ok()

            agent = RAGAgent(
                llm=mock_llm,
                embedding_service=mock_embedding,
                vector_repo=mock_vector_repo,
                bm25_index=mock_bm25,
            )
            state = make_state("What is the maternity leave entitlement?")
            result = await agent.run(state)

        assert result.structured_data is not None
        assert "chunks" in result.structured_data
        assert "sources" in result.structured_data
        assert "query_used" in result.structured_data
        assert len(result.structured_data["chunks"]) > 0
