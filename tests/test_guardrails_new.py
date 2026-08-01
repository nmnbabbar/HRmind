import pytest
from backend.orchestration.planner import planner_node, MAX_QUERY_LENGTH
from backend.state import make_initial_state
from unittest.mock import patch, MagicMock, AsyncMock
from langchain_core.exceptions import OutputParserException

# --- Tests for Planner Guardrails (Input Length, Profanity, JSON Schema) ---

def test_input_length_guardrail():
    # Create a query that is longer than MAX_QUERY_LENGTH
    long_query = "A" * (MAX_QUERY_LENGTH + 10)
    state = make_initial_state(long_query, session_id="test")
    
    result = planner_node(state)
    
    # Assert it short-circuited and returned a fallback without hitting LLM
    assert result["plan"]["agents"] == []
    assert "too long" in result["final_answer"]
    assert "too long" in result["previous_turn"]["answer"]

def test_profanity_guardrail():
    # A query with a profanity word (the word "fuck" is blocked by better_profanity by default)
    state = make_initial_state("What the fuck is this HR policy?", session_id="test")
    
    result = planner_node(state)
    
    # Assert it short-circuited due to profanity
    assert result["plan"]["agents"] == []
    assert "professional tone" in result["final_answer"]
    assert "professional tone" in result["previous_turn"]["answer"]

@patch("backend.orchestration.planner.ChatOpenAI")
def test_json_schema_guardrail(mock_chat_openai):
    # Mock the LLM chain to raise an OutputParserException to simulate malformed JSON
    mock_llm = MagicMock()
    mock_chat_openai.return_value.with_structured_output.return_value = mock_llm
    
    # Let's mock the chain | llm pipeline inside planner_node
    with patch("backend.orchestration.planner.ChatPromptTemplate.from_messages") as mock_prompt:
        mock_chain = MagicMock()
        mock_prompt.return_value.__or__.return_value = mock_chain
        # Make the invoke method raise an exception
        mock_chain.invoke.side_effect = OutputParserException("Malformed JSON")
        
        state = make_initial_state("Tell me about my salary", session_id="test")
        result = planner_node(state)
        
        # It should catch the exception and return a fallback plan safely
        assert result["plan"]["agents"] == []
        assert "internal error" in result["plan"]["reasoning"]
        assert "internal error" in result["final_answer"]


# --- Tests for RAG Agent Guardrail (Low Confidence) ---
from backend.agents.rag_agent.rag_agent import RAGAgent
from backend.agents.rag_agent.context_builder import NO_CONTEXT_ANSWER
import asyncio

@pytest.mark.asyncio
async def test_low_confidence_rag_guardrail():
    # Setup mocks
    mock_llm = MagicMock()
    mock_embedding = MagicMock()
    mock_embedding.embed_query = AsyncMock(return_value=[0.1]*768)
    mock_vector = MagicMock()
    mock_vector.similarity_search = AsyncMock(return_value=[])
    mock_bm25 = MagicMock()
    mock_bm25.get_top_n.return_value = []
    
    agent = RAGAgent(
        llm=mock_llm,
        embedding_service=mock_embedding,
        vector_repo=mock_vector,
        bm25_index=mock_bm25
    )
    
    state = make_initial_state("This query has nothing to do with HR policies", session_id="test")
    
    # The hybrid search will return empty chunks (no chunks retrieved)
    result = await agent.run(state)
    
    # Assert it short-circuits to NO_CONTEXT_ANSWER without calling LLM
    assert result.success is True
    assert result.answer == NO_CONTEXT_ANSWER
    assert result.metadata["retrieved_chunks"] == 0
    # Make sure LLM wasn't called
    mock_llm.ainvoke.assert_not_called()

@pytest.mark.asyncio
async def test_low_score_rag_guardrail():
    # Setup mocks
    mock_llm = MagicMock()
    mock_embedding = MagicMock()
    mock_embedding.embed_query = AsyncMock(return_value=[0.1]*768)
    
    # We will mock reciprocal_rank_fusion directly to return chunks with low rrf_score
    from langchain_core.documents import Document
    low_score_doc = Document(page_content="test", metadata={"rrf_score": 0.001}) # below 0.01 threshold
    
    mock_vector_repo = MagicMock()
    mock_vector_repo.similarity_search = AsyncMock(return_value=[])
    
    agent = RAGAgent(
        llm=mock_llm,
        embedding_service=mock_embedding,
        vector_repo=mock_vector_repo,
        bm25_index=MagicMock()
    )
    
    state = make_initial_state("Some query", session_id="test")
    
    with patch("backend.agents.rag_agent.rag_agent.reciprocal_rank_fusion") as mock_rrf:
        mock_rrf.return_value = [low_score_doc]
        result = await agent.run(state)
        
        # Should also short-circuit
        assert result.success is True
        assert result.answer == NO_CONTEXT_ANSWER
        mock_llm.ainvoke.assert_not_called()
