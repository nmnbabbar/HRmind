"""
tests/conftest.py
=================
Shared pytest fixtures used across all test modules.

Fixtures defined here:
- sample_state      : minimal valid GraphState for testing
- mock_llm          : AsyncMock-based fake LLM returning predefined responses
- passthrough_guard : CompositeGuardrail with no strategies (always passes)
- blocking_guard    : CompositeGuardrail that always blocks (for guardrail tests)
- context_builder   : ContextBuilder with default budget
"""

import os
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest

# ── Force test environment settings before importing config ───────────────────
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key-for-testing-only")


from backend.base.guardrail import CompositeGuardrail, PassthroughGuardrail
from backend.config import get_settings
from backend.memory.context_builder import ContextBudget, ContextBuilder
from backend.state import AgentResult, GraphState, PlannerOutput, make_initial_state


# ── Settings ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Clear lru_cache on Settings so each test starts with a clean config."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ── Graph state ───────────────────────────────────────────────────────────────

@pytest.fixture
def sample_state() -> GraphState:
    """Minimal valid GraphState for testing node functions."""
    return make_initial_state(
        query="What is the maternity leave policy?",
        session_id="test-session-001",
    )


@pytest.fixture
def state_with_history() -> GraphState:
    """GraphState with conversation history for memory testing."""
    state = make_initial_state(
        query="What is the maternity leave policy?",
        session_id="test-session-001",
    )
    state["conversation_summary"] = "The user previously asked about the alcohol policy."
    state["recent_turns"] = [
        {"role": "user", "content": "What is the alcohol policy?"},
        {"role": "assistant", "content": "The alcohol policy prohibits..."},
        {"role": "user", "content": "What about drugs?"},
    ]
    state["entity_store"] = {"person": "Alice Johnson", "org": "Engineering"}
    return state


# ── Mock LLM ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm():
    """
    Fake LangChain-compatible LLM that returns a predefined response.

    Usage::
        async def test_something(mock_llm):
            mock_llm.ainvoke.return_value.content = "Custom response"
            agent = MyAgent(llm=mock_llm)
    """
    llm = MagicMock()
    response = MagicMock()
    response.content = "This is a mock LLM response."
    llm.ainvoke = AsyncMock(return_value=response)
    llm.model_name = "mock-gpt-4o-mini"
    return llm


# ── Guardrails ────────────────────────────────────────────────────────────────

@pytest.fixture
def passthrough_guard() -> PassthroughGuardrail:
    """Guardrail that always passes — for testing agents without guardrail logic."""
    return PassthroughGuardrail()


@pytest.fixture
def always_blocking_guard():
    """Guardrail that always blocks — for testing guardrail rejection paths."""
    from backend.state import GuardrailResult

    class AlwaysBlock:
        async def check(self, query: str) -> GuardrailResult:
            return GuardrailResult.fail("Blocked for testing", "AlwaysBlockGuardrail")

    return CompositeGuardrail([AlwaysBlock()])


# ── Context building ──────────────────────────────────────────────────────────

@pytest.fixture
def context_builder() -> ContextBuilder:
    """ContextBuilder with default ContextBudget."""
    return ContextBuilder()


@pytest.fixture
def tight_budget_builder() -> ContextBuilder:
    """ContextBuilder with very tight budgets for trimming tests."""
    return ContextBuilder(
        budget=ContextBudget(
            system_prompt=100,
            current_query=50,
            rolling_summary=50,   # very tight — forces trimming
            recent_turns=100,
            agent_context=100,
        )
    )


# ── Agent results ─────────────────────────────────────────────────────────────

@pytest.fixture
def successful_rag_result() -> AgentResult:
    return AgentResult(
        agent_name="RAGAgent",
        success=True,
        answer="The maternity leave policy entitles employees to 52 weeks...",
        sources=["Maternity-Policy.docx#p1"],
        structured_data={"chunks": ["..."], "sources": ["Maternity-Policy.docx"]},
        metadata={"duration_ms": 342, "model": "gpt-4o-mini"},
    )


@pytest.fixture
def failed_sql_result() -> AgentResult:
    return AgentResult.failure(
        agent_name="SQLAgent",
        error="SQL generation failed: ambiguous column reference",
        metadata={"duration_ms": 89},
    )
