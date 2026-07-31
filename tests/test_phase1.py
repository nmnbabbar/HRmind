"""
tests/test_phase1.py
=====================
Phase 1 test suite — Foundation & Infrastructure.

Tests are grouped by module. All tests use fixtures from conftest.py.
No external services required — all LLM/spaCy calls are mocked.

Run with:
    docker compose exec backend pytest tests/test_phase1.py -v
    # or locally (no docker):
    pytest tests/test_phase1.py -v
"""

import asyncio
import operator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.base.guardrail import CompositeGuardrail, PassthroughGuardrail
from backend.memory.context_builder import ContextBudget, ContextBuilder
from backend.memory.entity_store import EntityStore
from backend.memory.summarizer import RollingSummarizer
from backend.state import (
    AgentResult,
    GraphState,
    GuardrailResult,
    PlannerOutput,
    make_initial_state,
)
from backend.utils.token_counter import (
    count_tokens,
    fits_in_budget,
    trim_to_token_budget,
)


# =============================================================================
# State + Data Contracts
# =============================================================================

class TestGraphState:
    def test_make_initial_state_has_all_fields(self):
        state = make_initial_state(query="test query", session_id="sess-001")

        assert state["query"] == "test query"
        assert state["session_id"] == "sess-001"
        assert state["conversation_summary"] == ""
        assert state["recent_turns"] == []
        assert state["entity_store"] == {}
        assert state["plan"] is None
        assert state["agent_results"] == []
        assert state["final_answer"] == ""
        assert state["uploaded_file_path"] is None
        assert state["error"] is None

    def test_agent_results_reducer_appends(self):
        """
        LangGraph reducer test: operator.add on lists should concatenate.
        This verifies the Annotated[list, operator.add] pattern works correctly.
        """
        list_a = [{"agent_name": "rag", "success": True}]
        list_b = [{"agent_name": "sql", "success": True}]

        combined = operator.add(list_a, list_b)
        assert len(combined) == 2
        assert combined[0]["agent_name"] == "rag"
        assert combined[1]["agent_name"] == "sql"


class TestAgentResult:
    def test_successful_result_defaults(self):
        result = AgentResult(
            agent_name="TestAgent",
            success=True,
            answer="Some answer",
        )
        assert result.sources == []
        assert result.structured_data is None
        assert result.error is None
        assert result.metadata == {}

    def test_failure_factory(self):
        result = AgentResult.failure(
            agent_name="SQLAgent",
            error="Connection refused",
            metadata={"duration_ms": 50},
        )
        assert result.success is False
        assert result.answer == ""
        assert result.error == "Connection refused"
        assert result.metadata["duration_ms"] == 50

    def test_to_dict_roundtrip(self, successful_rag_result):
        """AgentResult serializes to dict and back without data loss."""
        d = successful_rag_result.to_dict()

        assert isinstance(d, dict)
        assert d["agent_name"] == "RAGAgent"
        assert d["success"] is True

        restored = AgentResult.from_dict(d)
        assert restored.agent_name == successful_rag_result.agent_name
        assert restored.answer == successful_rag_result.answer
        assert restored.sources == successful_rag_result.sources

    def test_to_dict_is_json_serializable(self, successful_rag_result):
        """Dicts stored in LangGraph state must be JSON serializable."""
        import json
        d = successful_rag_result.to_dict()
        serialized = json.dumps(d)  # should not raise
        assert "RAGAgent" in serialized


class TestPlannerOutput:
    def test_valid_single_agent_plan(self):
        plan = PlannerOutput(
            agents=["rag"],
            queries={"rag": "maternity leave policy"},
            parallel=False,
        )
        assert plan.agents == ["rag"]
        assert plan.queries["rag"] == "maternity leave policy"
        assert plan.parallel is False

    def test_valid_multi_agent_parallel_plan(self):
        plan = PlannerOutput(
            agents=["rag", "sql"],
            queries={"rag": "alcohol policy", "sql": "employees on leave"},
            parallel=True,
        )
        assert len(plan.agents) == 2
        assert plan.parallel is True

    def test_empty_agents_rejected(self):
        with pytest.raises(Exception):  # pydantic ValidationError
            PlannerOutput(agents=[], queries={}, parallel=False)

    def test_missing_query_for_agent_rejected(self):
        with pytest.raises(Exception):
            PlannerOutput(
                agents=["rag", "sql"],
                queries={"rag": "some query"},  # missing "sql"
                parallel=False,
            )

    def test_roundtrip_via_dict(self):
        plan = PlannerOutput(
            agents=["doc_parser", "rag"],
            queries={"doc_parser": "extract notice period", "rag": "notice policy"},
            parallel=False,
            reasoning="sequential: doc_parser output feeds into rag",
        )
        d = plan.to_dict()
        restored = PlannerOutput.from_dict(d)
        assert restored.agents == plan.agents
        assert restored.parallel == plan.parallel


class TestGuardrailResult:
    def test_ok_factory(self):
        result = GuardrailResult.ok()
        assert result.passed is True
        assert result.reason == ""

    def test_fail_factory(self):
        result = GuardrailResult.fail("Non-HR query", "TopicGuardrail")
        assert result.passed is False
        assert result.reason == "Non-HR query"
        assert result.guardrail_name == "TopicGuardrail"


# =============================================================================
# Guardrails
# =============================================================================

class TestCompositeGuardrail:
    @pytest.mark.asyncio
    async def test_empty_composite_always_passes(self):
        guard = CompositeGuardrail([])
        result = await guard.check("any query")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_passthrough_always_passes(self):
        guard = PassthroughGuardrail()
        result = await guard.check("DROP TABLE employees")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_composite_passes_when_all_pass(self):
        guard = CompositeGuardrail([PassthroughGuardrail(), PassthroughGuardrail()])
        result = await guard.check("What is the leave policy?")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_composite_stops_at_first_failure(self):
        """Chain of Responsibility: first failure stops evaluation."""
        call_order = []

        class TrackingGuardrail:
            def __init__(self, name: str, should_fail: bool):
                self.name = name
                self.should_fail = should_fail

            async def check(self, query: str) -> GuardrailResult:
                call_order.append(self.name)
                if self.should_fail:
                    return GuardrailResult.fail(f"{self.name} blocked", self.name)
                return GuardrailResult.ok()

        guard = CompositeGuardrail([
            TrackingGuardrail("first", should_fail=False),
            TrackingGuardrail("second", should_fail=True),   # blocks here
            TrackingGuardrail("third", should_fail=False),   # should NOT be called
        ])

        result = await guard.check("some query")

        assert result.passed is False
        assert result.guardrail_name == "second"
        assert "first" in call_order
        assert "second" in call_order
        assert "third" not in call_order  # chain stopped at second

    @pytest.mark.asyncio
    async def test_composite_add_returns_new_instance(self):
        """add() is immutable — original composite is not modified."""
        base = CompositeGuardrail([PassthroughGuardrail()])
        extended = base.add(PassthroughGuardrail())

        assert extended is not base
        assert len(base.strategy_names) == 1
        assert len(extended.strategy_names) == 2

    def test_strategy_names(self):
        guard = CompositeGuardrail([PassthroughGuardrail(), PassthroughGuardrail()])
        names = guard.strategy_names
        assert names == ["PassthroughGuardrail", "PassthroughGuardrail"]

    @pytest.mark.asyncio
    async def test_always_blocking_fixture(self, always_blocking_guard):
        result = await always_blocking_guard.check("any query")
        assert result.passed is False


# =============================================================================
# Context Builder
# =============================================================================

class TestContextBudget:
    def test_default_total_is_sum_of_parts(self):
        budget = ContextBudget()
        expected = (
            budget.system_prompt
            + budget.current_query
            + budget.rolling_summary
            + budget.recent_turns
            + budget.agent_context
        )
        assert budget.total == expected

    def test_frozen_cannot_be_mutated(self):
        budget = ContextBudget()
        with pytest.raises((TypeError, AttributeError)):
            budget.system_prompt = 9999  # type: ignore[misc]

    def test_custom_budget(self):
        budget = ContextBudget(system_prompt=500, rolling_summary=300)
        assert budget.system_prompt == 500
        assert budget.rolling_summary == 300
        # others retain defaults
        assert budget.current_query == 200


class TestContextBuilder:
    def test_build_conversation_context_empty_state(self, context_builder, sample_state):
        """Empty history → empty context string."""
        result = context_builder.build_conversation_context(sample_state)
        assert result == ""

    def test_build_conversation_context_with_summary(
        self, context_builder, state_with_history
    ):
        result = context_builder.build_conversation_context(state_with_history)
        assert "[Conversation Summary]" in result
        assert "alcohol policy" in result

    def test_build_conversation_context_with_recent_turns(
        self, context_builder, state_with_history
    ):
        result = context_builder.build_conversation_context(state_with_history)
        assert "[Recent Messages]" in result
        assert "What is the alcohol policy?" in result

    def test_build_entity_context_empty(self, context_builder, sample_state):
        result = context_builder.build_entity_context(sample_state)
        assert result == ""

    def test_build_entity_context_with_entities(
        self, context_builder, state_with_history
    ):
        result = context_builder.build_entity_context(state_with_history)
        assert "[Known Entities]" in result
        assert "Alice Johnson" in result
        assert "Engineering" in result

    def test_tight_budget_trims_summary(self, tight_budget_builder, state_with_history):
        """With a 50-token summary budget, long text is trimmed."""
        long_summary = "A" * 1000  # 1000 chars ≈ 250 tokens — exceeds 50-token budget
        state_with_history["conversation_summary"] = long_summary
        result = tight_budget_builder.build_conversation_context(state_with_history)
        assert "[trimmed]" in result

    def test_get_prior_agent_results_filters_failures(
        self, context_builder, successful_rag_result, failed_sql_result
    ):
        state = make_initial_state("test", "sess-001")
        state["agent_results"] = [
            successful_rag_result.to_dict(),
            failed_sql_result.to_dict(),
        ]
        prior = context_builder.get_prior_agent_results(state)

        # Only successful results returned
        assert len(prior) == 1
        assert prior[0].agent_name == "RAGAgent"

    def test_context_token_counts_returns_dict(
        self, context_builder, state_with_history
    ):
        counts = context_builder.context_token_counts(state_with_history)
        assert isinstance(counts, dict)
        assert "conversation_summary" in counts
        assert "recent_turns" in counts
        assert "entity_store" in counts
        assert all(isinstance(v, int) for v in counts.values())


# =============================================================================
# Token Counter
# =============================================================================

class TestTokenCounter:
    def test_empty_string_returns_zero(self):
        assert count_tokens("") == 0

    def test_known_string_token_count(self):
        # "Hello, world!" is 4 tokens in cl100k_base
        count = count_tokens("Hello, world!")
        # Allow for fallback approximation (should be 1–10)
        assert 1 <= count <= 20

    def test_longer_text_has_more_tokens_than_shorter(self):
        short = count_tokens("Hi")
        long = count_tokens("This is a much longer sentence with many more words.")
        assert long > short

    def test_fits_in_budget_short_text(self):
        assert fits_in_budget("short text", max_tokens=1000) is True

    def test_fits_in_budget_exceeds_limit(self):
        # 10,000 char text will definitely exceed 10 tokens
        very_long = "word " * 5000
        assert fits_in_budget(very_long, max_tokens=10) is False

    def test_trim_no_op_when_within_budget(self):
        text = "This fits fine."
        result = trim_to_token_budget(text, max_tokens=1000)
        assert result == text

    def test_trim_truncates_and_appends_marker(self):
        long_text = "word " * 2000  # ~2000 tokens
        result = trim_to_token_budget(long_text, max_tokens=50)
        assert "[trimmed]" in result
        assert len(result) < len(long_text)


# =============================================================================
# Entity Store
# =============================================================================

class TestEntityStore:
    def test_initial_store_is_empty(self):
        store = EntityStore()
        assert store.get() == {}

    def test_clear_resets_store(self):
        store = EntityStore()
        store._store["person"] = "Alice"
        store.clear()
        assert store.get() == {}

    @pytest.mark.asyncio
    async def test_extract_returns_dict_when_spacy_unavailable(self):
        """Graceful degradation: if spaCy is not available, returns empty dict."""
        store = EntityStore()
        # Patch _load_nlp to return None (model not installed)
        with patch("backend.memory.entity_store._load_nlp", return_value=None):
            result = await store.extract_and_update("Alice Johnson is the manager.")
        # Should return empty dict, not raise
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_extract_with_mock_spacy(self):
        """Test extraction logic using a mocked spaCy doc."""
        store = EntityStore()

        # Create a mock entity
        mock_ent = MagicMock()
        mock_ent.label_ = "PERSON"
        mock_ent.text = "Alice Johnson"

        mock_doc = MagicMock()
        mock_doc.ents = [mock_ent]

        mock_nlp = MagicMock(return_value=mock_doc)

        with patch("backend.memory.entity_store._load_nlp", return_value=mock_nlp):
            result = await store.extract_and_update("Alice Johnson is the manager.")

        assert result.get("person") == "Alice Johnson"

    def test_resolve_pronoun_to_person(self):
        store = EntityStore()
        store._store["person"] = "Alice Johnson"

        assert store.resolve("she") == "Alice Johnson"
        assert store.resolve("her") == "Alice Johnson"
        assert store.resolve("him") == "Alice Johnson"  # latest person wins

    def test_resolve_returns_none_for_unknown(self):
        store = EntityStore()
        assert store.resolve("Alice") is None  # not a pronoun
        assert store.resolve("she") is None    # no person in store


# =============================================================================
# Rolling Summarizer
# =============================================================================

class TestRollingSummarizer:
    def test_should_not_summarize_on_zero_turns(self, mock_llm):
        summarizer = RollingSummarizer(llm=mock_llm)
        assert summarizer.should_summarize() is False

    def test_should_not_summarize_before_n_turns(self, mock_llm):
        summarizer = RollingSummarizer(llm=mock_llm)
        for _ in range(9):
            summarizer.increment()
        assert summarizer.should_summarize() is False

    def test_should_summarize_at_exactly_n_turns(self, mock_llm):
        summarizer = RollingSummarizer(llm=mock_llm)
        for _ in range(10):
            summarizer.increment()
        assert summarizer.should_summarize() is True

    def test_should_summarize_at_multiples_of_n(self, mock_llm):
        summarizer = RollingSummarizer(llm=mock_llm)
        trigger_turns = []
        for i in range(1, 35):
            summarizer.increment()
            if summarizer.should_summarize():
                trigger_turns.append(i)
        assert trigger_turns == [10, 20, 30]

    @pytest.mark.asyncio
    async def test_summarize_calls_llm_once(self, mock_llm):
        mock_llm.ainvoke.return_value.content = "Summary of the conversation."
        summarizer = RollingSummarizer(llm=mock_llm)

        result = await summarizer.summarize("User: Hi\nAssistant: Hello")

        assert result == "Summary of the conversation."
        assert mock_llm.ainvoke.call_count == 1

    @pytest.mark.asyncio
    async def test_summarize_includes_existing_summary(self, mock_llm):
        mock_llm.ainvoke.return_value.content = "Updated summary."
        summarizer = RollingSummarizer(llm=mock_llm)

        await summarizer.summarize(
            conversation_text="New: What about drugs policy?",
            existing_summary="Previously: discussed alcohol policy.",
        )

        # Verify existing summary was included in the prompt
        call_args = mock_llm.ainvoke.call_args[0][0]  # list of messages
        prompt_text = " ".join(m.content for m in call_args)
        assert "Previously" in prompt_text

    def test_turn_count_property(self, mock_llm):
        summarizer = RollingSummarizer(llm=mock_llm)
        assert summarizer.turn_count == 0
        summarizer.increment()
        summarizer.increment()
        assert summarizer.turn_count == 2
