"""
backend/memory/context_builder.py
==================================
Agent-scoped context construction with hard token budgets.

Purpose
-------
Prevent context bleed: each agent receives ONLY the context relevant to its
domain. This is the primary mechanism for keeping context windows small and
ensuring SQL schema never leaks into RAG prompts (and vice versa).

Priority order for token budget allocation (highest → lowest):
    1. System prompt          (always preserved)
    2. Current user query     (always preserved)
    3. Rolling summary        (compressed history)
    4. Recent turns verbatim  (last N turns)
    5. Agent-specific context (retrieved chunks / schema / prior results)

When token budget is exceeded, lower-priority content is trimmed first.

Usage
-----
    builder = ContextBuilder()
    conv_context = builder.build_conversation_context(state)
    entity_context = builder.build_entity_context(state)
    prior_results = builder.get_prior_agent_results(state)
"""

from dataclasses import dataclass

from backend.state import AgentResult, GraphState
from backend.utils.token_counter import count_tokens, trim_to_token_budget


@dataclass(frozen=True)
class ContextBudget:
    """
    Immutable token budget for a single agent call.

    Frozen dataclass — create new instances for different budget configurations
    rather than mutating (consistent with the immutability theme in this codebase).
    """

    system_prompt: int = 1000    # reserved for agent system prompt
    current_query: int = 200     # reserved for user's current question
    rolling_summary: int = 600   # max tokens for compressed conversation history
    recent_turns: int = 800      # max tokens for last N verbatim turns
    agent_context: int = 1000    # max tokens for retrieved chunks / schema / OCR

    @property
    def total(self) -> int:
        """Total token budget across all context types."""
        return (
            self.system_prompt
            + self.current_query
            + self.rolling_summary
            + self.recent_turns
            + self.agent_context
        )


class ContextBuilder:
    """
    Builds agent-scoped context slices from the shared GraphState.

    Each agent calls the methods it needs — nothing more.
    SQL Agent: build_conversation_context + build_entity_context (no chunks)
    RAG Agent: build_conversation_context + build_entity_context (no SQL schema)
    DocParser: no conversation context (stateless per file)
    """

    def __init__(self, budget: ContextBudget | None = None) -> None:
        self._budget = budget or ContextBudget()

    # ── Public API ─────────────────────────────────────────────────────────────

    def build_conversation_context(self, state: GraphState) -> str:
        """
        Build compressed conversation context from summary + recent turns.

        Returns a formatted string ready for injection into an agent system prompt.
        Returns empty string if no history exists (first turn of a conversation).
        """
        parts: list[str] = []

        summary = state.get("conversation_summary", "")
        if summary:
            trimmed_summary = trim_to_token_budget(summary, self._budget.rolling_summary)
            parts.append(f"[Conversation Summary]\n{trimmed_summary}")

        recent = state.get("recent_turns", [])
        if recent:
            recent_text = self._format_turns(recent)
            trimmed_recent = trim_to_token_budget(recent_text, self._budget.recent_turns)
            parts.append(f"[Recent Messages]\n{trimmed_recent}")

        return "\n\n".join(parts)

    def build_entity_context(self, state: GraphState) -> str:
        """
        Build entity context for coreference resolution in follow-up questions.

        Converts the entity_store dict into a compact string.
        Example: "Known entities — person: Alice Johnson, date: 2024-01-15"

        Returns empty string if entity store is empty.
        """
        entity_store = state.get("entity_store", {})
        if not entity_store:
            return ""
        entities = ", ".join(
            f"{label}: {value}" for label, value in entity_store.items()
        )
        return f"[Known Entities]\n{entities}"

    def get_prior_agent_results(self, state: GraphState) -> list[AgentResult]:
        """
        Return successfully completed AgentResults from earlier pipeline steps.

        Used by sequential agents that consume prior output
        (e.g. RAG agent consuming DocParser extraction results).
        """
        raw_results = state.get("agent_results", [])
        return [
            AgentResult.from_dict(r)
            for r in raw_results
            if r.get("success", False)
        ]

    def summarise_prior_results(self, state: GraphState) -> str:
        """
        Compact string representation of prior agent outputs for prompt injection.

        Only includes successful results. Trimmed to agent_context budget.
        """
        results = self.get_prior_agent_results(state)
        if not results:
            return ""

        lines: list[str] = []
        for r in results:
            lines.append(f"[{r.agent_name} output]\n{r.answer}")

        combined = "\n\n".join(lines)
        return trim_to_token_budget(combined, self._budget.agent_context)

    def context_token_counts(self, state: GraphState) -> dict[str, int]:
        """
        Debug helper: return token counts for each context component.

        Useful for tuning budget values and diagnosing context size issues.
        """
        return {
            "conversation_summary": count_tokens(state.get("conversation_summary", "")),
            "recent_turns": count_tokens(self._format_turns(state.get("recent_turns", []))),
            "entity_store": count_tokens(self.build_entity_context(state)),
            "prior_results": count_tokens(self.summarise_prior_results(state)),
        }

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _format_turns(turns: list[dict]) -> str:
        """Format recent turns list into a readable conversation string."""
        lines: list[str] = []
        for turn in turns:
            role = turn.get("role", "unknown").capitalize()
            content = turn.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)
