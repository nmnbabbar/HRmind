"""
backend/state.py
================
LangGraph shared state and core data contracts.

All data flowing through the graph is typed here. The GraphState TypedDict
is the single source of truth for what information travels between nodes.

Key design decisions:
- agent_results uses Annotated[list, operator.add] so multiple agents can
  append results without overwriting each other (LangGraph reducer pattern).
- plan and agent_results store plain dicts (not Pydantic models) for safe
  LangGraph serialization / MemorySaver checkpointing.
- PlannerOutput and AgentResult provide typed construction/validation on top.
"""

import operator
from dataclasses import dataclass, field, asdict
from typing import Annotated, Any, Literal
from typing_extensions import TypedDict

from pydantic import BaseModel, field_validator


# ── Planner output ─────────────────────────────────────────────────────────────

class PlannerOutput(BaseModel):
    """
    Structured output from the Planner node.

    The Planner returns an ordered list of agent names.
    The Router iterates this list in code (no conditional LLM routing).

    Example
    -------
    PlannerOutput(
        agents=["doc_parser", "rag"],
        queries={"doc_parser": "extract notice period", "rag": "notice periods policy"},
        parallel=False,   # doc_parser output feeds into rag → must be sequential
        reasoning="User uploaded a contract and wants policy validation"
    )
    """

    agents: list[Literal["rag", "sql", "doc_parser"]]
    queries: dict[str, str]   # per-agent query rewrite / instruction
    parallel: bool            # True only when agents have no output dependency
    reasoning: str = ""       # stored in metadata, never shown to user

    @field_validator("agents")
    @classmethod
    def agents_not_empty(cls, v: list) -> list:
        if not v:
            raise ValueError("agents list must contain at least one agent")
        return v

    @field_validator("queries")
    @classmethod
    def queries_match_agents(cls, v: dict, info: Any) -> dict:
        # Every agent must have a corresponding query
        agents = info.data.get("agents", [])
        missing = [a for a in agents if a not in v]
        if missing:
            raise ValueError(f"Missing queries for agents: {missing}")
        return v

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, d: dict) -> "PlannerOutput":
        return cls.model_validate(d)


# ── Guardrail result ───────────────────────────────────────────────────────────

@dataclass
class GuardrailResult:
    """
    Result from a single guardrail check.

    Immutable by convention — create a new instance rather than mutating.
    """

    passed: bool
    reason: str = ""
    guardrail_name: str = ""

    @classmethod
    def ok(cls) -> "GuardrailResult":
        return cls(passed=True)

    @classmethod
    def fail(cls, reason: str, guardrail_name: str = "") -> "GuardrailResult":
        return cls(passed=False, reason=reason, guardrail_name=guardrail_name)


# ── Agent result ───────────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    """
    Standardised output contract for all agents.

    Every agent — RAG, SQL, DocParser — returns an AgentResult.
    The Combiner node reads agent_results (list of dicts) and synthesizes
    the final answer using conflict resolution policy.

    stored_data contains agent-specific structured output:
    - RAG:        {"chunks": [...], "sources": [...]}
    - SQL:        {"sql_query": "...", "columns": [...], "rows": [...]}
    - DocParser:  {"document_type": "...", "extracted_fields": {...}}

    Serialization
    -------------
    AgentResult.to_dict()  → stored in GraphState["agent_results"] as dict
    AgentResult.from_dict() → reconstructed in Combiner / tests
    """

    agent_name: str
    success: bool
    answer: str                                   # natural language answer
    sources: list[str] = field(default_factory=list)
    structured_data: dict[str, Any] | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AgentResult":
        return cls(**d)

    @classmethod
    def failure(
        cls,
        agent_name: str,
        error: str,
        metadata: dict | None = None,
    ) -> "AgentResult":
        """Convenience constructor for error states."""
        return cls(
            agent_name=agent_name,
            success=False,
            answer="",
            error=error,
            metadata=metadata or {},
        )


# ── LangGraph shared state ─────────────────────────────────────────────────────

class GraphState(TypedDict, total=False):
    """
    Shared state flowing through the LangGraph StateGraph.

    All fields are optional (total=False) so nodes only need to return
    the keys they modify — LangGraph merges partials automatically.

    Special reducer:
        agent_results uses operator.add so each agent node APPENDS its
        AgentResult dict to the list (not overwrites). The Combiner reads
        the accumulated list.
    """

    # ── Input ──────────────────────────────────────────────────────────────
    query: str                           # current user query
    session_id: str                      # conversation thread identifier

    # ── Memory / context ───────────────────────────────────────────────────
    conversation_summary: str            # rolling LLM-compressed history
    recent_turns: list[dict]             # last N turns verbatim (role + content)
    entity_store: dict[str, str]         # {"person": "Alice", "date": "2024-01-15"}

    # ── Orchestration ──────────────────────────────────────────────────────
    plan: dict | None                    # serialized PlannerOutput (use .from_dict)

    # Reducer: each agent node appends its result dict — never overwrites
    agent_results: Annotated[list[dict], operator.add]

    # ── Output ─────────────────────────────────────────────────────────────
    final_answer: str                    # synthesized answer from Combiner

    # ── File handling (DocParser) ──────────────────────────────────────────
    uploaded_file_path: str | None       # absolute path to uploaded file in /uploads

    # ── Error handling ─────────────────────────────────────────────────────
    error: str | None                    # set if a node fails fatally


def make_initial_state(query: str, session_id: str) -> GraphState:
    """
    Factory: create a valid initial GraphState for a new turn.

    Separates state construction from graph execution — keeps node code clean.
    """
    return GraphState(
        query=query,
        session_id=session_id,
        conversation_summary="",
        recent_turns=[],
        entity_store={},
        plan=None,
        agent_results=[],
        final_answer="",
        uploaded_file_path=None,
        error=None,
    )
