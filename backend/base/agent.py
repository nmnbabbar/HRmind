"""
backend/base/agent.py
=====================
Abstract base class for all HrMind agents.

Enforces the execute-in-code pattern:
    1. Check guardrails (sync, no LLM)
    2. Execute domain logic in Python (retrieval / SQL / OCR)
    3. Pass results to LLM for synthesis (ONE LLM call per agent)
    4. Return AgentResult

Design principles applied
--------------------------
- Single Responsibility: each subclass handles exactly one domain
- Open/Closed: new agents extend BaseAgent without changing existing code
- Liskov Substitution: all agents are interchangeable via BaseAgent interface
- Dependency Inversion: depends on BaseChatModel abstraction, not ChatOpenAI

Subclasses
----------
RAGAgent      — Hybrid search + synthesis (Phase 2)
SQLAgent      — NL → SQL → execute → explain (Phase 3)
DocParserAgent — OCR → extract → return structured data (Phase 4)
"""

import time
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models import BaseChatModel

from backend.base.guardrail import CompositeGuardrail, GuardrailStrategy, PassthroughGuardrail
from backend.state import AgentResult, GraphState
from backend.utils.log import get_logger

logger = get_logger(__name__)


class BaseAgent(ABC):
    """
    Abstract base for all HrMind agents.

    Constructor Parameters
    ----------------------
    llm : BaseChatModel
        The language model to use for synthesis. Injected by AgentFactory.
        Subclasses should use self._llm — never instantiate models directly.
    guardrails : list[GuardrailStrategy] | None
        List of guardrail strategies to run before agent execution.
        Composed into a CompositeGuardrail (Chain of Responsibility).
        Pass [] or None to use a PassthroughGuardrail (no checks).
    """

    def __init__(
        self,
        llm: BaseChatModel,
        guardrails: list[GuardrailStrategy] | None = None,
    ) -> None:
        self._llm = llm
        self._guardrails = (
            CompositeGuardrail(guardrails) if guardrails else PassthroughGuardrail()  # type: ignore[arg-type]
        )

    @property
    def name(self) -> str:
        """Agent identifier — used in AgentResult.agent_name and logging."""
        return self.__class__.__name__

    @abstractmethod
    async def run(self, state: GraphState) -> AgentResult:
        """
        Execute the agent's core logic.

        Must be implemented by every subclass. The typical pattern is:
            1. Extract relevant context from state
            2. Run guardrails (use self._run_with_guardrails)
            3. Execute domain logic (retrieval / SQL / OCR) — no LLM tool calls
            4. Build a prompt with the execution results
            5. Call self._llm.ainvoke(messages) for synthesis
            6. Return AgentResult

        Parameters
        ----------
        state : GraphState
            The current LangGraph state. Read query, plan, agent_results, etc.
            Do NOT mutate state — return a new AgentResult instead.

        Returns
        -------
        AgentResult
            Always return an AgentResult (even on error — use AgentResult.failure()).
            Never raise exceptions from this method; catch and wrap them.
        """
        ...

    async def _run_with_guardrails(self, query: str) -> AgentResult | None:
        """
        Run guardrail checks. Returns a failure AgentResult if blocked, else None.

        Usage in subclasses::

            guard_block = await self._run_with_guardrails(query)
            if guard_block:
                return guard_block   # early exit
            # proceed with agent logic...
        """
        result = await self._guardrails.check(query)
        if not result.passed:
            logger.warning(
                "guardrail_blocked",
                agent=self.name,
                guardrail=result.guardrail_name,
                reason=result.reason,
            )
            return AgentResult.failure(
                agent_name=self.name,
                error=f"Blocked by {result.guardrail_name}: {result.reason}",
                metadata={"guardrail": result.guardrail_name},
            )
        return None

    def _timed_result(
        self,
        start_time: float,
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Build a metadata dict with timing information.

        Usage::
            t0 = time.monotonic()
            # ... do work ...
            metadata = self._timed_result(t0, {"model": self._llm.model_name})
        """
        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        meta: dict[str, Any] = {"duration_ms": elapsed_ms, "agent": self.name}
        if extra_metadata:
            meta.update(extra_metadata)
        return meta
