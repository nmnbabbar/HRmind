"""
backend/agents/rag_agent/guardrails.py
=======================================
RAG-specific guardrails:
1. TopicGuardrail   — rejects non-HR queries via LLM classification
2. GroundingGuardrail — post-generation NLI check: answer must follow from context

Both implement the GuardrailStrategy protocol.
"""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from backend.base.guardrail import GuardrailStrategy
from backend.state import GuardrailResult

logger = logging.getLogger(__name__)


class TopicGuardrail(GuardrailStrategy):
    """
    Rejects queries that are not HR-related using an LLM classifier.

    Implementation: a single lightweight LLM call (~50 input tokens).
    The classifier prompt forces a YES/NO answer — no open-ended generation.
    Blocked queries return a user-friendly explanation.

    This is intentionally a broad filter — it accepts anything plausibly
    HR-related (leave policies, salaries, contracts, conduct, etc.) and only
    blocks clearly off-topic queries (cooking recipes, sports, coding help, etc.).
    """

    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm

    async def check(self, query: str) -> GuardrailResult:
        """Classify the query as HR-related or not."""
        try:
            messages = [
                SystemMessage(
                    content=(
                        "You are a topic classifier for an HR assistant. "
                        "Determine if the following query is related to Human Resources "
                        "(employment, leave, payroll, policies, contracts, performance, "
                        "recruitment, onboarding, benefits, workplace conduct, etc.).\n\n"
                        "Answer with exactly one word: YES or NO."
                    )
                ),
                HumanMessage(content=f"Query: {query}"),
            ]
            response = await self._llm.ainvoke(messages)
            answer = response.content.strip().upper()

            if answer.startswith("YES"):
                return GuardrailResult.ok()
            else:
                logger.info("TopicGuardrail: blocked non-HR query: %r", query)
                return GuardrailResult.fail(
                    reason=(
                        "I can only assist with HR-related questions such as leave policies, "
                        "employment contracts, payroll, and workplace guidelines. "
                        "Please rephrase your question to be HR-related."
                    ),
                    guardrail_name="TopicGuardrail",
                )

        except Exception as exc:
            # On classifier failure, fail open (allow the query through)
            # Better to answer a borderline query than to break the system
            logger.warning(
                "TopicGuardrail check failed with exception: %s — allowing query through.",
                exc,
            )
            return GuardrailResult.ok()


class GroundingGuardrail(GuardrailStrategy):
    """
    Post-generation grounding check for RAG answers.

    Verifies that the generated answer is grounded in the retrieved context
    and does not hallucinate facts not present in the source documents.

    Note: this guardrail is used AFTER answer generation, not before.
    It is called by RAGAgent directly (not via CompositeGuardrail).

    Implementation: YES/NO LLM check — does the answer follow only from
    the provided context? If NO, the agent returns a safer fallback answer.
    """

    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm

    async def check_grounding(
        self, query: str, context: str, answer: str
    ) -> GuardrailResult:
        """
        Check whether the answer is grounded in the provided context.

        Parameters
        ----------
        query : str
            Original user query.
        context : str
            Retrieved context chunks passed to the LLM.
        answer : str
            Generated answer to verify.

        Returns
        -------
        GuardrailResult
            passed=True if answer is grounded; passed=False if hallucination detected.
        """
        try:
            messages = [
                SystemMessage(
                    content=(
                        "You are a fact-checking assistant. Your job is to determine "
                        "if an AI-generated answer is supported ONLY by the provided context.\n\n"
                        "Rules:\n"
                        "- Answer YES if every factual claim in the answer can be found in the context.\n"
                        "- Answer NO if the answer contains facts not present in the context, "
                        "  makes up statistics, or contradicts the context.\n"
                        "- Answer with exactly one word: YES or NO."
                    )
                ),
                HumanMessage(
                    content=(
                        f"CONTEXT:\n{context[:2000]}\n\n"
                        f"QUESTION: {query}\n\n"
                        f"ANSWER: {answer}\n\n"
                        "Is this answer fully grounded in the context?"
                    )
                ),
            ]
            response = await self._llm.ainvoke(messages)
            verdict = response.content.strip().upper()

            if verdict.startswith("YES"):
                return GuardrailResult.ok()
            else:
                logger.warning(
                    "GroundingGuardrail: answer failed grounding check for query: %r",
                    query[:100],
                )
                return GuardrailResult.fail(
                    reason="Answer may contain information not found in the HR documents.",
                    guardrail_name="GroundingGuardrail",
                )

        except Exception as exc:
            logger.warning(
                "GroundingGuardrail check failed: %s — passing answer through.", exc
            )
            return GuardrailResult.ok()

    # Make this compatible with the standard GuardrailStrategy protocol
    # (for use in CompositeGuardrail if needed, though typically called directly)
    async def check(self, query: str) -> GuardrailResult:
        """
        Standard protocol method — passes through (requires context + answer to check).

        For pre-query topic filtering, use TopicGuardrail instead.
        For post-generation grounding, call check_grounding() directly.
        """
        return GuardrailResult.ok()
