"""
backend/agents/rag_agent/context_builder.py
=============================================
Builds the LLM prompt messages for the RAG agent.

Responsibilities
-----------------
1. Enforces agent-scoped context injection (RAG gets ONLY query + summary + chunks;
   never SQL schema, OCR blobs, or prior SQL results).
2. Builds the system prompt with citation instructions.
3. Injects the rolling conversation summary and recent turns (within token budget).
4. Injects retrieved context with citation markers.
5. Appends the current user query.

The system prompt instructs the LLM to:
- Answer ONLY from the retrieved context
- Include inline citations in [Source, page N] format
- List all cited sources at the end
- Admit uncertainty clearly rather than hallucinating
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from backend.state import GraphState

logger = logging.getLogger(__name__)

# ── System prompt ──────────────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """\
You are HrMind, an expert HR assistant. You answer questions about HR policies, \
employment terms, and workplace guidelines STRICTLY based on the provided HR \
document excerpts below.

CRITICAL RULES:
1. Answer ONLY from the provided document excerpts. Do not use any external knowledge.
2. For every factual claim, include an inline citation in this exact format: [Filename, page N]
   Example: "Employees are entitled to 26 weeks of maternity leave [Maternity-Policy.docx, page 2]."
3. If multiple documents support a claim, cite all of them.
4. If the answer is not in the provided excerpts, say: "I could not find this information in the \
available HR documents. Please contact your HR department directly."
5. Never speculate or invent information.
6. At the end of your answer, include a "Sources:" section listing all cited documents.
7. Be precise and professional. Use plain English.

DOCUMENT EXCERPTS:
{context}
"""

# Fallback when no context was retrieved
NO_CONTEXT_ANSWER = (
    "I could not find relevant information in the HR documents for your question. "
    "Please contact your HR department directly or try rephrasing your question."
)


class RAGContextBuilder:
    """
    Builds the list of LangChain messages for a RAG agent invocation.

    Usage
    -----
        builder = RAGContextBuilder(max_summary_tokens=600, max_recent_tokens=800)
        messages = builder.build(state, query, context_str)
        response = await llm.ainvoke(messages)
    """

    def __init__(
        self,
        max_summary_tokens: int = 600,
        max_recent_tokens: int = 800,
    ) -> None:
        self._max_summary_tokens = max_summary_tokens
        self._max_recent_tokens = max_recent_tokens

    def build(
        self,
        state: GraphState,
        query: str,
        context: str,
    ) -> list[SystemMessage | HumanMessage | AIMessage]:
        """
        Build the full message list for the RAG LLM call.

        Message structure (in order):
            1. SystemMessage — RAG rules + retrieved context chunks
            2. (Optional) HumanMessage — conversation summary prefix
            3. (Optional) Recent turns as alternating Human/AI messages
            4. HumanMessage — current query

        Parameters
        ----------
        state : GraphState
            LangGraph state containing conversation history.
        query : str
            Current user query (may be rewritten by Planner).
        context : str
            Formatted context string from format_context_with_citations().

        Returns
        -------
        list[BaseMessage]
            Messages to pass to llm.ainvoke().
        """
        messages: list[SystemMessage | HumanMessage | AIMessage] = []

        # 1. System prompt with retrieved context
        system_content = RAG_SYSTEM_PROMPT.format(context=context if context else "No relevant documents found.")
        messages.append(SystemMessage(content=system_content))

        # 2. Conversation summary (if present) — trimmed to budget
        summary = state.get("conversation_summary", "").strip()
        if summary:
            truncated_summary = self._truncate(summary, self._max_summary_tokens)
            messages.append(
                HumanMessage(
                    content=f"[Conversation summary so far]\n{truncated_summary}"
                )
            )
            messages.append(
                AIMessage(content="Understood. I'll use this context for your question.")
            )

        # 3. Recent turns (last N turns verbatim)
        recent_turns = state.get("recent_turns", [])
        if recent_turns:
            remaining_tokens = self._max_recent_tokens
            for turn in recent_turns[-3:]:  # max 3 recent turns
                role = turn.get("role", "user")
                content = turn.get("content", "")
                truncated = self._truncate(content, remaining_tokens // 3)
                if role == "user":
                    messages.append(HumanMessage(content=truncated))
                else:
                    messages.append(AIMessage(content=truncated))
                remaining_tokens -= len(truncated) // 4  # rough token estimate

        # 4. Current query
        messages.append(HumanMessage(content=query))

        return messages

    @staticmethod
    def _truncate(text: str, max_tokens: int) -> str:
        """
        Truncate text to approximately max_tokens.

        Uses a 4 chars/token approximation — fast and close enough for
        soft limits. Hard truncation only; no summarization.
        """
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "... [truncated]"
