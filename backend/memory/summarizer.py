"""
backend/memory/summarizer.py
=============================
Rolling conversation summarizer — compresses history every N turns.

Purpose
-------
Keeps the LLM context window small across long conversations.
Instead of passing the full conversation history to every agent call,
we maintain a running summary that is updated periodically.

Strategy
--------
- Full history is preserved by LangGraph's MemorySaver (never discarded)
- After every SUMMARIZE_EVERY_N user turns, a summarization LLM call is made
- The resulting summary REPLACES all prior summaries (rolling, not appended)
- Agents receive: summary (max 600 tokens) + last 3 turns verbatim
- Cost: 1 extra LLM call per SUMMARIZE_EVERY_N turns (amortized overhead)

Context window achieved per agent call (approximate):
    System prompt:    ~1 000 tokens
    User query:       ~  200 tokens
    Rolling summary:  ~  600 tokens
    Last 3 turns:     ~  800 tokens
    Agent context:    ~1 000 tokens
    Total:            ~3 600 tokens  (well within GPT-4o-mini's 128k context)
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from backend.utils.log import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a conversation memory compressor for an enterprise HR AI assistant.
Summarize the conversation below into a single concise paragraph (max 150 words).

Focus strictly on:
- The core intent of the user (e.g., asked about maternity policies, requested salary details)
- The key facts or answers provided by the assistant
- Critical HR entities mentioned (names, policy names, specific dates, departments, document names)

Rules:
- Write in the third person.
- Be highly factual and information-dense.
- Strip out pleasantries, opinions, and filler words.
- Ensure context for follow-up questions is preserved.\
"""


class RollingSummarizer:
    """
    Tracks turn count and triggers LLM summarization on schedule.

    Thread-safety: this class is NOT thread-safe. One instance per session.
    In FastAPI, each session gets its own summarizer stored in app.state
    keyed by session_id (managed by the orchestration layer in Phase 5).

    Usage
    -----
        from backend.config import get_settings
        settings = get_settings()
        summarizer = RollingSummarizer(
            llm=ChatOpenAI(
                model=settings.agent_model,
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key
            )
        )
        summarizer.increment()   # call after each user turn

        if summarizer.should_summarize():
            new_summary = await summarizer.summarize(
                conversation_text=format_history(state["recent_turns"]),
                existing_summary=state["conversation_summary"],
            )
            # Store new_summary in state["conversation_summary"]
    """

    SUMMARIZE_EVERY_N: int = 10  # trigger every 10 user turns

    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm
        self._turn_count: int = 0

    def increment(self) -> None:
        """Increment the turn counter. Call once per user query."""
        self._turn_count += 1

    def should_summarize(self) -> bool:
        """
        Return True if the summarizer should trigger on this turn.

        Triggers on turns 10, 20, 30, ... (multiples of SUMMARIZE_EVERY_N).
        Returns False on turn 0 (no history to summarize yet).
        """
        return (
            self._turn_count > 0
            and self._turn_count % self.SUMMARIZE_EVERY_N == 0
        )

    async def summarize(
        self,
        conversation_text: str,
        existing_summary: str = "",
    ) -> str:
        """
        Generate a compressed summary of the conversation so far.

        If an existing summary is provided, it is prepended so the model
        can build on prior context rather than summarizing from scratch.

        Parameters
        ----------
        conversation_text : str
            Formatted conversation to summarize (e.g. "User: ...\nAssistant: ...")
        existing_summary : str
            Previous rolling summary, if any. Empty string on first call.

        Returns
        -------
        str
            New compressed summary (replaces the previous summary in GraphState).
        """
        content_to_summarize = conversation_text
        if existing_summary:
            content_to_summarize = (
                f"Previous summary:\n{existing_summary}\n\n"
                f"New conversation:\n{conversation_text}"
            )

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=content_to_summarize),
        ]

        logger.info(
            "summarizing_conversation",
            turn_count=self._turn_count,
            has_existing_summary=bool(existing_summary),
        )

        response = await self._llm.ainvoke(messages)
        return response.content

    @property
    def turn_count(self) -> int:
        """Current turn count — useful for debugging and metrics."""
        return self._turn_count
