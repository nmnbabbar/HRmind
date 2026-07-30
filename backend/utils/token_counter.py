"""
backend/utils/token_counter.py
================================
Tiktoken-based token counting utilities.

Used by ContextBuilder to enforce hard token budgets per agent call.
Uses cl100k_base encoding — the tokeniser for gpt-4o and gpt-4o-mini.

Design notes
------------
- Encoder is loaded once and cached as a module-level singleton (expensive to init)
- All functions have a graceful fallback if tiktoken is unavailable or fails:
  approximate character-based counting (1 token ≈ 4 chars) never crashes
- trim_to_token_budget truncates at the character level (no partial tokens)
"""

from __future__ import annotations

from functools import lru_cache

from backend.utils.log import get_logger

logger = get_logger(__name__)

# cl100k_base: tokeniser for gpt-4o, gpt-4o-mini, text-embedding-3-*
_ENCODING_NAME = "cl100k_base"
# Approximate characters per token — used as fallback when tiktoken fails
_CHARS_PER_TOKEN_APPROX = 4


@lru_cache(maxsize=1)
def _get_encoder():
    """Load and cache the tiktoken encoder. Returns None on failure."""
    try:
        import tiktoken  # noqa: PLC0415

        enc = tiktoken.get_encoding(_ENCODING_NAME)
        logger.debug("tiktoken_encoder_loaded", encoding=_ENCODING_NAME)
        return enc
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "tiktoken_load_failed",
            error=str(exc),
            fallback="character approximation (1 token ≈ 4 chars)",
        )
        return None


def count_tokens(text: str) -> int:
    """
    Count the number of tokens in a string.

    Parameters
    ----------
    text : str
        Input text to count.

    Returns
    -------
    int
        Token count. Falls back to len(text) // 4 if tiktoken unavailable.
    """
    if not text:
        return 0

    encoder = _get_encoder()
    if encoder is not None:
        return len(encoder.encode(text))

    # Fallback: character-based approximation
    return len(text) // _CHARS_PER_TOKEN_APPROX


def count_messages_tokens(messages: list[dict]) -> int:
    """
    Count tokens for a list of LangChain/OpenAI-style chat messages.

    Adds per-message overhead (role token + separators ≈ 4 tokens each)
    and a priming token (2 tokens) to approximate the actual API token count.

    Parameters
    ----------
    messages : list[dict]
        List of dicts with at minimum a "content" key.
    """
    total = 2  # conversation priming tokens
    for msg in messages:
        total += count_tokens(msg.get("content", ""))
        total += 4  # role + separator overhead per message
    return total


def fits_in_budget(text: str, max_tokens: int) -> bool:
    """Return True if text fits within the given token budget."""
    return count_tokens(text) <= max_tokens


def trim_to_token_budget(text: str, max_tokens: int) -> str:
    """
    Trim text to fit within a token budget.

    Trimming is approximate: uses character-based slicing to avoid the O(n)
    cost of repeatedly re-encoding. Appends "... [trimmed]" when truncated.

    Parameters
    ----------
    text : str
        Input text to trim.
    max_tokens : int
        Maximum allowed tokens.

    Returns
    -------
    str
        Trimmed text (original if it already fits).
    """
    if fits_in_budget(text, max_tokens):
        return text

    # Approximate character limit: tokens × chars/token with 10% safety margin
    char_limit = int(max_tokens * _CHARS_PER_TOKEN_APPROX * 0.9)
    return text[:char_limit] + "... [trimmed]"
