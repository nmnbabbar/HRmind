"""
backend/memory/entity_store.py
================================
Named entity extraction and lookup for follow-up question resolution.

Purpose
-------
Enables coreference resolution without passing full conversation history.
When a user asks "What is her notice period?" after discussing Alice Johnson,
the entity store resolves "her" → "Alice Johnson" via entity lookup.

Implementation
--------------
Uses spaCy's en_core_web_sm model for fast, CPU-friendly NER.
The model is loaded lazily and cached as a module-level singleton.
The extraction call is async-safe via asyncio.run_in_executor (spaCy is blocking).

Graceful degradation: if spaCy model is not installed, extraction silently
returns an empty dict — the system continues to function without entity resolution.

Entity types tracked (aligned with HR domain):
    PERSON     → employee/manager names
    ORG        → departments, companies
    DATE       → hire dates, review periods, leave dates
    MONEY      → salary figures
    GPE        → office locations
    WORK_OF_ART → policy/document names (spaCy sometimes classifies these here)
"""

import asyncio
from typing import Optional

from backend.utils.log import get_logger

logger = get_logger(__name__)

# ── spaCy singleton ────────────────────────────────────────────────────────────
# False = load attempted but model not found; None = not yet attempted
_nlp: Optional[object] = None
_nlp_loaded: bool = False


def _load_nlp() -> Optional[object]:
    """Lazily load spaCy model. Returns None if unavailable."""
    global _nlp, _nlp_loaded
    if _nlp_loaded:
        return _nlp

    _nlp_loaded = True
    try:
        import spacy  # noqa: PLC0415

        _nlp = spacy.load("en_core_web_sm")
        logger.info("spacy_model_loaded", model="en_core_web_sm")
    except OSError:
        logger.warning(
            "spacy_model_not_found",
            message="en_core_web_sm not installed; entity extraction disabled",
            hint="Run: python -m spacy download en_core_web_sm",
        )
        _nlp = None
    except ImportError:
        logger.warning("spacy_not_installed", message="spacy package missing")
        _nlp = None

    return _nlp


# ── Entity types to track ─────────────────────────────────────────────────────
_TRACKED_ENTITY_TYPES: frozenset[str] = frozenset({
    "PERSON",      # employee / manager names
    "ORG",         # departments, companies
    "DATE",        # hire dates, leave dates, review periods
    "MONEY",       # salary figures, expense amounts
    "GPE",         # office locations
    "WORK_OF_ART", # policy document names
})


class EntityStore:
    """
    Stateful store for named entities extracted from conversation turns.

    Entities are accumulated across turns (not reset per-turn), so "Alice"
    mentioned in turn 1 is still resolvable in turn 5.

    The store is keyed by entity TYPE (not entity text), so only the MOST
    RECENT entity of each type is remembered. This keeps the store compact
    and focused on the current conversational context.

    Usage
    -----
        store = EntityStore()
        await store.extract_and_update("Alice Johnson joined the Engineering team.")
        store.get()
        # → {"person": "Alice Johnson", "org": "Engineering"}
    """

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    async def extract_and_update(self, text: str) -> dict[str, str]:
        """
        Extract entities from text and merge into the store.

        Blocks in a thread pool (spaCy is CPU-bound / not async-native).
        Returns the updated store after merging.
        """
        nlp = _load_nlp()
        if nlp is None:
            return self.get()

        loop = asyncio.get_event_loop()
        new_entities: dict[str, str] = await loop.run_in_executor(
            None, self._extract_sync, text, nlp
        )

        # Merge: new entities overwrite old ones (most-recent-wins per type)
        self._store.update(new_entities)
        return self.get()

    def get(self) -> dict[str, str]:
        """Return a copy of the current entity store."""
        return dict(self._store)

    def clear(self) -> None:
        """Reset the store — called when starting a new conversation."""
        self._store.clear()

    def resolve(self, pronoun_or_reference: str) -> str | None:
        """
        Attempt to resolve a reference (pronoun / partial name) to a known entity.

        Simple heuristic: if input is a pronoun (he/she/they/it),
        return the PERSON entity if known.

        Returns None if no resolution is found.
        """
        lower = pronoun_or_reference.lower().strip()
        pronouns = {"he", "she", "they", "his", "her", "their", "him", "them"}
        if lower in pronouns:
            return self._store.get("person")
        return None

    @staticmethod
    def _extract_sync(text: str, nlp: object) -> dict[str, str]:
        """Synchronous spaCy NER extraction — called in thread pool."""
        doc = nlp(text)  # type: ignore[call-arg]
        return {
            ent.label_.lower(): ent.text
            for ent in doc.ents  # type: ignore[attr-defined]
            if ent.label_ in _TRACKED_ENTITY_TYPES
        }
