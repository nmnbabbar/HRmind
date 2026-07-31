"""
backend/agents/doc_parser_agent/field_extractor.py
===================================================
LLM-based structured field extraction per document type, with retry logic.

Changes from v1:
- Exponential backoff retry on both LLM calls (max 3 attempts).
  Handles transient failures: rate limits, timeouts, connection errors.
- Text is truncated to MAX_EXTRACTION_CHARS before sending to the LLM.
  Key fields appear in the first 1-2 pages. Sending a full 20-page contract
  wastes ~15,000 tokens for the same extraction quality.
- Returns completeness_score alongside the fields dict.

LLM calls per document: still 2 (detection is in DocTypeDetector).
Each call has 3 retry attempts with exponential backoff (1s, 2s, 4s).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from backend.agents.doc_parser_agent.schemas import (
    DOC_TYPE_SCHEMA_MAP,
    DocType,
    ExtractionResult,
)

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────────

# Key fields (name, salary, ID, dates) appear in the first ~4000 chars.
# A typical HR document page is ~1000–2000 chars.
# This truncation covers 2–4 pages, which is sufficient for all supported types.
MAX_EXTRACTION_CHARS = 4_000

# Retry config — exponential backoff
MAX_RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY_S = 1.0   # 1s → 2s → 4s


# ── Per-type extraction prompts ────────────────────────────────────────────────

_TYPE_HINTS = {
    DocType.EMPLOYMENT_CONTRACT: (
        "employment contract. Extract: employee name, job role/title, department, "
        "employment start date (YYYY-MM-DD), annual salary, notice period in days "
        "(convert weeks/months to days), and contract type (permanent, fixed-term, or contractor)."
    ),
    DocType.PAYSLIP: (
        "payslip / pay statement. Extract: employee name, employee ID, pay period "
        "(YYYY-MM), gross pay, net pay, and total deductions."
    ),
    DocType.EMPLOYEE_ID: (
        "employee ID card or badge. Extract: employee full name, employee ID number, "
        "department, and job role/title."
    ),
    DocType.OFFER_LETTER: (
        "job offer letter. Extract: candidate name, offered job role/title, "
        "offered annual salary, proposed start date (YYYY-MM-DD), and department."
    ),
    DocType.UNKNOWN: (
        "HR document. Extract any relevant information: names, dates, monetary amounts, "
        "job titles, departments, or employee IDs."
    ),
}

FIELD_EXTRACTOR_SYSTEM_PROMPT = """\
You are a precise data extraction assistant for an HR system.

Your task: read the following {doc_type_hint} and extract the specified fields.

Rules:
1. Extract ONLY information explicitly stated in the document.
2. If a field cannot be found, return null — never guess or invent values.
3. For dates, use ISO 8601 format: YYYY-MM-DD.
4. For notice periods, convert to calendar days (e.g. "1 month" = 30 days, "4 weeks" = 28 days).
5. For salary/pay amounts, return the numeric value only — no currency symbols or commas.
6. Return structured data only — no explanation, no preamble.
"""


# ── Retry helper ───────────────────────────────────────────────────────────────

async def _invoke_with_retry(
    structured_llm: Any,
    messages: list,
    context: str = "",
    max_attempts: int = MAX_RETRY_ATTEMPTS,
    base_delay: float = RETRY_BASE_DELAY_S,
) -> Any:
    """
    Invoke a structured LLM with exponential backoff retry.

    Parameters
    ----------
    structured_llm : Any
        LLM instance from .with_structured_output(). Has an .ainvoke() method.
    messages : list
        LangChain message list to send.
    context : str
        Human-readable label for logging (e.g. "EMPLOYEE_ID extraction").
    max_attempts : int
        Maximum number of attempts before re-raising the last exception.
    base_delay : float
        Base delay in seconds. Doubled on each retry (1s → 2s → 4s).

    Returns
    -------
    Any
        The LLM response on success.

    Raises
    ------
    Exception
        Re-raises the last exception if all attempts fail.
    """
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return await structured_llm.ainvoke(messages)

        except Exception as exc:
            last_exc = exc
            if attempt == max_attempts:
                logger.error(
                    "FieldExtractor: all %d attempts failed for %s. Last error: %s",
                    max_attempts, context, exc,
                )
                break

            delay = base_delay * (2 ** (attempt - 1))  # 1s, 2s, 4s
            logger.warning(
                "FieldExtractor: attempt %d/%d failed for %s (%s). Retrying in %.1fs...",
                attempt, max_attempts, context, exc, delay,
            )
            await asyncio.sleep(delay)

    raise last_exc  # type: ignore[misc]


# ── Field extractor ────────────────────────────────────────────────────────────

class FieldExtractor:
    """
    Extracts structured fields from document text using a schema-aware LLM call.

    One instance handles all document types. The correct schema is selected
    at call time based on the DocType from DocTypeDetector.

    Improvements over v1:
    - Text truncated to MAX_EXTRACTION_CHARS (~4000 chars / ~1000 tokens).
    - LLM call retried up to 3 times with exponential backoff.
    - Returns completeness_score (fraction of non-None fields).
    """

    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm

    async def extract(
        self,
        doc_type: DocType,
        text: str,
    ) -> tuple[BaseModel, dict[str, Any], float]:
        """
        Extract structured fields from document text.

        Parameters
        ----------
        doc_type : DocType
            The classified document type (from DocTypeDetector).
        text : str
            Full extracted document text. Internally truncated to MAX_EXTRACTION_CHARS.

        Returns
        -------
        tuple[BaseModel, dict[str, Any], float]
            (pydantic_model_instance, fields_dict, completeness_score)
            - pydantic_model_instance : Validated schema instance
            - fields_dict             : .model_dump() — JSON-safe for GraphState
            - completeness_score      : Fraction of non-None fields (0.0–1.0)
        """
        schema_class = DOC_TYPE_SCHEMA_MAP.get(doc_type)
        doc_type_hint = _TYPE_HINTS.get(doc_type, _TYPE_HINTS[DocType.UNKNOWN])

        if schema_class is None:
            logger.warning(
                "FieldExtractor: no schema for DocType '%s' — using generic extraction",
                doc_type.value,
            )
            return await self._extract_generic(text)

        # Truncate text to avoid sending full 20-page contracts to the LLM
        truncated_text = text[:MAX_EXTRACTION_CHARS]
        if len(text) > MAX_EXTRACTION_CHARS:
            logger.debug(
                "FieldExtractor: text truncated %d → %d chars for '%s'",
                len(text), MAX_EXTRACTION_CHARS, doc_type.value,
            )

        structured_llm = self._llm.with_structured_output(schema_class)
        messages = [
            SystemMessage(
                content=FIELD_EXTRACTOR_SYSTEM_PROMPT.format(doc_type_hint=doc_type_hint)
            ),
            HumanMessage(content=f"Document text:\n\n{truncated_text}"),
        ]

        try:
            result: BaseModel = await _invoke_with_retry(  # type: ignore[assignment]
                structured_llm=structured_llm,
                messages=messages,
                context=f"{doc_type.value} extraction",
            )
            fields_dict = result.model_dump()
            completeness = ExtractionResult.compute_completeness(fields_dict)

            non_null = sum(1 for v in fields_dict.values() if v is not None)
            logger.info(
                "FieldExtractor: extracted %d/%d fields from '%s' (completeness=%.0f%%)",
                non_null, len(fields_dict), doc_type.value, completeness * 100,
            )
            return result, fields_dict, completeness

        except Exception as exc:
            logger.error(
                "FieldExtractor: extraction failed for '%s' after retries: %s",
                doc_type.value, exc, exc_info=True,
            )
            # Return empty schema — never crash, never raise from an agent
            empty = schema_class()
            return empty, empty.model_dump(), 0.0

    async def _extract_generic(
        self,
        text: str,
    ) -> tuple[BaseModel, dict[str, Any], float]:
        """
        Fallback for UNKNOWN document types.
        Returns raw text snippet — no structured extraction attempted.
        """
        from pydantic import BaseModel as PM

        class GenericFields(PM):
            raw_info: str = ""

        generic = GenericFields(raw_info=text[:MAX_EXTRACTION_CHARS])
        return generic, {"raw_info": generic.raw_info}, 0.0
