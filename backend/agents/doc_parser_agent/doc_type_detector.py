"""
backend/agents/doc_parser_agent/doc_type_detector.py
=====================================================
LLM-based document type classification.

Strategy:
- Feed the first 500 characters of the extracted text to the LLM.
- Force a structured response using .with_structured_output(DocTypeResponse).
- Returns a DocType enum value.

Why only 500 chars?
- Document headers/titles appear in the first paragraph.
- "EMPLOYMENT CONTRACT", "PAYSLIP", "OFFER LETTER" are usually in the first ~100 chars.
- Sending more text wastes tokens on a classification call that only needs a label.
- Reduces cost: this is ~100 input tokens total.

Fallback:
- If the LLM returns DocType.UNKNOWN, FieldExtractor still attempts extraction
  using a generic prompt. AgentResult will reflect the uncertainty.
"""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from backend.agents.doc_parser_agent.schemas import DocType

logger = logging.getLogger(__name__)

# ── Number of chars sent to the detector ──────────────────────────────────────
DETECTOR_CHAR_LIMIT = 500

# ── System prompt ──────────────────────────────────────────────────────────────
DETECTOR_SYSTEM_PROMPT = """\
You are a document classifier for an HR system.

Your task: read the beginning of a document and classify it into exactly one of these types:
- employment_contract : A signed employment contract or contract of employment
- payslip             : A salary slip, payslip, or pay statement
- employee_id         : An employee ID card, badge, or staff ID document
- offer_letter        : A job offer letter or offer of employment
- unknown             : Does not match any of the above

Return ONLY the document_type value. No explanation.
"""


# ── Structured output schema ───────────────────────────────────────────────────

class DocTypeResponse(BaseModel):
    """Structured output for the doc type detector LLM call."""
    document_type: DocType = Field(
        description="The classified document type."
    )


# ── Detector ───────────────────────────────────────────────────────────────────

class DocTypeDetector:
    """
    Classifies a document from its first 500 characters.

    Uses .with_structured_output() so the LLM is forced to return
    a valid DocType value — no string parsing needed.

    Constructor Parameters
    ----------------------
    llm : BaseChatModel
        Language model for classification. A fast, cheap model works well
        (e.g. gpt-4o-mini) — this is a very simple classification task.
    """

    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm.with_structured_output(DocTypeResponse)

    async def detect(self, text: str) -> DocType:
        """
        Classify a document from its text content.

        Parameters
        ----------
        text : str
            Full or partial document text. Only the first DETECTOR_CHAR_LIMIT
            characters are sent to the LLM.

        Returns
        -------
        DocType
            Detected document type. Falls back to DocType.UNKNOWN on any error.
        """
        from backend.agents.doc_parser_agent.field_extractor import _invoke_with_retry

        preview = text[:DETECTOR_CHAR_LIMIT].strip()

        if not preview:
            logger.warning("DocTypeDetector: empty text — returning UNKNOWN")
            return DocType.UNKNOWN

        messages = [
            SystemMessage(content=DETECTOR_SYSTEM_PROMPT),
            HumanMessage(content=f"Document beginning:\n\n{preview}"),
        ]

        try:
            response: DocTypeResponse = await _invoke_with_retry(  # type: ignore[assignment]
                structured_llm=self._llm,
                messages=messages,
                context="doc type detection",
            )
            doc_type = response.document_type
            logger.info("DocTypeDetector: classified as '%s'", doc_type.value)
            return doc_type

        except Exception as exc:
            logger.warning(
                "DocTypeDetector: classification failed after retries (%s) — returning UNKNOWN", exc
            )
            return DocType.UNKNOWN
