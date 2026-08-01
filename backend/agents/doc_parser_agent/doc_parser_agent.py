"""
backend/agents/doc_parser_agent/doc_parser_agent.py
====================================================
DocParserAgent — the concrete document parser agent.

Execution flow (execute-in-code pattern):
    1. Extract file_path from state
    2. Validate: FileSizeGuardrail → FileTypeGuardrail
    3. Compute file hash → check ExtractionCache (zero LLM calls on hit)
    4. Extract text: pdfplumber (PDF, parallel pages) / Docx2txtLoader (DOCX)
    5. Assess text quality — fail fast with clear message if unusable
    6. Classify document type (LLM call #1, retried up to 3x, ~80 tokens)
    7. Extract structured fields (LLM call #2, retried up to 3x, ≤4000 chars)
    8. Compute completeness score
    9. Cache result by file hash
   10. Return AgentResult

Total LLM calls per NEW document: 2 (detection + extraction), each with retry.
Total LLM calls on cache HIT: 0.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from langchain_core.language_models import BaseChatModel

from backend.agents.doc_parser_agent.cache import (
    ExtractionCache,
    compute_file_hash,
    extraction_cache,
)
from backend.agents.doc_parser_agent.doc_type_detector import DocTypeDetector
from backend.agents.doc_parser_agent.entity_mapper import (
    build_entity_context_summary,
    map_to_entity_store,
)
from backend.agents.doc_parser_agent.extractor import assess_text_quality, extract_text
from backend.agents.doc_parser_agent.field_extractor import FieldExtractor
from backend.agents.doc_parser_agent.guardrails import (
    FileTypeGuardrail,
    FileSizeGuardrail,
)
from backend.agents.doc_parser_agent.schemas import DocType, ExtractionResult
from backend.base.agent import BaseAgent
from backend.state import AgentResult, GraphState

logger = logging.getLogger(__name__)

# Completeness threshold — warn the user if fewer than this fraction of fields
# were successfully extracted. Does NOT block the result; just surfaces a note.
LOW_COMPLETENESS_THRESHOLD = 0.4


class DocParserAgent(BaseAgent):
    """
    Document parsing agent for HR files (PDF and DOCX).

    Inherits from BaseAgent (SOLID: Liskov + Open/Closed).

    Supports 4 document types:
        employment_contract, payslip, employee_id, offer_letter

    Key cross-agent use case:
        Upload employee ID card → extract name + employee_id →
        SQL agent queries salary, leave, payroll, performance.

    Follow-up routing fix:
        After this agent runs, the Router writes extracted fields into
        entity_store and parsed_document in GraphState. On subsequent
        turns, the Planner sees parsed_document is set and skips DocParser.

    Performance notes:
        - Cache: same file (same SHA-256) → 0 LLM calls
        - Text truncated to 4000 chars → ~50-80% token reduction on long docs
        - Parallel page extraction for multi-page PDFs
        - Retry: up to 3 attempts with exponential backoff on LLM failures

    Constructor Parameters
    ----------------------
    llm : BaseChatModel
        LLM for document type detection and field extraction.
        gpt-4o-mini is sufficient — both tasks are well-scoped.
    cache : ExtractionCache | None
        Extraction result cache. Defaults to the process-scoped singleton.
        Pass a fresh ExtractionCache() in tests for isolation.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        cache: ExtractionCache | None = None,
    ) -> None:
        file_guardrails = [FileSizeGuardrail(), FileTypeGuardrail()]
        super().__init__(llm=llm, guardrails=file_guardrails)

        self._detector  = DocTypeDetector(llm)
        self._extractor = FieldExtractor(llm)
        self._cache     = cache or extraction_cache  # default to process singleton

    @property
    def name(self) -> str:
        return "doc_parser"

    async def run(self, state: GraphState) -> AgentResult:
        """
        Execute the full document parsing pipeline.

        Returns
        -------
        AgentResult
            Always returned (never raises). On failure, success=False.

        GraphState fields read:
            uploaded_file_path : str | None

        AgentResult.structured_data schema:
            {
                "document_type":      "employee_id",
                "extracted_fields":   { "employee_name": "John Smith", ... },
                "page_count":         1,
                "source_file":        "employee-id.pdf",
                "char_count":         412,
                "completeness_score": 1.0,   ← fraction of non-None fields
                "cache_hit":          False
            }
        """
        t0 = time.monotonic()

        # ── 1. Get file path from state ────────────────────────────────────
        file_path_str = state.get("uploaded_file_path")
        if not file_path_str:
            return AgentResult.failure(
                agent_name=self.name,
                error=(
                    "No file was uploaded. Please upload a PDF or DOCX file "
                    "to use the document parser."
                ),
                metadata=self._timed_result(t0),
            )

        file_path = Path(file_path_str)
        logger.info("DocParserAgent: processing '%s'", file_path.name)

        # ── 2. Validate file (size + magic bytes) ──────────────────────────
        guard_block = await self._run_with_guardrails(str(file_path))
        if guard_block:
            return guard_block

        # ── 3. Cache check (SHA-256 hash of raw file bytes) ───────────────
        try:
            file_hash = compute_file_hash(file_path)
        except OSError as exc:
            logger.error("DocParserAgent: could not hash file: %s", exc)
            file_hash = ""   # skip cache on hash failure — not fatal

        if file_hash:
            cached = self._cache.get(file_hash)
            if cached is not None:
                logger.info(
                    "DocParserAgent: cache HIT for '%s' — returning cached result",
                    file_path.name,
                )
                cached_copy = dict(cached)
                cached_copy["cache_hit"] = True
                entity_store = map_to_entity_store(
                                DocType(cached_copy["document_type"]),
                                cached_copy["extracted_fields"],
                            )
                return AgentResult(
                    agent_name=self.name,
                    success=True,
                    answer=_build_answer(
                        DocType(cached_copy["document_type"]),
                        cached_copy["extracted_fields"],
                        file_path.name,
                        build_entity_context_summary(entity_store),
                        completeness=cached_copy.get("completeness_score", 1.0),
                        from_cache=True,
                    ),
                    sources=[file_path.name],
                    structured_data=cached_copy,
                    metadata=self._timed_result(t0, {"cache_hit": True, "file_hash": file_hash[:12], "entity_store": entity_store}),
                )

        # ── 4. Extract text (pdfplumber parallel / Docx2txtLoader) ────────
        try:
            raw_text, page_count = await extract_text(file_path)
        except ValueError as exc:
            logger.error("DocParserAgent: text extraction failed: %s", exc)
            return AgentResult.failure(
                agent_name=self.name,
                error=str(exc),
                metadata=self._timed_result(t0),
            )

        # ── 5. Text quality assessment — fail fast, no LLM waste ──────────
        quality = assess_text_quality(raw_text, source_name=file_path.name)
        if not quality.usable:
            return AgentResult.failure(
                agent_name=self.name,
                error=quality.reason,
                metadata=self._timed_result(t0, {"page_count": page_count}),
            )

        # ── 6. Classify document type (LLM call #1, with retry) ───────────
        doc_type: DocType = await self._detector.detect(raw_text)
        logger.info("DocParserAgent: doc_type='%s'", doc_type.value)

        # ── 7. Extract structured fields (LLM call #2, with retry) ────────
        _, fields_dict, completeness = await self._extractor.extract(doc_type, raw_text)

        # ── 8. Build entity context for answer ────────────────────────────
        entity_store = map_to_entity_store(doc_type, fields_dict)
        entity_summary = build_entity_context_summary(entity_store)

        # ── 9. Assemble ExtractionResult ───────────────────────────────────
        extraction_result = ExtractionResult(
            document_type=doc_type,
            extracted_fields=fields_dict,
            page_count=page_count,
            source_file=file_path.name,
            char_count=len(raw_text),
            completeness_score=completeness,
            cache_hit=False,
        )
        structured_data = extraction_result.model_dump()

        # ── 10. Store in cache for future identical uploads ────────────────
        if file_hash:
            self._cache.set(file_hash, structured_data)

        # ── 11. Build metadata ─────────────────────────────────────────────
        metadata = self._timed_result(
            t0,
            {
                "document_type":      doc_type.value,
                "page_count":         page_count,
                "char_count":         len(raw_text),
                "fields_extracted":   sum(1 for v in fields_dict.values() if v is not None),
                "completeness_score": completeness,
                "cache_hit":          False,
                "file_hash":          file_hash[:12] if file_hash else "n/a",
                "entity_store":       entity_store,
            },
        )

        logger.info(
            "DocParserAgent: done in %dms — type=%s, completeness=%.0f%%, cache=MISS",
            metadata["duration_ms"],
            doc_type.value,
            completeness * 100,
        )

        return AgentResult(
            agent_name=self.name,
            success=True,
            answer=_build_answer(doc_type, fields_dict, file_path.name, entity_summary, completeness),
            sources=[file_path.name],
            structured_data=structured_data,
            metadata=metadata,
        )


# ── Answer formatter ───────────────────────────────────────────────────────────

def _build_answer(
    doc_type: DocType,
    fields: dict,
    filename: str,
    entity_summary: str,
    completeness: float = 1.0,
    from_cache: bool = False,
) -> str:
    """
    Build a concise natural-language summary of the extraction result.

    Appends a low-completeness warning when fewer than 40% of fields were found.
    When DocParser feeds into another agent, the Combiner will synthesize
    the final answer from all AgentResults; this answer is used in standalone mode.
    """
    non_null = {k: v for k, v in fields.items() if v is not None}

    if not non_null:
        return (
            f"I processed '{filename}' but could not extract any structured fields. "
            "The document may not match the expected format for "
            f"a {doc_type.value.replace('_', ' ')}. Please check the file and try again."
        )

    type_labels = {
        DocType.EMPLOYMENT_CONTRACT: "employment contract",
        DocType.PAYSLIP:             "payslip",
        DocType.EMPLOYEE_ID:         "employee ID card",
        DocType.OFFER_LETTER:        "offer letter",
        DocType.UNKNOWN:             "HR document",
    }
    label = type_labels.get(doc_type, "document")
    cache_note = " _(cached)_" if from_cache else ""

    lines = [
        f"I successfully extracted information from the **{label}** '{filename}'{cache_note}:\n"
    ]
    for key, value in non_null.items():
        display_key = key.replace("_", " ").title()
        lines.append(f"- **{display_key}**: {value}")

    if entity_summary:
        lines.append(f"\n_{entity_summary}_")

    # Low completeness warning — surfaces to user without blocking
    if completeness < LOW_COMPLETENESS_THRESHOLD:
        lines.append(
            f"\n> ⚠️ Only partial information was extracted ({completeness:.0%} of expected fields). "
            "The document may not be in the standard format. "
            "Cross-check the values before acting on them."
        )

    return "\n".join(lines)
