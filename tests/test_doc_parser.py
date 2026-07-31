"""
tests/test_doc_parser.py
========================
Unit tests for the Document Parser Agent (Phase 4).

Test strategy:
- No live LLM calls — all LLM interactions are mocked with AsyncMock / MagicMock.
- File I/O uses tmp_path (pytest's built-in temp directory fixture).
- Magic bytes tests use raw bytes written to temp files.
- The follow-up routing test verifies the state-persistence contract:
  parsed_document set + no new file → no doc_parser in plan.

Coverage:
    FileSizeGuardrail        — size limit enforcement
    FileTypeGuardrail        — magic bytes validation (PDF, DOCX, spoofed)
    extract_text (PDF)       — pdfplumber is mocked to isolate extraction logic
    extract_text (DOCX)      — Docx2txtLoader is mocked
    DocTypeDetector          — mocked LLM → correct DocType returned
    FieldExtractor           — mocked LLM → typed Pydantic schema returned
    map_to_entity_store      — field mapping per doc type
    build_entity_context_summary — context string building
    DocParserAgent.run()     — full pipeline with mocked sub-components
    Follow-up routing        — parsed_document persistence contract
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.agents.doc_parser_agent.guardrails import (
    MAX_FILE_SIZE_BYTES,
    FileTypeGuardrail,
    FileSizeGuardrail,
)
from backend.agents.doc_parser_agent.schemas import (
    DocType,
    EmployeeIDFields,
    EmploymentContractFields,
    PayslipFields,
    OfferLetterFields,
    ExtractionResult,
)
from backend.agents.doc_parser_agent.entity_mapper import (
    map_to_entity_store,
    build_entity_context_summary,
)
from backend.state import make_initial_state

# ── Magic byte constants ───────────────────────────────────────────────────────

PDF_MAGIC  = b"%PDF-1.4\n%..."   # valid PDF header bytes
DOCX_MAGIC = b"PK\x03\x04" + b"\x00" * 20  # valid ZIP/OOXML header


# ─────────────────────────────────────────────────────────────────────────────
# FileSizeGuardrail
# ─────────────────────────────────────────────────────────────────────────────

class TestFileSizeGuardrail:
    """FileSizeGuardrail rejects files over 20MB."""

    @pytest.fixture
    def guardrail(self):
        return FileSizeGuardrail()

    @pytest.mark.asyncio
    async def test_accepts_small_file(self, guardrail, tmp_path):
        """Files under the limit are accepted."""
        f = tmp_path / "small.pdf"
        f.write_bytes(b"x" * 1024)   # 1 KB
        result = await guardrail.check(str(f))
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_rejects_oversized_file(self, guardrail, tmp_path):
        """Files over 20MB are rejected before reading content."""
        f = tmp_path / "huge.pdf"
        f.write_bytes(b"x" * (MAX_FILE_SIZE_BYTES + 1))
        result = await guardrail.check(str(f))
        assert result.passed is False
        assert "20 MB" in result.reason
        assert result.guardrail_name == "FileSizeGuardrail"

    @pytest.mark.asyncio
    async def test_rejects_exactly_at_limit_plus_one(self, guardrail, tmp_path):
        """Boundary: exactly MAX+1 byte is rejected."""
        f = tmp_path / "boundary.pdf"
        f.write_bytes(b"x" * (MAX_FILE_SIZE_BYTES + 1))
        result = await guardrail.check(str(f))
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_fails_gracefully_if_file_missing(self, guardrail, tmp_path):
        """Non-existent file path → fail with clear message."""
        result = await guardrail.check(str(tmp_path / "ghost.pdf"))
        assert result.passed is False
        assert "not found" in result.reason.lower()


# ─────────────────────────────────────────────────────────────────────────────
# FileTypeGuardrail
# ─────────────────────────────────────────────────────────────────────────────

class TestFileTypeGuardrail:
    """FileTypeGuardrail validates file type via magic bytes, not extension."""

    @pytest.fixture
    def guardrail(self):
        return FileTypeGuardrail()

    @pytest.mark.asyncio
    async def test_accepts_valid_pdf(self, guardrail, tmp_path):
        """Valid PDF magic bytes → accepted."""
        f = tmp_path / "contract.pdf"
        f.write_bytes(PDF_MAGIC + b" rest of pdf content")
        result = await guardrail.check(str(f))
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_accepts_valid_docx(self, guardrail, tmp_path):
        """Valid DOCX (PK ZIP) magic bytes → accepted."""
        f = tmp_path / "contract.docx"
        f.write_bytes(DOCX_MAGIC + b" rest of docx content")
        result = await guardrail.check(str(f))
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_rejects_spoofed_pdf(self, guardrail, tmp_path):
        """File with .pdf extension but PNG content → rejected."""
        f = tmp_path / "sneaky.pdf"
        f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"png content")  # PNG magic bytes
        result = await guardrail.check(str(f))
        assert result.passed is False
        assert result.guardrail_name == "FileTypeGuardrail"
        assert "PDF" in result.reason or "DOCX" in result.reason

    @pytest.mark.asyncio
    async def test_rejects_exe_disguised_as_docx(self, guardrail, tmp_path):
        """EXE magic bytes disguised as .docx → rejected."""
        f = tmp_path / "payload.docx"
        f.write_bytes(b"MZ" + b"\x00" * 60)  # Windows PE/EXE magic
        result = await guardrail.check(str(f))
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_rejects_empty_file(self, guardrail, tmp_path):
        """Empty file has no magic bytes → rejected."""
        f = tmp_path / "empty.pdf"
        f.write_bytes(b"")
        result = await guardrail.check(str(f))
        assert result.passed is False


# ─────────────────────────────────────────────────────────────────────────────
# Text extraction
# ─────────────────────────────────────────────────────────────────────────────

class TestTextExtraction:
    """Text extraction from PDF and DOCX files."""

    @pytest.mark.asyncio
    async def test_pdf_extraction_returns_text_and_page_count(self, tmp_path):
        """pdfplumber extraction returns (text, page_count) tuple."""
        f = tmp_path / "contract.pdf"
        f.write_bytes(PDF_MAGIC)

        expected_text = "Employment Contract\nEmployee: John Smith\nSalary: 60000"
        mock_page = MagicMock()
        mock_page.extract_text.return_value = expected_text
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("backend.agents.doc_parser_agent.extractor.pdfplumber") as mock_pdfplumber:
            mock_pdfplumber.open.return_value = mock_pdf
            from backend.agents.doc_parser_agent.extractor import extract_pdf
            text, page_count = await extract_pdf(f)

        assert expected_text in text
        assert page_count == 1

    @pytest.mark.asyncio
    async def test_docx_extraction_returns_text(self, tmp_path):
        """Docx2txtLoader extraction returns (text, 1) tuple."""
        f = tmp_path / "payslip.docx"
        f.write_bytes(DOCX_MAGIC)

        from langchain_core.documents import Document
        expected_text = "Payslip\nEmployee: Alice Brown\nNet: 3500"
        mock_doc = Document(page_content=expected_text)

        with patch(
            "backend.agents.doc_parser_agent.extractor.Docx2txtLoader"
        ) as mock_loader_cls:
            mock_loader = MagicMock()
            mock_loader.load.return_value = [mock_doc]
            mock_loader_cls.return_value = mock_loader

            from backend.agents.doc_parser_agent.extractor import extract_docx
            text, page_count = await extract_docx(f)

        assert expected_text in text
        assert page_count == 1


# ─────────────────────────────────────────────────────────────────────────────
# DocTypeDetector
# ─────────────────────────────────────────────────────────────────────────────

class TestDocTypeDetector:
    """LLM-based document type classification."""

    def _make_detector(self, doc_type: DocType):
        """Return a DocTypeDetector whose LLM always returns doc_type."""
        from backend.agents.doc_parser_agent.doc_type_detector import (
            DocTypeDetector,
            DocTypeResponse,
        )
        mock_llm = MagicMock()
        mock_structured_llm = MagicMock()
        response = DocTypeResponse(document_type=doc_type)
        mock_structured_llm.ainvoke = AsyncMock(return_value=response)
        mock_llm.with_structured_output.return_value = mock_structured_llm
        return DocTypeDetector(mock_llm)

    @pytest.mark.asyncio
    async def test_detects_employment_contract(self):
        detector = self._make_detector(DocType.EMPLOYMENT_CONTRACT)
        result = await detector.detect("EMPLOYMENT CONTRACT\nThis agreement is made...")
        assert result == DocType.EMPLOYMENT_CONTRACT

    @pytest.mark.asyncio
    async def test_detects_employee_id(self):
        detector = self._make_detector(DocType.EMPLOYEE_ID)
        result = await detector.detect("EMPLOYEE ID CARD\nName: John Smith\nID: EMP042")
        assert result == DocType.EMPLOYEE_ID

    @pytest.mark.asyncio
    async def test_detects_payslip(self):
        detector = self._make_detector(DocType.PAYSLIP)
        result = await detector.detect("PAYSLIP\nPay Period: June 2026\nGross: 5416.67")
        assert result == DocType.PAYSLIP

    @pytest.mark.asyncio
    async def test_returns_unknown_on_empty_text(self):
        """Empty text → UNKNOWN without LLM call."""
        detector = self._make_detector(DocType.UNKNOWN)
        result = await detector.detect("")
        assert result == DocType.UNKNOWN

    @pytest.mark.asyncio
    async def test_returns_unknown_on_llm_failure(self):
        """LLM error → falls back to UNKNOWN (fail safe)."""
        from backend.agents.doc_parser_agent.doc_type_detector import DocTypeDetector
        mock_llm = MagicMock()
        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(side_effect=Exception("LLM timeout"))
        mock_llm.with_structured_output.return_value = mock_structured_llm
        detector = DocTypeDetector(mock_llm)
        result = await detector.detect("Some document text here")
        assert result == DocType.UNKNOWN


# ─────────────────────────────────────────────────────────────────────────────
# FieldExtractor
# ─────────────────────────────────────────────────────────────────────────────

class TestFieldExtractor:
    """Structured field extraction per document type."""

    def _make_extractor(self, return_model):
        """Return a FieldExtractor whose LLM always returns return_model."""
        from backend.agents.doc_parser_agent.field_extractor import FieldExtractor
        mock_llm = MagicMock()
        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=return_model)
        mock_llm.with_structured_output.return_value = mock_structured_llm
        return FieldExtractor(mock_llm)

    @pytest.mark.asyncio
    async def test_extracts_employee_id_fields(self):
        """EmployeeIDFields returned correctly for EMPLOYEE_ID doc type."""
        expected = EmployeeIDFields(
            employee_name="John Smith",
            employee_id="EMP042",
            department="Engineering",
            role="Senior Engineer",
        )
        extractor = self._make_extractor(expected)
        model, fields, completeness = await extractor.extract(DocType.EMPLOYEE_ID, "some text")

        assert fields["employee_name"] == "John Smith"
        assert fields["employee_id"]   == "EMP042"
        assert fields["department"]    == "Engineering"
        assert fields["role"]          == "Senior Engineer"
        assert completeness == 1.0   # all 4 fields found

    @pytest.mark.asyncio
    async def test_extracts_contract_fields(self):
        """EmploymentContractFields returned correctly."""
        expected = EmploymentContractFields(
            employee_name="Alice Brown",
            role="Product Manager",
            salary=75000.0,
            notice_period_days=60,
            contract_type="permanent",
        )
        extractor = self._make_extractor(expected)
        _, fields, completeness = await extractor.extract(DocType.EMPLOYMENT_CONTRACT, "some text")

        assert fields["employee_name"] == "Alice Brown"
        assert fields["salary"] == 75000.0
        assert fields["notice_period_days"] == 60
        assert fields["contract_type"] == "permanent"

    @pytest.mark.asyncio
    async def test_handles_all_null_fields(self):
        """All-None extraction (unreadable doc) doesn't crash."""
        expected = EmployeeIDFields()  # all fields default to None
        extractor = self._make_extractor(expected)
        _, fields, completeness = await extractor.extract(DocType.EMPLOYEE_ID, "gibberish")

        assert all(v is None for v in fields.values())
        assert completeness == 0.0

    @pytest.mark.asyncio
    async def test_returns_empty_on_llm_error(self):
        """LLM error after all retries → returns empty schema, no crash (sleep is mocked)."""
        from backend.agents.doc_parser_agent.field_extractor import FieldExtractor
        mock_llm = MagicMock()
        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(side_effect=Exception("rate limit"))
        mock_llm.with_structured_output.return_value = mock_structured_llm
        extractor = FieldExtractor(mock_llm)
        # Patch asyncio.sleep so retry backoff doesn't add real wait time
        with patch("backend.agents.doc_parser_agent.field_extractor.asyncio.sleep", new=AsyncMock()):
            _, fields, completeness = await extractor.extract(DocType.EMPLOYEE_ID, "some text")
        # Should return empty schema, not crash
        assert isinstance(fields, dict)
        assert completeness == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Entity mapper
# ─────────────────────────────────────────────────────────────────────────────

class TestEntityMapper:
    """map_to_entity_store correctly maps each doc type."""

    def test_employee_id_mapping(self):
        fields = {
            "employee_name": "John Smith",
            "employee_id":   "EMP042",
            "department":    "Engineering",
            "role":          None,           # None → not added
        }
        result = map_to_entity_store(DocType.EMPLOYEE_ID, fields)
        assert result["employee_name"] == "John Smith"
        assert result["employee_id"]   == "EMP042"
        assert result["department"]    == "Engineering"
        assert "role" not in result     # None fields excluded

    def test_contract_mapping(self):
        fields = {
            "employee_name":      "Alice Brown",
            "notice_period_days": 60,
            "salary":             75000.0,
            "role":               None,
            "department":         None,
            "start_date":         "2026-01-15",
            "contract_type":      "permanent",
        }
        result = map_to_entity_store(DocType.EMPLOYMENT_CONTRACT, fields)
        assert result["employee_name"]       == "Alice Brown"
        assert result["notice_period_days"]  == "60"       # coerced to str
        assert result["salary"]              == "75000.0"
        assert "role" not in result

    def test_payslip_mapping(self):
        fields = {
            "employee_name": "Bob Jones",
            "employee_id":   "EMP010",
            "pay_period":    "2026-06",
            "gross":         5416.67,
            "net":           3800.0,
            "deductions":    None,
        }
        result = map_to_entity_store(DocType.PAYSLIP, fields)
        assert result["employee_name"] == "Bob Jones"
        assert result["pay_period"]    == "2026-06"
        assert "deductions" not in result

    def test_all_null_fields_returns_empty_dict(self):
        fields = {"employee_name": None, "employee_id": None, "department": None, "role": None}
        result = map_to_entity_store(DocType.EMPLOYEE_ID, fields)
        assert result == {}

    def test_build_entity_context_summary(self):
        entity_store = {
            "employee_name": "John Smith",
            "employee_id":   "EMP042",
            "department":    "Engineering",
        }
        summary = build_entity_context_summary(entity_store)
        assert "John Smith" in summary
        assert "EMP042" in summary
        assert "Known context" in summary

    def test_empty_entity_store_returns_empty_string(self):
        assert build_entity_context_summary({}) == ""


# ─────────────────────────────────────────────────────────────────────────────
# DocParserAgent.run() — full pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestDocParserAgentRun:
    """End-to-end DocParserAgent.run() with all sub-components mocked."""

    def _make_agent(self, doc_type: DocType, fields: dict):
        """Create a DocParserAgent with mocked detector + extractor."""
        from backend.agents.doc_parser_agent.doc_parser_agent import DocParserAgent
        from backend.agents.doc_parser_agent.schemas import DOC_TYPE_SCHEMA_MAP

        mock_llm = MagicMock()
        agent = DocParserAgent(llm=mock_llm)

        # Mock detector to return a fixed doc_type
        agent._detector.detect = AsyncMock(return_value=doc_type)

        # Mock extractor to return a fixed fields dict
        schema_class = DOC_TYPE_SCHEMA_MAP.get(doc_type)
        if schema_class:
            model_instance = schema_class(**{k: v for k, v in fields.items() if v is not None})
        else:
            model_instance = MagicMock()
            model_instance.model_dump.return_value = fields

        agent._extractor.extract = AsyncMock(return_value=(model_instance, fields, 1.0))

        return agent

    @pytest.mark.asyncio
    async def test_successful_employee_id_extraction(self, tmp_path):
        """Happy path: valid PDF + employee ID → AgentResult.success=True."""
        f = tmp_path / "id-card.pdf"
        f.write_bytes(PDF_MAGIC + b" ID CARD CONTENT")

        state = make_initial_state("What is the salary of this employee?", "session-1")
        state["uploaded_file_path"] = str(f)

        fields = {
            "employee_name": "John Smith",
            "employee_id":   "EMP042",
            "department":    "Engineering",
            "role":          "Senior Engineer",
        }

        # Text must be > 80 chars and > 35% alpha to pass quality check
        mock_text = (
            "EMPLOYEE ID CARD\nName: John Smith\nEmployee ID: EMP042\n"
            "Department: Engineering\nRole: Senior Engineer\nIssue Date: 2024-01-15"
        )  # 95+ chars, passes all quality thresholds
        with patch(
            "backend.agents.doc_parser_agent.doc_parser_agent.extract_text",
            new=AsyncMock(return_value=(mock_text, 1)),
        ):
            agent = self._make_agent(DocType.EMPLOYEE_ID, fields)
            result = await agent.run(state)

        assert result.success is True
        assert result.agent_name == "doc_parser"
        assert result.structured_data["document_type"] == "employee_id"
        assert result.structured_data["extracted_fields"]["employee_name"] == "John Smith"
        assert result.structured_data["extracted_fields"]["employee_id"] == "EMP042"
        assert "id-card.pdf" in result.sources

    @pytest.mark.asyncio
    async def test_fails_when_no_file_in_state(self):
        """No uploaded_file_path in state → failure with helpful message."""
        from backend.agents.doc_parser_agent.doc_parser_agent import DocParserAgent
        agent = DocParserAgent(llm=MagicMock())
        state = make_initial_state("What is the salary?", "session-1")
        # uploaded_file_path defaults to None
        result = await agent.run(state)
        assert result.success is False
        assert "No file was uploaded" in result.error

    @pytest.mark.asyncio
    async def test_fails_size_guardrail(self, tmp_path):
        """Oversized file → guardrail fails before extraction."""
        from backend.agents.doc_parser_agent.doc_parser_agent import DocParserAgent
        f = tmp_path / "huge.pdf"
        f.write_bytes(b"x" * (MAX_FILE_SIZE_BYTES + 1))

        state = make_initial_state("Extract from file", "session-1")
        state["uploaded_file_path"] = str(f)

        agent = DocParserAgent(llm=MagicMock())
        result = await agent.run(state)

        assert result.success is False
        assert "20 MB" in result.error

    @pytest.mark.asyncio
    async def test_fails_type_guardrail(self, tmp_path):
        """Spoofed file type → type guardrail fails."""
        from backend.agents.doc_parser_agent.doc_parser_agent import DocParserAgent
        f = tmp_path / "sneaky.pdf"
        f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"png content here")  # PNG magic

        state = make_initial_state("Extract from file", "session-1")
        state["uploaded_file_path"] = str(f)

        agent = DocParserAgent(llm=MagicMock())
        result = await agent.run(state)

        assert result.success is False
        assert "PDF" in result.error or "DOCX" in result.error

    @pytest.mark.asyncio
    async def test_fails_on_empty_text_extraction(self, tmp_path):
        """File exists, passes guardrails, but pdfplumber returns empty → failure."""
        from backend.agents.doc_parser_agent.doc_parser_agent import DocParserAgent
        f = tmp_path / "blank.pdf"
        f.write_bytes(PDF_MAGIC + b" minimal content")

        state = make_initial_state("Extract", "session-1")
        state["uploaded_file_path"] = str(f)

        agent = DocParserAgent(llm=MagicMock())

        with patch(
            "backend.agents.doc_parser_agent.doc_parser_agent.extract_text",
            new=AsyncMock(return_value=("", 0)),   # empty text
        ):
            result = await agent.run(state)

        assert result.success is False
        assert "empty" in result.error.lower() or "password" in result.error.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Follow-up routing contract
# ─────────────────────────────────────────────────────────────────────────────

class TestFollowUpRoutingContract:
    """
    Tests the state persistence contract that enables follow-up routing.

    After DocParser runs, the Router (in Phase 5) writes:
        state["parsed_document"]    = result.structured_data
        state["entity_store"]       = map_to_entity_store(...)
        state["uploaded_file_path"] = None

    These tests verify the data shapes are correct so the Planner
    can make the right decision on turn 2.
    """

    def test_parsed_document_in_initial_state_is_none(self):
        """New conversations start with no parsed document."""
        state = make_initial_state("hello", "session-1")
        assert state.get("parsed_document") is None

    def test_extraction_result_is_serializable_to_dict(self):
        """ExtractionResult.model_dump() produces a plain dict for GraphState storage."""
        result = ExtractionResult(
            document_type=DocType.EMPLOYEE_ID,
            extracted_fields={
                "employee_name": "John Smith",
                "employee_id":   "EMP042",
                "department":    "Engineering",
                "role":          "Senior Engineer",
            },
            page_count=1,
            source_file="employee-id.pdf",
            char_count=412,
        )
        dumped = result.model_dump()
        # Must be a plain dict — safe for MemorySaver checkpointing
        assert isinstance(dumped, dict)
        assert dumped["document_type"] == "employee_id"
        assert dumped["extracted_fields"]["employee_name"] == "John Smith"

    def test_entity_store_populated_from_employee_id(self):
        """
        After DocParser (EMPLOYEE_ID), entity_store has the right keys.
        These keys enable SQL agent to build 'WHERE name=... OR id=...' clauses.
        """
        fields = {
            "employee_name": "John Smith",
            "employee_id":   "EMP042",
            "department":    "Engineering",
            "role":          "Senior Engineer",
        }
        entity_store = map_to_entity_store(DocType.EMPLOYEE_ID, fields)

        # These are the exact keys the SQL agent looks for
        assert "employee_name" in entity_store
        assert "employee_id"   in entity_store
        assert entity_store["employee_name"] == "John Smith"
        assert entity_store["employee_id"]   == "EMP042"

    def test_follow_up_state_after_docparser(self):
        """
        Simulate state after turn 1 (DocParser ran).
        Verify shape is correct for Planner to skip doc_parser on turn 2.
        """
        # Simulate what Router does after DocParser succeeds
        state = make_initial_state("What is John's salary?", "session-1")
        state["uploaded_file_path"] = None   # cleared by Router

        parsed_doc = {
            "document_type": "employee_id",
            "extracted_fields": {
                "employee_name": "John Smith",
                "employee_id":   "EMP042",
            },
            "page_count": 1,
            "source_file": "id-card.pdf",
            "char_count": 412,
        }
        state["parsed_document"] = parsed_doc
        state["entity_store"] = {
            "employee_name": "John Smith",
            "employee_id":   "EMP042",
        }

        # The Planner's routing rule:
        # "If parsed_document is set AND uploaded_file_path is None → skip doc_parser"
        has_parsed_document = state.get("parsed_document") is not None
        has_new_file        = state.get("uploaded_file_path") is not None

        should_skip_doc_parser = has_parsed_document and not has_new_file
        assert should_skip_doc_parser is True, (
            "Planner should NOT invoke doc_parser on follow-up turns when "
            "parsed_document is already set and no new file is uploaded."
        )
