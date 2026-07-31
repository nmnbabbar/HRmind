"""
backend/agents/doc_parser_agent/schemas.py
==========================================
Pydantic schemas for each document type the DocParser agent can extract.

These schemas are used with `llm.with_structured_output(SchemaClass)` to force
the LLM to return validated, typed data.

Design decisions:
- All fields are `str | None` or `float | None` — the LLM returns None for any
  field it cannot find, never hallucinating a value.
- Dates are stored as ISO 8601 strings ("YYYY-MM-DD") for easy serialization.
- Salary / financial amounts are raw floats in the document's currency unit.
- Field validators silently coerce out-of-range values to None rather than
  raising — the LLM may produce plausible-looking but wrong values, and None
  is always safer than a wrong number.
"""

from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, Field, field_validator


# ── Document type enum ─────────────────────────────────────────────────────────

class DocType(str, Enum):
    """Supported document types the DocParser can classify and extract from."""

    EMPLOYMENT_CONTRACT = "employment_contract"
    PAYSLIP             = "payslip"
    EMPLOYEE_ID         = "employee_id"
    OFFER_LETTER        = "offer_letter"
    UNKNOWN             = "unknown"


# ── Per-document-type extraction schemas ───────────────────────────────────────

class EmploymentContractFields(BaseModel):
    """
    Structured fields extracted from an employment contract.

    Standalone use: answer questions about the contract's terms.
    Cross-agent use (DocParser → RAG): check if notice_period_days matches policy.
    """

    employee_name: str | None = Field(
        default=None,
        description="Full name of the employee as it appears in the contract."
    )
    role: str | None = Field(
        default=None,
        description="Job title or role as specified in the contract."
    )
    department: str | None = Field(
        default=None,
        description="Department the employee belongs to."
    )
    start_date: str | None = Field(
        default=None,
        description="Employment start date in ISO 8601 format (YYYY-MM-DD)."
    )
    salary: float | None = Field(
        default=None,
        description="Annual salary as a numeric value. Do not include currency symbols."
    )
    notice_period_days: int | None = Field(
        default=None,
        description="Notice period in calendar days. Convert weeks/months to days if needed."
    )
    contract_type: str | None = Field(
        default=None,
        description="Type of contract: 'permanent', 'fixed-term', or 'contractor'."
    )

    @field_validator("salary", mode="before")
    @classmethod
    def salary_in_range(cls, v: float | None) -> float | None:
        """Reject salary values outside a plausible range (£0–£10M annual)."""
        if v is None:
            return None
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        if v <= 0 or v > 10_000_000:
            return None   # coerce to None — safer than a hallucinated value
        return v

    @field_validator("notice_period_days", mode="before")
    @classmethod
    def notice_in_range(cls, v: int | None) -> int | None:
        """Reject notice periods outside 0–730 days (0 days to 2 years)."""
        if v is None:
            return None
        try:
            v = int(v)
        except (TypeError, ValueError):
            return None
        if v < 0 or v > 730:
            return None
        return v

    @field_validator("start_date", mode="before")
    @classmethod
    def start_date_format(cls, v: str | None) -> str | None:
        """Coerce non-ISO dates to None. Accept YYYY-MM-DD only."""
        if v is None:
            return None
        if re.match(r"^\d{4}-\d{2}-\d{2}$", str(v)):
            return str(v)
        return None   # LLM returned a non-standard date format — discard

    @field_validator("contract_type", mode="before")
    @classmethod
    def contract_type_normalise(cls, v: str | None) -> str | None:
        """Normalise contract_type to known values."""
        if v is None:
            return None
        v_lower = str(v).lower().strip()
        if "permanent" in v_lower:
            return "permanent"
        if "fixed" in v_lower:
            return "fixed-term"
        if "contract" in v_lower or "freelance" in v_lower:
            return "contractor"
        return v  # keep as-is if unrecognised


class PayslipFields(BaseModel):
    """
    Structured fields extracted from a payslip.

    Standalone use: answer questions about the payslip.
    Cross-agent use (DocParser → SQL): verify extracted gross against payroll DB record.
    """

    employee_name: str | None = Field(
        default=None,
        description="Full name of the employee as it appears on the payslip."
    )
    employee_id: str | None = Field(
        default=None,
        description="Employee ID or staff number (e.g. 'EMP042')."
    )
    pay_period: str | None = Field(
        default=None,
        description="Pay period in YYYY-MM format (e.g. '2026-06')."
    )
    gross: float | None = Field(
        default=None,
        description="Gross pay as a numeric value before deductions."
    )
    net: float | None = Field(
        default=None,
        description="Net pay as a numeric value after all deductions."
    )
    deductions: float | None = Field(
        default=None,
        description="Total deductions as a numeric value (gross - net)."
    )

    @field_validator("gross", "net", "deductions", mode="before")
    @classmethod
    def pay_amount_positive(cls, v: float | None) -> float | None:
        """Pay amounts must be non-negative and below £1M monthly."""
        if v is None:
            return None
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        if v < 0 or v > 1_000_000:
            return None
        return v

    @field_validator("pay_period", mode="before")
    @classmethod
    def pay_period_format(cls, v: str | None) -> str | None:
        """Accept YYYY-MM format only."""
        if v is None:
            return None
        if re.match(r"^\d{4}-\d{2}$", str(v)):
            return str(v)
        # Try to extract YYYY-MM from a longer date string
        match = re.search(r"(\d{4})-(\d{2})", str(v))
        if match:
            return f"{match.group(1)}-{match.group(2)}"
        return None


class EmployeeIDFields(BaseModel):
    """
    Structured fields extracted from an employee ID card.

    Key cross-agent use case (DocParser → SQL):
        Upload ID card → extract employee_name / employee_id → SQL queries
        salary, leave, payroll, performance, etc.
    """

    employee_name: str | None = Field(
        default=None,
        description="Full name of the employee as printed on the ID card."
    )
    employee_id: str | None = Field(
        default=None,
        description="Employee ID, staff number, or badge number (e.g. 'EMP042')."
    )
    department: str | None = Field(
        default=None,
        description="Department shown on the ID card."
    )
    role: str | None = Field(
        default=None,
        description="Job title or role shown on the ID card."
    )

    @field_validator("employee_name", "role", "department", mode="before")
    @classmethod
    def strip_and_validate_text(cls, v: str | None) -> str | None:
        """Strip whitespace and reject single-character or all-numeric values."""
        if v is None:
            return None
        v = str(v).strip()
        if len(v) < 2:
            return None   # single char is not a real name/role
        if v.isdigit():
            return None   # all-numeric value in a text field is suspicious
        return v

    @field_validator("employee_id", mode="before")
    @classmethod
    def employee_id_nonempty(cls, v: str | None) -> str | None:
        """Employee ID must be at least 2 chars long."""
        if v is None:
            return None
        v = str(v).strip()
        return v if len(v) >= 2 else None


class OfferLetterFields(BaseModel):
    """
    Structured fields extracted from an offer letter.

    Standalone use: answer questions about the offer.
    Cross-agent use (DocParser → RAG): check if proposed salary matches salary bands policy.
    """

    candidate_name: str | None = Field(
        default=None,
        description="Full name of the candidate the offer is addressed to."
    )
    role: str | None = Field(
        default=None,
        description="Job title or role being offered."
    )
    salary: float | None = Field(
        default=None,
        description="Offered annual salary as a numeric value."
    )
    start_date: str | None = Field(
        default=None,
        description="Proposed start date in ISO 8601 format (YYYY-MM-DD)."
    )
    department: str | None = Field(
        default=None,
        description="Department the candidate will be joining."
    )

    @field_validator("salary", mode="before")
    @classmethod
    def salary_in_range(cls, v: float | None) -> float | None:
        """Offered salary must be in a plausible range."""
        if v is None:
            return None
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        if v <= 0 or v > 10_000_000:
            return None
        return v

    @field_validator("start_date", mode="before")
    @classmethod
    def start_date_format(cls, v: str | None) -> str | None:
        """Accept ISO 8601 dates only."""
        if v is None:
            return None
        if re.match(r"^\d{4}-\d{2}-\d{2}$", str(v)):
            return str(v)
        return None


# ── Schema registry ────────────────────────────────────────────────────────────

DOC_TYPE_SCHEMA_MAP: dict[DocType, type[BaseModel]] = {
    DocType.EMPLOYMENT_CONTRACT: EmploymentContractFields,
    DocType.PAYSLIP:             PayslipFields,
    DocType.EMPLOYEE_ID:         EmployeeIDFields,
    DocType.OFFER_LETTER:        OfferLetterFields,
}


# ── ExtractionResult — the full DocParser output ───────────────────────────────

class ExtractionResult(BaseModel):
    """
    Full result of the document parsing pipeline.

    Stored in AgentResult.structured_data and (after Router processing)
    in GraphState["parsed_document"] for cross-turn persistence.

    completeness_score:
        Fraction of non-None fields (0.0–1.0).
        1.0 = all fields extracted, 0.0 = nothing found.
        Used by the Combiner to assess extraction quality and decide whether
        to surface a warning ("only partial information was extracted").
    """

    document_type:       DocType
    extracted_fields:    dict          # .model_dump() of per-type schema
    page_count:          int
    source_file:         str
    char_count:          int
    completeness_score:  float = 0.0   # 0.0–1.0, computed post-extraction
    cache_hit:           bool  = False # True if result came from cache (no LLM)

    @classmethod
    def compute_completeness(cls, fields_dict: dict) -> float:
        """
        Compute what fraction of extracted fields are non-None.

        Parameters
        ----------
        fields_dict : dict
            .model_dump() output from the per-type Pydantic schema.

        Returns
        -------
        float
            0.0 if all fields are None, 1.0 if all fields have values.
        """
        if not fields_dict:
            return 0.0
        non_null = sum(1 for v in fields_dict.values() if v is not None)
        return round(non_null / len(fields_dict), 2)
