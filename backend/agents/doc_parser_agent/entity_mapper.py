"""
backend/agents/doc_parser_agent/entity_mapper.py
=================================================
Maps extracted document fields into the GraphState entity_store format.

Purpose:
    After DocParser extracts fields (e.g. EmployeeIDFields), the Router
    calls map_to_entity_store() to populate entity_store with the key
    identity fields. This enables subsequent agents (SQL, RAG) to reference
    the employee without re-parsing the document.

Entity store format:
    entity_store is a flat dict[str, str] — all values are strings for
    consistent downstream handling (SQL agent builds WHERE clauses from them).

Follow-up routing mechanism:
    After entity_store is populated from a DocParser result:
    - entity_store["employee_name"] = "John Smith"
    - entity_store["employee_id"]   = "EMP042"
    
    On follow-up turns, the Planner sees parsed_document is set and
    entity_store is populated → routes to SQL/RAG directly, not doc_parser.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.agents.doc_parser_agent.schemas import DocType

logger = logging.getLogger(__name__)


def map_to_entity_store(
    doc_type: DocType,
    extracted_fields: dict[str, Any],
) -> dict[str, str]:
    """
    Convert extracted document fields into entity_store entries.

    Only non-None fields are added to the entity_store.
    All values are coerced to str for consistency.

    Parameters
    ----------
    doc_type : DocType
        The type of document that was parsed.
    extracted_fields : dict[str, Any]
        The .model_dump() output from the extraction Pydantic schema.

    Returns
    -------
    dict[str, str]
        Key-value pairs to merge into GraphState["entity_store"].
        Returns an empty dict if no useful entities are found.

    Examples
    --------
    >>> map_to_entity_store(
    ...     DocType.EMPLOYEE_ID,
    ...     {"employee_name": "John Smith", "employee_id": "EMP042", "department": "Engineering", "role": None}
    ... )
    {"employee_name": "John Smith", "employee_id": "EMP042", "department": "Engineering"}
    """
    entity_store: dict[str, str] = {}

    def _add(key: str, value: Any) -> None:
        """Add a non-None value to entity_store as a string."""
        if value is not None:
            entity_store[key] = str(value)

    if doc_type == DocType.EMPLOYEE_ID:
        _add("employee_name", extracted_fields.get("employee_name"))
        _add("employee_id",   extracted_fields.get("employee_id"))
        _add("department",    extracted_fields.get("department"))
        _add("role",          extracted_fields.get("role"))

    elif doc_type == DocType.EMPLOYMENT_CONTRACT:
        _add("employee_name",       extracted_fields.get("employee_name"))
        _add("role",                extracted_fields.get("role"))
        _add("department",          extracted_fields.get("department"))
        _add("salary",              extracted_fields.get("salary"))
        _add("notice_period_days",  extracted_fields.get("notice_period_days"))
        _add("start_date",          extracted_fields.get("start_date"))
        _add("contract_type",       extracted_fields.get("contract_type"))

    elif doc_type == DocType.PAYSLIP:
        _add("employee_name", extracted_fields.get("employee_name"))
        _add("employee_id",   extracted_fields.get("employee_id"))
        _add("pay_period",    extracted_fields.get("pay_period"))
        _add("gross",         extracted_fields.get("gross"))
        _add("net",           extracted_fields.get("net"))

    elif doc_type == DocType.OFFER_LETTER:
        _add("candidate_name", extracted_fields.get("candidate_name"))
        _add("role",           extracted_fields.get("role"))
        _add("department",     extracted_fields.get("department"))
        _add("salary",         extracted_fields.get("salary"))

    if entity_store:
        logger.info(
            "entity_mapper: populated %d entity_store keys from %s: %s",
            len(entity_store),
            doc_type.value,
            list(entity_store.keys()),
        )
    else:
        logger.warning(
            "entity_mapper: no entities extracted from %s — entity_store unchanged",
            doc_type.value,
        )

    return entity_store


def build_entity_context_summary(entity_store: dict[str, str]) -> str:
    """
    Build a human-readable summary of the current entity_store for injection
    into SQL/RAG agent prompts.

    This summary is prepended to the query so agents know which employee
    the user is asking about — even on follow-up turns with no file.

    Example output:
        "Known context from uploaded document: employee_name=John Smith, employee_id=EMP042"

    Parameters
    ----------
    entity_store : dict[str, str]
        The current GraphState entity_store.

    Returns
    -------
    str
        Empty string if entity_store is empty, otherwise a context hint.
    """
    if not entity_store:
        return ""

    # Priority fields to surface first
    priority_keys = ["employee_name", "employee_id", "department", "role"]
    other_keys = [k for k in entity_store if k not in priority_keys]

    ordered_keys = [k for k in priority_keys if k in entity_store] + other_keys
    pairs = ", ".join(f"{k}={entity_store[k]}" for k in ordered_keys)

    return f"Known context from uploaded document: {pairs}"
