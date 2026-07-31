"""
backend/agents/doc_parser_agent
================================
Document Parser Agent — Phase 4.

Extracts structured fields from uploaded PDFs and DOCX files.
Supports 4 document types: employment_contract, payslip, employee_id, offer_letter.

Public interface:
    from backend.agents.doc_parser_agent import DocParserAgent
"""

from backend.agents.doc_parser_agent.doc_parser_agent import DocParserAgent

__all__ = ["DocParserAgent"]
