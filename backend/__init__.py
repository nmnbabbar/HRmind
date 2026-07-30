"""
HrMind Backend Package
======================
Multi-agent HR intelligence platform.

Entry point for the backend package. Sub-packages:
- backend.api       — FastAPI application and route handlers
- backend.agents    — RAG, SQL, and DocParser agent implementations
- backend.base      — Abstract interfaces (Agent, Guardrail, Repository)
- backend.memory    — Context compression, entity extraction, summarization
- backend.orchestration — LangGraph graph: Planner → Router → Combiner
- backend.utils     — Logging, token counting, shared helpers
"""
