"""
backend/config.py
=================
Centralised application settings loaded from environment variables / .env file.

All configuration lives here — no magic strings scattered across the codebase.
Uses Pydantic Settings for automatic env-var binding and validation.

Usage
-----
    from backend.config import get_settings
    settings = get_settings()
    settings.llm_api_key  # type-safe, validated
"""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application-wide configuration.

    Values are read (in order):
    1. Environment variables (highest priority)
    2. .env file in the working directory
    3. Field defaults (lowest priority)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # ignore unknown env vars — don't blow up in prod
    )

    # ── LLM Provider (OpenAI-compatible) ─────────────────────────────────────
    # Supports any OpenAI-compatible API: OpenAI, NVIDIA NIM, Together, Groq, etc.
    # Set LLM_BASE_URL to override the endpoint (e.g. NVIDIA NIM).
    # Leave LLM_BASE_URL empty / unset to use the default OpenAI endpoint.
    llm_api_key: str = ""
    llm_base_url: str = "https://api.groq.com/openai/v1"  # Groq endpoint

    # Model names — all three can point to the same model or different ones.
    planner_model: str = "openai/gpt-oss-120b"
    combiner_model: str = "openai/gpt-oss-120b"
    agent_model: str = "openai/gpt-oss-120b"


    # ── ChromaDB ──────────────────────────────────────────────────────────────
    # mode="server"  → AsyncHttpClient (Docker / production)
    # mode="local"   → PersistentClient (local dev, no Docker required)
    chroma_mode: str = "local"                      # "server" | "local"
    chroma_host: str = "localhost"                   # used only in server mode
    chroma_port: int = 8000                          # used only in server mode
    chroma_persist_dir: str = "./chroma_data"        # used only in local mode
    chroma_collection_name: str = "hr_documents"

    # ── Embeddings ────────────────────────────────────────────────────────────
    # bge-large: 335M params, highest accuracy in BGE family
    # Slower than bge-small (~5x) but provides maximum retrieval quality
    embedding_model: str = "BAAI/bge-large-en-v1.5"

    # ── Retrieval ─────────────────────────────────────────────────────────────
    dense_top_k: int = 20    # candidates from dense (ChromaDB) search
    sparse_top_k: int = 20   # candidates from sparse (BM25) search
    final_top_k: int = 8     # after RRF fusion: top chunks sent to LLM
    rrf_k: int = 60          # RRF constant (higher = less aggressive rank discounting)

    # ── SQL Agent ─────────────────────────────────────────────────────────────
    sql_max_rows: int = 100  # hard cap on query result rows

    # ── Memory / Context Compression ──────────────────────────────────────────
    summary_every_n_turns: int = 10    # compress history after N user turns
    recent_turns_to_keep: int = 3      # verbatim turns always passed to agents

    # ── Context Token Budgets ─────────────────────────────────────────────────
    # Priority: system_prompt > current_query > summary > recent_turns > agent_context
    max_system_prompt_tokens: int = 1000
    max_summary_tokens: int = 600
    max_recent_turns_tokens: int = 800
    max_agent_context_tokens: int = 1000

    # ── File Upload ───────────────────────────────────────────────────────────
    upload_dir: str = "/app/uploads"
    max_file_size_mb: int = 20
    upload_ttl_hours: int = 1  # files deleted after this many hours

    # ── Paths ─────────────────────────────────────────────────────────────────
    hr_docs_path: str = "./data/hr_documents"        # local dev path (override in Docker)
    sqlite_db_path: str = "./data/hr_database.sqlite"  # local dev path

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = "INFO"  # DEBUG | INFO | WARNING | ERROR

    # ── LangSmith Tracing ─────────────────────────────────────────────────────
    langsmith_tracing: str = "true"
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_api_key: str = ""
    langsmith_project: str = "HRMIND"


def setup_langsmith(settings: Settings) -> None:
    """Export LangSmith configuration to environment variables for LangChain / LangGraph tracing."""
    if settings.langsmith_tracing:
        val = str(settings.langsmith_tracing).strip('"\'').lower()
        os.environ["LANGSMITH_TRACING"] = val
        os.environ["LANGCHAIN_TRACING_V2"] = val
        if settings.langsmith_endpoint:
            os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
            os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint
        if settings.langsmith_api_key:
            os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
            os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        if settings.langsmith_project:
            proj = settings.langsmith_project.strip('"\'')
            os.environ["LANGSMITH_PROJECT"] = proj
            os.environ["LANGCHAIN_PROJECT"] = proj


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the singleton Settings instance.

    Cached after first call — safe to call repeatedly without re-reading env.
    Cache is invalidated in tests via: get_settings.cache_clear()
    """
    settings = Settings()
    setup_langsmith(settings)
    return settings

