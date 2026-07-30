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
    llm_api_key: str           # required — API key for the LLM provider
    llm_base_url: str = "https://integrate.api.nvidia.com/v1"  # NVIDIA NIM endpoint

    # Model names — all three can point to the same model or different ones.
    # Currently: deepseek-ai/deepseek-r1-0528-qwen3-8b via NVIDIA NIM
    planner_model: str = "deepseek-ai/deepseek-r1-0528-qwen3-8b"
    combiner_model: str = "deepseek-ai/deepseek-r1-0528-qwen3-8b"
    agent_model: str = "deepseek-ai/deepseek-r1-0528-qwen3-8b"

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
    sqlite_db_path: str = "/app/data/hr_database.sqlite"
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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the singleton Settings instance.

    Cached after first call — safe to call repeatedly without re-reading env.
    Cache is invalidated in tests via: get_settings.cache_clear()
    """
    return Settings()
