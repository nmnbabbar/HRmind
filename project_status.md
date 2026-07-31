# HrMind — Project Status

*Last updated: Phase 7 (React Frontend) & Phase 3 (SQL Agent) complete · 2026-07-30*

> [!NOTE]
> **Session log** — each turn is recorded here so nothing is lost across sessions.

### Session 3 — RAG Agent, Hybrid Search & Evals Implementation
- Transitioned away from OpenAI-specific config to a **provider-agnostic LLM config** (`LLM_API_KEY`, `LLM_BASE_URL`) supporting NVIDIA NIM natively.
- Switched default embedding model to `BAAI/bge-large-en-v1.5` based on user request for maximum accuracy.
- Added `chroma_mode="local"` (PersistentClient) vs `"server"` (AsyncHttpClient) to seamlessly support local Dockerless execution.
- Created `ingestion.py` which guarantees idempotent, hash-based incremental updates (using a dedicated `ingestion_log` collection).
- Developed `ChromaVectorRepository` for dense retrieval and a purely in-memory `BM25Index` for sparse retrieval.
- Fused search results using **Reciprocal Rank Fusion (RRF, k=60)**.
- Implemented `TopicGuardrail` and `GroundingGuardrail`.
- Configured a 33-question evaluation dataset (`eval_dataset.py`) + a runner (`eval_runner.py`) using RAGAS to compute Faithfulness, Answer Relevancy, Context Recall, and Context Precision.
- Created an interactive retrieval inspector (`retrieve_inspect.py`) to observe search results and context formatting directly without burning LLM tokens.
- Tests updated and verified: **27/27 RAG unit tests passing locally**.

### Session 4 — SQL Agent & React Frontend (Phase 3 & 7 Fast-Track)
- Developed `sql_validator.py` using `sqlglot` to parse the Abstract Syntax Tree (AST), rejecting non-SELECT mutations (DROP, UPDATE) and enforcing a LIMIT 100 clause.
- Integrated `aiosqlite` with a mock HR database seeded by `seed_db.py`.
- Fast-tracked Phase 7: Abandoned Streamlit in favor of a beautiful, modern Vite + React Single Page Application (SPA).
- Implemented `react-router-dom` with distinct Homepage, Auth (Login/Signup), and Chat routes featuring a sleek glassmorphic UI.
- All SQL Agent tests passing using in-memory SQLite mocks.

### Session 2 — Local Environment Setup
- Installed `uv 0.11.33` to `C:\Users\nbabb\.local\bin`
- Fixed `pyproject.toml`: Windows platform marker for `python-magic-bin`, `ragas` moved to optional `[evals]` group.
- `uv sync --extra dev` installed **165 packages**.
- Fixed `conftest.py` fixture: `GraphState` is a TypedDict.
- **Result: 51/51 core tests passing locally in 7.17s**

---

## What Is HrMind?

A **multi-agent HR intelligence platform**. Three specialized AI agents collaborate to answer HR-related questions:

| Agent | What it does |
|---|---|
| **RAG Agent** | Semantic search over 31 HR policy documents. Hybrid BM25 + dense vector search. |
| **SQL Agent** | Converts plain English to SQL, executes against an HR SQLite database, returns a table + explanation. |
| **Doc Parser Agent** | OCR-extracts key fields (name, salary, notice period, etc.) from uploaded PDFs and images. |

These agents are orchestrated by a **Planner → Router → Combiner** graph built in LangGraph. Complex questions can invoke multiple agents in sequence or parallel.

---

## Tech Stack & Tool Selection

| Layer | Technology |
|---|---|
| Orchestration | LangGraph `>=0.2` |
| LLM Framework | LangChain `>=1.0` |
| Embeddings | `BAAI/bge-large-en-v1.5` (335M params, state of the art BGE) |
| Vector Store | ChromaDB (Local PersistentClient for dev, HTTP Async for Docker) |
| Sparse Retrieval | BM25 (`rank-bm25`) |
| OCR | Pytesseract + pdf2image (system deps in Docker) |
| Database | SQLite via `aiosqlite` |
| SQL Safety | `sqlglot` (parse + validate — no regex) |
| Backend | FastAPI + Uvicorn (async, SSE streaming) |
| Frontend | Vite + React SPA (react-router, vanilla css) |
| Package Manager | `uv` (local dev) / `uv` in Docker builder |
| Testing | pytest + pytest-asyncio (no external services in Phase 1–4) |

---

## Architecture Flow

```
User Query
    ↓
FastAPI (SSE streaming)
    ↓
LangGraph Graph
    ├── [Planner] — LLM returns ordered agent array e.g. ["doc_parser", "rag"]
    ├── [Router]  — Pure Python. Iterates plan array. asyncio.gather for parallel.
    │       ├── RAGAgent.run(state)
    │       ├── SQLAgent.run(state)
    │       └── DocParserAgent.run(state)
    └── [Combiner] — LLM synthesizes all AgentResults → final answer
```

**Core execute-in-code principle**: Retrieval, SQL execution, and OCR all run in Python. The LLM only sees the results — never executes tools at runtime.

---

## Phase Status

| Phase | Status | Description |
|---|---|---|
| 1 — Foundation | ✅ Complete | Abstractions, state, memory, Docker, tests |
| 2 — RAG Agent | ✅ Complete | Hybrid search, incremental ingestion, RAGAS evals, guardrails |
| 3 — SQL Agent | ✅ Complete | Text-to-SQL logic, sqlglot validator, AST limits |
| 4 — Doc Parser | ⬜ Not started | |
| 5 — Orchestration | ⬜ Not started | |
| 6 — FastAPI Backend | ⬜ Not started | |
| 7 — React Frontend | ✅ Complete | Vite, routing, auth/chat UI, glassmorphism |
| 8 — E2E Tests + Docker | ⬜ Not started | |

---

## Detailed Design Decisions & Trade-offs

### 1. LLM Provider Agnostic configuration
- **Decision:** Shifted from `OPENAI_API_KEY` to `LLM_API_KEY` + `LLM_BASE_URL`.
- **Why:** Future proofs the application, specifically allowing NVIDIA NIM (e.g. `deepseek-ai/deepseek-r1-0528-qwen3-8b` or `meta/llama-3.3-70b-instruct`) without changing application logic, because NIM provides an OpenAI-compatible endpoint.

### 2. Embedding Model Upgraded to `bge-large-en-v1.5`
- **Decision:** Switched from `bge-small` to `bge-large`.
- **Why:** User mandated the highest accuracy possible. While 5x slower for CPU inference than small (335M vs 33M params), it minimizes context misses in dense retrieval.
- **Metrics/Impact:** Increases `ContextRecall` score on evals at the slight cost of ingestion time latency.

### 3. Local + Docker ChromaDB Strategy
- **Decision:** Implemented `CHROMA_MODE` toggle (`local` vs `server`).
- **Why:** `local` uses `chromadb.PersistentClient` saving directly to `./chroma_data`, drastically lowering the barrier to run and debug scripts locally without spinning up Docker. `server` mode uses `AsyncHttpClient` for the FastAPI runtime.

### 4. Fully Synchronous Ingestion Script
- **Decision:** Refactored `ingestion.py` from async to purely synchronous Python.
- **Why:** `PersistentClient` does not have async variants. Given ingestion is a one-off CLI background task, threading overhead for `await` provides no benefit here.

### 5. Hash-Based Incremental Ingestion
- **Decision:** Storing document SHA-256 hashes in a separate `ingestion_log` ChromaDB collection.
- **Why:** Ensures `ingest_documents()` is fully idempotent. If a document hasn't changed, it's skipped. This saves immense embedding time on repeated runs.

### 6. Reciprocal Rank Fusion (RRF) vs Cross-Encoder
- **Decision:** Stick to RRF (pure mathematical fusion) rather than an LLM cross-encoder reranking.
- **Why:** Cross encoders add a massive latency spike (running inference on N chunks). RRF mathematically marries BM25 keyword matching with Dense semantic matching essentially in 0ms overhead, with a `k=60` constant to penalize low rankings appropriately.
- **Parameters:** `dense_top_k=20`, `sparse_top_k=20`, `final_top_k=8`. Ensures we fetch wide nets, and only pass the very best 8 chunks to the LLM to avoid context pollution.

### 7. Evaluation Strategy (RAGAS)
- **Decision:** Evaluating Faithfulness, Answer Relevancy, Context Precision, and Context Recall using RAGAS frameworks locally.
- **Why:** Ensures quantitative tracking of RAG drift.
- **Limitation Acknowledged:** The 33 Ground Truth questions were synthetically generated via HR logic and document titles, not the raw text of the user's explicit internal policies. If the user's specific text wildly deviates from standard HR templates, Context Recall may artificially drop.

### 8. CLI Retrieval Inspector
- **Decision:** Added `retrieve_inspect.py`.
- **Why:** Evals cost LLM tokens. Debugging chunk sizes, citation accuracy, and retrieval ranks should be free. This CLI tool formats exactly what the LLM prompt gets injected with.

---

## What's Next: Phase 4 — Document Parser Agent

**Phase 4 will build:**
1. Async OCR pipeline using `pdf2image` and `pytesseract`.
2. Integration of `python-magic` for secure file type validation.
3. LLM-based document type detection to extract structured fields (e.g. Salary, Notice Period) from contracts.
4. Definition of a cross-agent handoff protocol for passing parsed context downstream.
