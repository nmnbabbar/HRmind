# HrMind — Revised Implementation Plan (v2)

> All issues from the critique have been addressed. Changes from v1 are marked with `[CHANGED]`.

---

## Design Principles

> [!IMPORTANT]
> **Core Theme: Execute in Code, Pass Results to LLM.** We never use tool-calling when the execution order is known. Retrieval, SQL execution, and OCR all happen in Python. The LLM only sees the *results* of these operations, not the tools themselves.

1. **SOLID throughout** — Abstract base classes for all agents, strategy pattern for guardrails, repository pattern for data access, factory for agent creation
2. **Async-first** — `asyncio` everywhere; blocking calls (HF embeddings, OCR) wrapped in `asyncio.run_in_executor`
3. **Docker-only** — No local environment assumptions. `uv` for all package management. Multi-stage builds
4. **Minimum LLM calls** — Planner → Router is code-driven. Retrieval/SQL/OCR execute in Python, LLM only synthesizes
5. **Speed-optimized** — `bge-small-en-v1.5` over large, hybrid search without cross-encoder, ChromaDB HTTP client (no file locking), async fan-out for independent agents

---

## Architecture

```
User Query (Streamlit)
        │
        ▼
  [FastAPI + SSE Streaming]
        │
        ▼
  ┌─────────────────────────────────────────────────────────┐
  │                  LangGraph Orchestrator                 │
  │                                                         │
  │  ┌──────────┐    ┌──────────┐    ┌──────────────────┐  │
  │  │ PLANNER  │───▶│  ROUTER  │───▶│    COMBINER      │  │
  │  │          │    │          │    │                  │  │
  │  │ Returns  │    │ Iterates │    │ Merges results + │  │
  │  │ ordered  │    │ plan[]   │    │ conflict policy  │  │
  │  │ array    │    │ in code  │    │                  │  │
  │  └──────────┘    └────┬─────┘    └──────────────────┘  │
  │                       │                    ▲            │
  │          asyncio.gather (parallel)         │            │
  │          ┌────────────┼────────────┐       │            │
  │          ▼            ▼            ▼       │            │
  │    [RAG Agent]  [SQL Agent]  [Doc Agent] ──┘            │
  │                                                         │
  │  [LangGraph MemorySaver — per session_id checkpoints]   │
  └─────────────────────────────────────────────────────────┘
```

---

## Workflow: Planner → Router → Agents → Combiner

### The Critical Fix: Router Drives Execution in Code `[CHANGED]`

The Planner returns a **simple ordered array** of agent names. The Router is pure Python — no LLM call, no tool calls.

```python
# Planner LLM output (structured via .with_structured_output())
class PlannerOutput(BaseModel):
    agents: list[Literal["rag", "sql", "doc_parser"]]  # e.g. ["doc_parser", "rag"]
    queries: dict[str, str]   # per-agent query rewrite
    parallel: bool            # can agents run simultaneously?

# Router node — pure Python, no LLM
async def router_node(state: GraphState) -> GraphState:
    plan = state["plan"]
    if plan.parallel:
        # asyncio.gather for independent agents
        tasks = [agent_registry[name].run(state) for name in plan.agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    else:
        # Sequential — pass previous result into next agent's context
        results = []
        for name in plan.agents:
            result = await agent_registry[name].run(state)
            state["agent_results"].append(result)   # next agent sees prior output
    return {**state, "agent_results": results}
```

### Why No `Send()` API `[CHANGED]`

`Send()` is the LangGraph API for fanning out to *dynamic* subgraph instances (e.g. a MapReduce pattern with unknown N). Our agent set is fixed (3 agents), so `asyncio.gather` inside the router node is simpler, more predictable, and easier to debug. The Router is a single node that handles all concurrency internally.

### Combiner Conflict Resolution Policy `[CHANGED]`

| Scenario | Strategy |
|---|---|
| Agents agree | Merge and present unified answer |
| Agents provide complementary info | Synthesize sequentially (Doc Parser findings → RAG validation) |
| Direct contradiction (e.g. contract says X, policy says Y) | Surface **both** explicitly: *"Your contract specifies X, however the current policy states Y. Please consult HR."* |
| One agent failed, others succeeded | Present partial result with clear failure notice |

---

## SOLID Design Patterns

### Abstract Base Agent (Open/Closed + Liskov)
```python
class BaseAgent(ABC):
    def __init__(self, llm: BaseChatModel, guardrails: list[GuardrailStrategy]):
        self.llm = llm
        self.guardrails = CompositeGuardrail(guardrails)  # injected

    @abstractmethod
    async def run(self, state: GraphState) -> AgentResult: ...

    async def _check_guardrails(self, query: str) -> GuardrailResult:
        return await self.guardrails.check(query)  # composite pattern
```

### Strategy Pattern for Guardrails (Interface Segregation)
```python
class GuardrailStrategy(Protocol):
    async def check(self, query: str) -> GuardrailResult: ...

class TopicGuardrail(GuardrailStrategy): ...       # HR domain boundary
class ReadOnlySQLGuardrail(GuardrailStrategy): ... # blocks mutating SQL
class FileSizeGuardrail(GuardrailStrategy): ...    # file upload limits
class ToxicityGuardrail(GuardrailStrategy): ...    # global profanity check

class CompositeGuardrail:                          # Chain of responsibility
    def __init__(self, strategies: list[GuardrailStrategy]): ...
    async def check(self, query: str) -> GuardrailResult:
        for strategy in self.strategies:
            result = await strategy.check(query)
            if not result.passed:
                return result
        return GuardrailResult(passed=True)
```

### Repository Pattern for Vector Store (Dependency Inversion)
```python
class VectorRepository(Protocol):
    async def similarity_search(self, query: str, k: int) -> list[Document]: ...
    async def add_documents(self, docs: list[Document]) -> None: ...

class ChromaVectorRepository(VectorRepository): ...  # concrete impl
# Swap to Qdrant/Pinecone later without touching agent code
```

### Agent Factory
```python
class AgentFactory:
    @staticmethod
    def create(agent_type: AgentType, config: Settings) -> BaseAgent:
        match agent_type:
            case AgentType.RAG:       return RAGAgent(...)
            case AgentType.SQL:       return SQLAgent(...)
            case AgentType.DOC:       return DocParserAgent(...)
```

### AgentResult — The Central Data Contract `[CHANGED]`
```python
@dataclass
class AgentResult:
    agent_name: str
    success: bool
    answer: str                          # NL answer
    sources: list[str]                   # citations / SQL query / file name
    structured_data: dict | None         # SQL table rows, extracted fields, etc.
    error: str | None                    # populated if success=False
    metadata: dict                       # timing, token usage, model used
```

---

## Memory & Context Management `[CHANGED]`

### LangGraph MemorySaver (Native Checkpointing)
```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

# Each conversation is a thread_id:
config = {"configurable": {"thread_id": session_id}}
await graph.ainvoke(state, config=config)  # auto-saves checkpoint
```

LangGraph's built-in checkpointer handles full conversation history natively. No custom Redis or dict storage needed in phases 1–7. The interface is compatible with `AsyncPostgresSaver` for production upgrade.

### Context Compression (Three-Layer Strategy)

```
Full History (LangGraph checkpoint — never trimmed)
       │
       ▼
Rolling Summary (LLM-compressed every 10 turns)
       │ + last 3 turns verbatim
       ▼
Agent-Scoped Slice (only relevant fields injected per agent)
```

**Layer 1 — Rolling Summary**: A background summarization runs after every 10th user turn. Stores a `conversation_summary` in state. Cost: 1 LLM call per 10 turns.

**Layer 2 — Agent-Scoped Injection**: Strictly enforced per agent:
| Agent | Gets | Does NOT get |
|---|---|---|
| RAG | query + summary | SQL schema, OCR blobs, prior SQL results |
| SQL | query + schema DDL + last SQL result | Document chunks, OCR text |
| Doc Parser | file path + doc type hint | Conversation history entirely |

**Layer 3 — Entity Memory**: After each turn, `spaCy en_core_web_sm` extracts entities (PERSON, ORG, DATE, MONEY). Stored as `entity_store: dict`. Follow-up questions like *"What's her salary?"* resolve *her* via entity lookup — no full history scan needed.

**Token Budget Enforcement**:
```python
class ContextBudget:
    SYSTEM_PROMPT    = 1000   # reserved
    CURRENT_QUERY    = 200    # reserved
    ROLLING_SUMMARY  = 600    # max
    RECENT_TURNS     = 800    # last 3 turns
    AGENT_CONTEXT    = 1000   # retrieved chunks / schema / OCR
    # Total ≈ 3600 tokens → well within 128k context but keeps costs low
```

---

## Guardrails — Concrete Implementations `[CHANGED]`

| Guardrail | Implementation |
|---|---|
| **HR Topic Boundary** | LLM classifier call: "Is this query HR-related? Answer YES/NO." — Simple, reliable, ~50 tokens |
| **SQL Read-Only** | Parse with `sqlglot`, check statement type. If not `SELECT`, reject. No regex. |
| **SQL Parameter Safety** | Use `aiosqlite` parameterized queries. User input never concatenated into SQL string |
| **File Type** | Magic bytes check (not extension) via `python-magic` |
| **File Size** | Hard cap at 20MB before OCR begins |
| **Toxicity** | Lightweight keyword list + LLM fallback for ambiguous cases |
| **Answer Grounding** | Post-generation: "Does this answer follow only from the provided context? YES/NO" — NLI-style LLM check, only on RAG agent |

---

## Phase-by-Phase Plan `[CHANGED]`

---

## Phase 1: Foundation, Abstractions & Docker `[CHANGED]` ✅

**Goal**: Project skeleton, `pyproject.toml` with pinned deps, Docker setup, base abstractions, LangGraph state.

### Key Changes from v1
- Docker + uv from day one (not Phase 8)
- Exact version pins resolved
- `MemorySaver` replaces custom memory classes
- Abstract base classes defined before any concrete agent

### File Structure
```
HrMind/
├── backend/
│   ├── __init__.py
│   ├── config.py               # Pydantic Settings, reads .env
│   ├── state.py                # GraphState TypedDict + AgentResult dataclass
│   ├── base/
│   │   ├── __init__.py
│   │   ├── agent.py            # BaseAgent ABC
│   │   ├── guardrail.py        # GuardrailStrategy Protocol + CompositeGuardrail
│   │   └── repository.py       # VectorRepository Protocol
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── context_builder.py  # Per-agent context slice builder
│   │   ├── entity_store.py     # spaCy entity extraction + lookup
│   │   └── summarizer.py       # Rolling summarization (every 10 turns)
│   └── utils/
│       ├── __init__.py
│       ├── logging.py          # structlog structured logging
│       └── token_counter.py    # tiktoken-based counter
├── pyproject.toml              # uv project file, pinned deps
├── uv.lock                     # locked dependency graph
├── .env.example
├── Dockerfile.backend          # multi-stage with uv
├── Dockerfile.frontend
├── docker-compose.yml          # backend + frontend + chromadb services
└── tests/
    ├── conftest.py
    └── test_phase1.py
```

### Version Pins (`pyproject.toml`)
```toml
[project]
name = "hrmind-backend"
requires-python = ">=3.11"

[project.dependencies]
langchain = ">=1.0.0,<2.0"
langchain-openai = ">=0.3.0"
langchain-community = ">=0.3.0"
langgraph = ">=0.2.0,<0.3"
chromadb = ">=0.6.0"          # HTTP client mode
sentence-transformers = ">=3.0.0"
rank-bm25 = ">=0.2.2"
spacy = ">=3.7.0"
aiosqlite = ">=0.20.0"
fastapi = ">=0.115.0"
uvicorn = {extras = ["standard"], version = ">=0.30.0"}
pydantic-settings = ">=2.5.0"
pytesseract = ">=0.3.13"
pdf2image = ">=1.17.0"
python-magic = ">=0.4.27"
ragas = ">=0.2.0"
structlog = ">=24.0.0"
tiktoken = ">=0.7.0"
sqlglot = ">=23.0.0"
streamlit = ">=1.40.0"
httpx = {extras = ["http2"], version = ">=0.27.0"}
plotly = ">=5.24.0"
```

### Optimized Dockerfile (Multi-Stage with uv) `[CHANGED]`
```dockerfile
# Dockerfile.backend
FROM ghcr.io/astral-sh/uv:python3.11-slim AS builder
WORKDIR /app
# Copy dependency files first (layer cache: only invalidated when deps change)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

FROM python:3.11-slim AS runtime
# Install system dependencies (Tesseract, Poppler) in one RUN layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# Copy only the built venv from builder — no uv in runtime image
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
# Copy source last (most frequently changed layer)
COPY backend/ ./backend/
EXPOSE 8000
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

```yaml
# docker-compose.yml
services:
  chromadb:
    image: chromadb/chroma:0.6.0
    volumes:
      - chroma_data:/chroma/chroma
    ports:
      - "8001:8000"
    environment:
      ALLOW_RESET: "true"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/heartbeat"]
      interval: 10s

  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    env_file: .env
    ports:
      - "8000:8000"
    depends_on:
      chromadb:
        condition: service_healthy
    volumes:
      - ./data:/app/data:ro          # HR docs read-only
      - uploads:/app/uploads

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "8501:8501"
    depends_on:
      - backend
    environment:
      BACKEND_URL: "http://backend:8000"

volumes:
  chroma_data:
  uploads:
```

### GraphState
```python
class GraphState(TypedDict):
    query: str
    session_id: str
    conversation_summary: str              # rolling compressed history
    recent_turns: list[dict]               # last 3 turns verbatim
    entity_store: dict[str, str]           # {"employee_name": "John", ...}
    plan: PlannerOutput | None             # Planner's agent array
    agent_results: Annotated[list[AgentResult], operator.add]  # reducer: append
    final_answer: str
    uploaded_file_path: str | None
    error: str | None
```

### Test
```bash
docker compose up -d chromadb
pytest tests/test_phase1.py -v
# Tests: GraphState creation, ContextBudget limits, entity extraction, CompositeGuardrail chain
```

---

## Phase 2: RAG Agent — Hybrid Search (No Cross-Encoder) `[CHANGED]` ✅

**Goal**: Production-grade async RAG with BM25 + dense hybrid search, citations, evals.

### Key Changes from v1
- `bge-small-en-v1.5` (33M params, ~5x faster) instead of `bge-large`
- No cross-encoder — hybrid RRF is fast enough and avoids N×M reranking cost
- ChromaDB in HTTP client mode (no file locking with multiple workers) `[CHANGED]`
- HuggingFace embedding calls wrapped in `run_in_executor` `[CHANGED]`

### Retrieval Architecture
```
Query
  │
  ├──▶ Dense: bge-small-en-v1.5 → ChromaDB HTTP → top-20 chunks
  │
  └──▶ Sparse: BM25Okapi (pre-built at startup) → top-20 chunks
            │
            ▼
    Reciprocal Rank Fusion (RRF k=60)  ← pure Python, ~0ms
            │
            ▼
    Top-8 chunks → injected into LLM prompt
            │
            ▼
    LLM generates answer WITH citations (chunk source + page)
```

### Execute-in-Code Pattern (No Tool Calls)
```python
async def run(self, state: GraphState) -> AgentResult:
    query = state["plan"].queries["rag"]

    # Step 1: Execute retrieval IN CODE (no LLM tool call)
    dense_results, sparse_results = await asyncio.gather(
        self.vector_repo.similarity_search(query, k=20),
        asyncio.get_event_loop().run_in_executor(None, self.bm25.get_top_n, query, 20)
    )
    chunks = reciprocal_rank_fusion(dense_results, sparse_results, k=60)[:8]

    # Step 2: Build prompt with retrieved context
    context = format_chunks(chunks)
    messages = self.context_builder.build_rag_messages(state, query, context)

    # Step 3: LLM sees only the final prompt — one call, no tools
    response = await self.llm.ainvoke(messages)
    return AgentResult(agent_name="rag", answer=response.content, sources=[c.metadata for c in chunks], ...)
```

### File Structure
```
backend/agents/rag_agent/
├── __init__.py
├── ingestion.py        # Async doc loader, chunker, embedder, ChromaDB uploader
├── retriever.py        # ChromaVectorRepository impl + BM25 index builder
├── hybrid_search.py    # RRF fusion function
├── rag_agent.py        # RAGAgent(BaseAgent) — the concrete agent
├── context_builder.py  # Builds agent-scoped prompt messages
├── guardrails.py       # TopicGuardrail + GroundingGuardrail
└── evals/
    ├── eval_runner.py  # RAGAS orchestrator — async
    ├── eval_dataset.py # 30 golden Q&A pairs (manually curated)
    └── metrics.py      # Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
```

### RAGAS Evals
```python
# 30 golden pairs across all 31 docs (1 per doc + a few cross-doc multi-hop)
eval_dataset = [
    {"question": "What is the probationary period policy?",
     "ground_truth": "..."},
    {"question": "What are the notice periods for different roles?",
     "ground_truth": "..."},
    ...
]
# Metrics: Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
# Eval endpoint: POST /api/eval/rag → returns JSON report
```

### Test
```bash
# Inside docker:
docker compose exec backend python -m backend.agents.rag_agent.ingestion
docker compose exec backend pytest tests/test_rag_agent.py -v
```

---

## Phase 3: SQL Agent — Custom, Schema-in-Prompt `[CHANGED]` ✅

**Goal**: NL → SQL → execute → table + NL. No LangChain toolkit. Schema injected in prompt.

### Key Design: Execute in Code, Explain with LLM `[CHANGED]`
```python
async def run(self, state: GraphState) -> AgentResult:
    query = state["plan"].queries["sql"]

    # Step 1: Guardrail check
    guard = await self.guardrails.check(query)
    if not guard.passed:
        return AgentResult(success=False, error=guard.reason, ...)

    # Step 2: LLM generates SQL (schema is in system prompt — no tool call)
    sql_messages = self.context_builder.build_sql_messages(state, query)
    # System prompt contains full DDL. LLM returns ONLY the SQL string.
    sql_response = await self.llm.ainvoke(sql_messages)
    sql_query = extract_sql(sql_response.content)  # parse from ```sql block

    # Step 3: Validate SQL (sqlglot — no LLM)
    guard = await ReadOnlySQLGuardrail().check(sql_query)
    if not guard.passed:
        return AgentResult(success=False, error="Non-SELECT query rejected", ...)

    # Step 4: Execute in Python (no LLM)
    rows, columns = await self.db.execute(sql_query)   # aiosqlite

    # Step 5: LLM explains the result (one final LLM call)
    explanation = await self._explain_result(query, sql_query, rows, columns)

    return AgentResult(
        answer=explanation,
        structured_data={"columns": columns, "rows": rows, "sql": sql_query},
        ...
    )
```

### SQL System Prompt (Schema Injection)
```
You are an HR data analyst. Given the user's question and the database schema below,
write a single valid SQLite SELECT query to answer it.
Return ONLY the SQL inside a ```sql block. No explanation.

SCHEMA:
CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    ...
);
CREATE TABLE leave_requests (...);
...
```

Since the schema is small (~20 columns across 5 tables), it fits comfortably in the system prompt at ~400 tokens. No schema discovery tool calls needed.

### File Structure
```
backend/agents/sql_agent/
├── __init__.py
├── database.py         # AsyncSQLiteDB — aiosqlite wrapper, parameterized queries only
├── schema.py           # Schema DDL as Python constants (single source of truth)
├── sql_agent.py        # SQLAgent(BaseAgent) — concrete agent
├── sql_parser.py       # extract_sql() — parse SQL from LLM markdown response
├── context_builder.py  # Builds SQL-scoped prompt (query + schema + last result)
├── formatter.py        # rows + columns → formatted table + NL explanation
├── guardrails.py       # ReadOnlySQLGuardrail (sqlglot) + RowLimitGuardrail
└── seed/
    └── seed_db.py      # Creates SQLite + seeds ~50 employees
```

### Database Schema (SQLite)
```sql
employees (id, name, department, role, hire_date, salary, manager_id, status, email)
departments (id, name, head_id, budget, location)
leave_requests (id, employee_id, leave_type, start_date, end_date, days, status, approved_by)
payroll (id, employee_id, month, year, gross, deductions, net, paid_on)
performance_reviews (id, employee_id, year, quarter, rating, reviewer_id, notes)
```

### Test
```bash
docker compose exec backend pytest tests/test_sql_agent.py -v
# Tests: SQL generation mock, sqlglot read-only enforcement, aiosqlite execution,
#        formatter output, blocked INSERT/UPDATE/DELETE, row limit cap
```

---

## Phase 4: Document Parser Agent ✅ COMPLETE

**Goal**: Text extraction from PDFs and DOCX files, LLM-based structured field extraction,
cross-agent integration (DocParser → SQL, DocParser → RAG), and a state-persistence mechanism
that prevents redundant re-parsing on follow-up questions.

**Status**: Fully implemented and tested. 35/35 unit tests passing.

---

### Key Decisions & Rationale

#### 1. Text-based extraction only (no OCR)
- **Decision**: Use `pdfplumber` for PDFs. No Tesseract/pytesseract.
- **Rationale**: HR documents (contracts, payslips, ID cards) are digitally created PDFs —
  they contain actual text in the PDF's internal structure. `pdfplumber` reads this directly.
  OCR introduces noise, requires system dependencies (Tesseract + Poppler), and is 10x slower.
  If a document fails text extraction (encrypted, scanned), a quality check catches it before
  any LLM call is made.
- **Fallback**: Quality assessment detects garbled/empty text and returns a clear user-facing
  error ("please upload a text-based PDF") — no silent failure, no OCR fallback.

#### 2. File type validation via magic bytes (not file extension)
- **Decision**: Read the first 8 bytes of every upload; reject anything that isn't `%PDF-` or
  `PK\x03\x04` (ZIP/OOXML header used by all `.docx` files).
- **Rationale**: File extensions are trivially spoofed. A renamed `.exe` with a `.pdf`
  extension passes extension checks but fails magic byte validation. This is a security
  requirement, not just a UX nicety.
- **Implementation**: `FileTypeGuardrail` in `guardrails.py`. Runs before any bytes are read
  into memory.

#### 3. Two LLM calls per document (type detection + field extraction)
- **Decision**: Split into two sequential LLM calls rather than one large universal schema.
- **Call 1** (`doc_type_detector.py`): First 500 chars → `DocType` enum (4 types + unknown).
  ~80 input tokens. Uses `.with_structured_output(DocTypeResponse)`.
- **Call 2** (`field_extractor.py`): First 4000 chars of full text → typed Pydantic schema
  matching the detected type. Uses `.with_structured_output(schema_class)`.
- **Rationale**: A focused schema of 4 fields is more accurate than a universal schema of 15+.
  Classification and extraction are different cognitive tasks — splitting them makes each
  LLM call narrow and reliable. Type detection is ~80 tokens; it's not worth merging for
  marginal accuracy gains.
- **Truncation**: Text is truncated to 4000 chars (not full document) for extraction.
  Key fields (names, salary, dates) appear in the first 2 pages. This cuts token cost by
  50–80% on long employment contracts without reducing extraction quality.

#### 4. Follow-up routing fix — `parsed_document` in GraphState
- **Problem**: After parsing an ID card in Turn 1 (DocParser → SQL for salary), a follow-up
  "How many leaves has he taken?" should route directly to SQL — not re-invoke DocParser
  on a file that may no longer exist (1-hour TTL).
- **Solution**: `parsed_document: dict | None` added to `GraphState`. Written by the Router
  after DocParser succeeds. Persists across all future turns via MemorySaver checkpointing.
- **Planner rule**: "If `parsed_document` is set AND `uploaded_file_path` is None → skip
  `doc_parser`. Use entity_store for context."
- **entity_store hydration**: `entity_mapper.py` maps extracted fields (e.g. `employee_name`,
  `employee_id`) into `entity_store` so the SQL agent can build `WHERE` clauses directly.
- **Turn 1**: Plan = `["doc_parser", "sql"]`. DocParser extracts → entity_store populated.
- **Turn 2**: Plan = `["sql"]`. SQL reads entity_store. Zero LLM calls for doc parsing.

#### 5. In-memory extraction cache keyed by SHA-256
- **Decision**: Cache `ExtractionResult` by SHA-256 hash of the raw file bytes.
- **Rationale**: If the same file is uploaded again (same content, regardless of filename),
  skip both LLM calls entirely. Zero cost, zero latency on cache hit.
- **Scope**: In-memory dict (`cache.py`), process-scoped singleton. Suitable for single-worker
  Uvicorn. Swappable with Redis without changing the public interface.
- **Key insight**: The cache survives the 1-hour file TTL. The file gets deleted, but the
  extracted result stays in memory until process restart.

#### 6. Parallel page extraction for multi-page PDFs
- **Decision**: Use `ThreadPoolExecutor` to extract text from PDF pages concurrently.
- **Rationale**: `pdfplumber`'s `page.extract_text()` is CPU-bound and page-independent.
  Running pages in parallel gives ~3–5x speedup on 10+ page documents.
- **Implementation**: Single-page documents skip the executor (no overhead).
  Multi-page: `ThreadPoolExecutor(max_workers=min(page_count, 4))`.

#### 7. Text quality assessment before any LLM call
- **Decision**: Check extracted text with 4 heuristic rules before spending tokens.
- **Rationale**: If pdfplumber returns garbage (encrypted file, scanned image, binary dump),
  sending it to the LLM wastes tokens and produces nonsense. Better to fail fast with a
  clear user message.
- **Four checks** (in `extractor.py → assess_text_quality()`):

| Check | Threshold | Catches |
|---|---|---|
| Minimum length | < 80 chars | Blank pages, empty/encrypted PDFs |
| Alphabetic ratio | < 35% alpha | Binary dumps, image-only PDFs |
| Vocabulary size | < 8 unique words | Trivially short content, form stubs |
| Character repetition | > 60% one char | Corrupted files, fill characters |

#### 8. Field validators on all Pydantic schemas
- **Decision**: Add `@field_validator` methods to every schema class.
- **Rationale**: The LLM may return plausible-looking but wrong values (e.g. a monthly salary
  as annual, notice period of 9999 days). Validators silently coerce out-of-range values to
  `None` — this is safer than keeping a hallucinated number downstream.
- **Key validators**:
  - `salary`: Must be 0 < salary ≤ 10,000,000 (annual)
  - `notice_period_days`: Must be 0–730 (up to 2 years)
  - `start_date`, `pay_period`: ISO 8601 format enforced; non-conforming dates → `None`
  - `contract_type`: Normalised to `"permanent"`, `"fixed-term"`, `"contractor"`
  - `employee_id`: Must be ≥ 2 chars; all-numeric in a text field → `None`
  - `pay amounts` (gross/net/deductions): Must be 0 ≤ value ≤ 1,000,000 monthly

#### 9. Completeness score
- **Decision**: Compute `completeness_score = non_null_fields / total_fields` after extraction.
- **Rationale**: The Combiner (Phase 5) needs to know if extraction was high-quality (score=1.0)
  or partial (score=0.2) to decide whether to surface a warning to the user.
- **Threshold**: If `completeness_score < 0.4`, the agent's answer includes:
  > ⚠️ Only partial information was extracted (X% of expected fields). Cross-check the values
  > before acting on them.
- **Stored in**: `ExtractionResult.completeness_score`, `AgentResult.metadata["completeness_score"]`,
  and `GraphState["parsed_document"]["completeness_score"]`.

#### 10. Retry logic with exponential backoff
- **Decision**: Retry both LLM calls up to 3 times with delays of 1s → 2s → 4s.
- **Rationale**: Rate limit errors, transient timeouts, and connection drops are common in
  production. Without retry, a single transient failure kills the entire pipeline.
- **Shared helper**: `_invoke_with_retry()` in `field_extractor.py` — imported by
  `doc_type_detector.py` to keep retry logic DRY.
- **After all retries fail**: Return empty schema (`completeness_score = 0.0`). Never raise
  from `run()`. Always return an `AgentResult`.

---

### Final Execution Flow

```
DocParserAgent.run(state)
│
├── [Step 1]  Get uploaded_file_path from GraphState
│             → fail if None ("No file was uploaded")
│
├── [Step 2]  FileSizeGuardrail  → reject if > 20MB
│             FileTypeGuardrail  → reject if magic bytes ≠ %PDF- or PK\x03\x04
│
├── [Step 3]  compute_file_hash(file_path)
│             → check extraction_cache
│             → CACHE HIT:  return cached AgentResult instantly (0 LLM calls)
│             → CACHE MISS: continue
│
├── [Step 4]  extract_text(file_path)
│             PDF:  _extract_pdf_sync() via ThreadPoolExecutor (parallel pages)
│             DOCX: _extract_docx_sync() via Docx2txtLoader
│             → (raw_text, page_count)
│
├── [Step 5]  assess_text_quality(raw_text)
│             → FAIL: return failure AgentResult with clear user message (0 LLM calls)
│             → PASS: continue
│
├── [Step 6]  DocTypeDetector.detect(raw_text[:500])
│             LLM Call #1 — ~80 tokens
│             Retry: up to 3 attempts (1s, 2s, 4s backoff)
│             → DocType enum: employment_contract | payslip | employee_id | offer_letter | unknown
│
├── [Step 7]  FieldExtractor.extract(doc_type, raw_text[:4000])
│             Selects schema from DOC_TYPE_SCHEMA_MAP[doc_type]
│             LLM Call #2 — truncated to 4000 chars (~1000 tokens)
│             Retry: up to 3 attempts (1s, 2s, 4s backoff)
│             → (typed Pydantic model, fields_dict, completeness_score)
│             Field validators run automatically on Pydantic model construction
│
├── [Step 8]  map_to_entity_store(doc_type, fields_dict)
│             → context summary for agent answer
│
├── [Step 9]  Assemble ExtractionResult + AgentResult
│             ExtractionResult: {doc_type, extracted_fields, page_count,
│                                source_file, char_count, completeness_score, cache_hit}
│
├── [Step 10] extraction_cache.set(file_hash, structured_data)
│             → cached for future identical uploads
│
└── [Step 11] Return AgentResult(success=True, structured_data=..., metadata=...)
              If completeness_score < 0.4 → answer includes ⚠️ warning
```

**Post-run (Router, Phase 5)**:
```python
state["parsed_document"]    = result.structured_data   # persists across turns
state["entity_store"]       |= map_to_entity_store(...)  # SQL context
state["uploaded_file_path"] = None                       # cleared — file processed
```

---

### File Structure (Final)

```
backend/agents/doc_parser_agent/
├── __init__.py              # Package init — exports DocParserAgent
├── doc_parser_agent.py      # DocParserAgent(BaseAgent) — main orchestrator (10 steps)
├── extractor.py             # Text extraction + assess_text_quality() + parallel pages
├── doc_type_detector.py     # LLM call #1 — DocType enum from first 500 chars (with retry)
├── field_extractor.py       # LLM call #2 — typed schema extraction, truncated, with retry
├── schemas.py               # 4 Pydantic schemas + validators + ExtractionResult + DocType
├── guardrails.py            # FileSizeGuardrail (20MB) + FileTypeGuardrail (magic bytes)
├── entity_mapper.py         # fields_dict → entity_store + context summary string
└── cache.py                 # SHA-256 keyed in-memory ExtractionCache (singleton)
```

**Modified files:**
- `backend/state.py` — `parsed_document: dict | None` added to `GraphState` and `make_initial_state()`
- `pyproject.toml` — `pdfplumber>=0.11.0` added (installs: pdfminer-six, pypdfium2, cryptography)

---

### LLM Call Summary

| Call | Module | Tokens (approx.) | Retries | Purpose |
|---|---|---|---|---|
| #1 Detection | `doc_type_detector.py` | ~80 in / ~5 out | 3 (1s/2s/4s) | Classify document type |
| #2 Extraction | `field_extractor.py` | ~1100 in / ~200 out | 3 (1s/2s/4s) | Extract structured fields |
| **Cache hit** | `cache.py` | **0** | — | Repeat upload → skip both calls |

Total LLM calls for a new document: **2**
Total LLM calls on cache hit: **0**
Estimated cost per new document (gpt-4o-mini): ~$0.0002

---

### Schemas & Document Types

| DocType | Schema Class | Fields | Key Validators |
|---|---|---|---|
| `employment_contract` | `EmploymentContractFields` | name, role, dept, start_date, salary, notice_period_days, contract_type | salary 0–10M, notice 0–730 days, ISO date |
| `payslip` | `PayslipFields` | employee_name, employee_id, pay_period, gross, net, deductions | amounts 0–1M, YYYY-MM format |
| `employee_id` | `EmployeeIDFields` | employee_name, employee_id, department, role | name ≥ 2 chars, not all-numeric |
| `offer_letter` | `OfferLetterFields` | candidate_name, role, salary, start_date, department | salary 0–10M, ISO date |

All fields are `str | None` or `float | None` — the LLM returns `null` for missing fields,
never a hallucinated value.

---

### Cross-Agent Use Cases (Implemented Patterns)

#### ID Card → SQL (primary use case)
```
Turn 1: Upload employee-id.pdf + "What is John's salary?"
  Plan: ["doc_parser", "sql"]
  DocParser → EmployeeIDFields(name="John Smith", id="EMP042", ...)
  Router writes: entity_store["employee_name"] = "John Smith"
                 entity_store["employee_id"]   = "EMP042"
                 parsed_document = {...}
                 uploaded_file_path = None
  SQL agent rewrites query: "SELECT salary WHERE name='John Smith' OR id='EMP042'"
  Combiner → "John Smith's salary is £65,000."

Turn 2: "How many leaves has he taken?" (no file)
  Planner: parsed_document ≠ None AND uploaded_file_path = None → skip doc_parser
  Plan: ["sql"]
  SQL reads entity_store → queries leave_requests for EMP042
  Combiner → "He has taken 12 days of leave this year."
```

#### Contract → RAG (compliance check)
```
Upload contract.pdf + "Does my notice period comply with policy?"
  Plan: ["doc_parser", "rag"]
  DocParser → EmploymentContractFields(notice_period_days=30, ...)
  Router builds RAG query: "30-day notice period compliance for senior roles"
  RAG → "Policy requires 60 days for Senior roles"
  Combiner → "Your contract specifies 30 days, but policy requires 60 days. Please consult HR."
```

---

### Test Coverage (35/35 passing)

| Test Class | Tests | What's Covered |
|---|---|---|
| `TestFileSizeGuardrail` | 4 | Size limit (< 1KB accepted, > 20MB rejected, boundary, missing file) |
| `TestFileTypeGuardrail` | 5 | Valid PDF, valid DOCX, spoofed PDF (PNG bytes), spoofed DOCX (EXE bytes), empty file |
| `TestTextExtraction` | 2 | pdfplumber (mocked), Docx2txtLoader (mocked) |
| `TestDocTypeDetector` | 5 | Contract, ID, payslip detection; empty text → UNKNOWN; LLM failure → UNKNOWN |
| `TestFieldExtractor` | 4 | EmployeeIDFields, ContractFields, all-null, retry failure (sleep mocked) |
| `TestEntityMapper` | 6 | ID mapping, contract mapping, payslip mapping, null fields, summary string, empty store |
| `TestDocParserAgentRun` | 5 | Happy path (full pipeline mocked), no file, size guardrail, type guardrail, empty text |
| `TestFollowUpRoutingContract` | 4 | Initial state=None, serializable dict, entity_store populated, routing decision logic |

```bash
pytest tests/test_doc_parser.py -v
# 35 passed in 15.34s
```

---

### Deferred / Out of Scope

- **OCR fallback for scanned PDFs**: Explicitly excluded. Clear error message returned instead.
- **Multi-language support**: Assumed English documents only.
- **Redis cache**: Deferred to Phase 8 (multi-worker). Current in-memory cache is sufficient
  for single-worker Uvicorn deployment.
- **File TTL cleanup job**: Designed (1-hour TTL in `/uploads`), implemented in Phase 8
  as a background scheduler task.
- **DOCX table extraction**: `Docx2txtLoader` does not extract table content well.
  If payslips in DOCX format have tabular gross/net data, this may miss fields.
  `pdfplumber` handles PDF tables correctly. DOCX table support is Phase 8 polish.



## Phase 5: Orchestration — Planner + Router + Combiner ✅ COMPLETE

**Goal**: Wire all agents into the LangGraph state machine with correct memory persistence and intelligent routing. 
**Status**: Fully implemented and tested.

### The Components

#### 1. Planner Node (`planner.py`)
- **Role**: Analyzes the user's query and current `GraphState` to decide which agents to invoke.
- **Implementation**: Uses `gpt-4o-mini` (or the configured `planner_model`) with `.with_structured_output(PlannerOutput)`.
- **Rules applied**:
  - If a file is uploaded, include `"doc_parser"`.
  - If a file was parsed in a previous turn (`parsed_document` exists) but no new file is uploaded, do **NOT** invoke `"doc_parser"`. Instead, the context is drawn from `entity_store`.
  - Sets `parallel=True` if the agents selected are completely independent.
  - Outputs a specific rewritten query for each agent in `plan.queries`.

#### 2. Router Node (`router.py`)
- **Role**: Pure Python execution block (no LLM). Executes the plan generated by the Planner.
- **Key Cases Handled**:
  - **Parallel Execution**: If `plan.parallel` is `True`, it uses `asyncio.gather(*tasks)` to run all selected agents concurrently.
  - **Sequential State Handoff**: If `plan.parallel` is `False` (e.g., DocParser runs before SQL), the Router loops through the agents sequentially. 
    - *Example*: DocParser extracts an employee ID from a PDF and returns `entity_store: {"employee_id": "123"}` in its metadata. The Router intercepts this output and injects it into the `current_state` *before* invoking the SQL agent, allowing the SQL agent to use `"employee_id": "123"` in its query.
  - **Memory Persistence**: Forwards updates like `parsed_document` and `entity_store` to the global `GraphState` so LangGraph's `MemorySaver` can persist them for the next conversational turn. It also clears `uploaded_file_path` so it isn't parsed again next turn.

#### 3. Combiner Node (`combiner.py`)
- **Role**: The final synthesizer. Reads all accumulated `agent_results` from the state and writes a cohesive final answer.
- **Key Cases Handled**:
  - **Conflict & Error Resolution**: Checks if any agent failed or reported partial data.
  - *Example*: If the DocParser returned a `completeness_score < 0.4`, the Combiner detects this and appends a standardized warning to the final text: `⚠️ Only partial information was extracted from the document. Please cross-check the values.`
  - **Tone Enforcement**: Instructed to merge the responses naturally without exposing the internal agent names (e.g., avoids saying "The SQL agent found...").

#### 4. Agent Factory (`factory.py`)
- **Role**: Centralizes dependency injection and agent instantiation.
- **Decision**: Avoids re-initializing heavy connections (like ChromaDB or BM25 indexes) on every run. Currently, it instantiates the agents dynamically based on the configuration defined in `backend/config.py` (e.g., `agent_model`, `combiner_model`).

#### 5. LangGraph Assembly (`graph.py`)
- **Role**: Defines the `StateGraph` topology.
- **Flow**: `Planner` → `Router` → `Combiner` → `END`.
- **Checkpointing**: Uses `MemorySaver` so the state (like the `entity_store` built by the DocParser) persists across multiple user messages in the same thread.

### File Structure (Final)
```
backend/orchestration/
├── __init__.py
├── planner.py      # planner_node: LLM → PlannerOutput
├── router.py       # router_node: Pure Python, handles handoffs & asyncio.gather
├── combiner.py     # combiner_node: LLM → final answer & warnings
├── factory.py      # AgentFactory: dependency injection & instantiation
└── graph.py        # StateGraph assembly + MemorySaver
```

### Test Coverage
```bash
uv run pytest tests/test_orchestrator.py -v
```
- **Sequential Execution Test**: Validated that when DocParser and SQL run sequentially, the Router successfully passes `entity_store` from the DocParser's output into the global state.
- **Combiner Warnings Test**: Validated that a low `completeness_score` from an agent triggers the proper UI warning string in the Combiner's final answer.

---

## Phase 6: FastAPI Backend — Async + SSE Streaming `[CHANGED]` ✅

**Goal**: REST API with SSE streaming (not WebSocket), file upload, eval endpoint.

### Streaming: SSE Not WebSocket `[CHANGED]`

SSE (Server-Sent Events) is simpler to consume from Streamlit via `httpx` and works over HTTP/1.1. LangGraph's `astream_events` integrates naturally with SSE:

```python
@router.get("/api/chat/stream")
async def chat_stream(query: str, session_id: str):
    async def event_generator():
        config = {"configurable": {"thread_id": session_id}}
        async for event in graph.astream_events(state, config=config, version="v2"):
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"].content
                yield f"data: {json.dumps({'token': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

### Endpoints
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/chat` | Full response (non-streaming) |
| GET | `/api/chat/stream` | SSE token stream |
| POST | `/api/upload` | Multipart file upload → returns `file_id` |
| POST | `/api/eval/rag` | Trigger RAGAS eval → returns report JSON |
| GET | `/api/health` | Service + ChromaDB health check |
| GET | `/api/sessions/{id}` | Conversation history |
| DELETE | `/api/sessions/{id}` | Clear session + reset checkpointer |

### File Structure
```
backend/api/
├── __init__.py
├── main.py             # FastAPI app, lifespan (startup: load embeddings, BM25 index)
├── routes/
│   ├── chat.py         # POST /chat, GET /chat/stream (SSE)
│   ├── upload.py       # POST /upload — file validation + storage
│   ├── eval.py         # POST /eval/rag
│   ├── sessions.py     # GET/DELETE /sessions/{id}
│   └── health.py       # GET /health
├── dependencies.py     # FastAPI DI: get_graph(), get_agent_registry()
├── middleware/
│   ├── cors.py
│   └── rate_limiter.py  # slowapi
└── schemas/
    ├── chat.py          # ChatRequest, ChatResponse Pydantic models
    └── upload.py        # UploadResponse
```

### Startup (Lifespan Event) — Load Once, Not Per-Request
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load HF embedding model once at startup (blocking — before server accepts requests)
    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    # Build BM25 index once at startup
    bm25_index = build_bm25_index(load_all_chunks())
    # Connect to ChromaDB HTTP client
    chroma_client = chromadb.AsyncHttpClient(host=settings.CHROMA_HOST, port=settings.CHROMA_PORT)
    app.state.embedder = embedder
    app.state.bm25 = bm25_index
    app.state.chroma = chroma_client
    yield
    # Cleanup
```

### Test
```bash
docker compose exec backend pytest tests/test_api.py -v
# Uses httpx.AsyncClient with TestClient
# Tests: all endpoints, SSE stream parsing, file upload, rate limiting
```

---

## Phase 7: Vite + React Frontend ✅

**Goal**: Beautiful, modern Single Page Application consuming FastAPI via `fetch`. SSE for streaming chat, custom routing, and premium glassmorphic UI.

### Architecture
- **Framework**: React via Vite (`npm run dev`)
- **Styling**: Vanilla CSS (`index.css`) with premium dark mode, glassmorphism, and gradients. No Tailwind.
- **Routing**: `react-router-dom`

### Pages
```
frontend/
├── src/
│   ├── App.jsx                 # react-router setup (Home, Login, Chat routes)
│   ├── index.css               # Premium design system (variables, animations, glassmorphism)
│   ├── main.jsx                # React entry
│   ├── pages/
│   │   ├── Home.jsx            # Stunning landing page with call-to-actions
│   │   ├── Auth.jsx            # Split Login/Signup glassmorphism card
│   │   └── Chat.jsx            # Multi-agent chat interface with sidebar & dynamic feed
│   └── components/
│       ├── ChatBubble.jsx      # Agent/User message bubble
│       ├── Sidebar.jsx         # Chat history and settings
│       └── ...                 # Other reusable UI components
```

### SSE Consumption in React
```javascript
// React chat with SSE streaming via Fetch API
const startStream = async (query) => {
  const response = await fetch(`${BACKEND_URL}/api/chat/stream?query=${query}`);
  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  
  let fullText = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    const chunk = decoder.decode(value);
    // Parse SSE lines (data: {...})
    const lines = chunk.split('\n').filter(line => line.startsWith('data: '));
    for (const line of lines) {
      if (line === 'data: [DONE]') return;
      const data = JSON.parse(line.substring(6));
      fullText += data.token;
      setStreamingMessage(fullText); // Update React state
    }
  }
};
```

### Test
```bash
cd frontend && npm run dev
# Visit http://localhost:5173 to visually test the landing page, auth, and chat UI.
```

---

## Phase 8: E2E Tests, Upload Cleanup, Polish (IN PROGRESS)

**Goal**: Full conversation flow tests, scheduled upload cleanup, final Docker polish.

### Status
- **SQLite Data Seeded**: Mock HR data (50 employees) generated. ✅
- **Models Loaded**: LLMs properly dynamically load keys and base URLs from `.env`. ✅
- **Next Up**: Full stack test validation (Docker/E2E).

### E2E Test Scenarios
1. **RAG only**: "What is the maternity leave policy?" → single agent
2. **SQL only**: "How many employees are in the Engineering department?" → single agent
3. **Doc Parser only**: Upload contract PDF → extract fields
4. **Multi-agent sequential**: Upload contract → check notice period against policy → DocParser + RAG
5. **Multi-agent parallel**: "What's our alcohol policy and how many employees have had leave this year?" → RAG + SQL parallel
6. **Follow-up memory**: "What about drug testing?" (after alcohol policy question) → correct context
7. **Context bleed test**: SQL query then RAG query → RAG prompt must NOT contain SQL schema
8. **Guardrail test**: "DROP TABLE employees" → SQL guardrail blocks, returns error message
9. **Partial failure test**: One agent returns error → Combiner surfaces partial result gracefully

### Upload Cleanup (Background Task)
```python
# FastAPI lifespan: register cleanup job
async def cleanup_old_uploads():
    """Delete files in /uploads older than 1 hour every 30 minutes."""
    ...
```

### Final Docker Compose Validation
```bash
docker compose up --build        # full stack
docker compose exec backend pytest tests/ -v --tb=short  # all tests
```

## Phase 9: Authentication & Authorization ✅

**Goal**: Implement secure user registration, login, and JWT-based session management to protect the chat and upload endpoints.

### Architecture
- **Backend Auth**: FastAPI OAuth2 with Password (and hashing via `passlib[bcrypt]`), JWT issuance via `python-jose[cryptography]`.
- **Database**: New `users` table in the SQLite database to store `email` and `password_hash`.
- **Frontend State**: React Context (`AuthContext`) to manage user session and JWT storage in `localStorage`.
- **Protected Routes**: React Router will redirect unauthenticated users away from `/chat` to `/login`.
- **API Protection**: The `/api/chat/stream` and `/api/upload` endpoints will require a valid `Bearer` token.

### Proposed Endpoints
- `POST /api/auth/register`: Create a new user account.
- `POST /api/auth/login`: Authenticate and receive a JWT token.

---

## Technology Stack (Final)

| Layer | Technology | Version |
|---|---|---|
| **LLM** | `ChatOpenAI` — Deepseek via NVIDIA NIM for all | — |
| **Orchestration** | LangGraph | `>=0.2.0,<0.3` |
| **LLM Framework** | LangChain | `>=1.0.0,<2.0` |
| **Embeddings** | `BAAI/bge-large-en-v1.5` | SentenceTransformers >=3.0 |
| **Sparse Retrieval** | BM25Okapi | rank_bm25 >=0.2.2 |
| **Vector Store** | ChromaDB **HTTP client mode** (Docker service) | >=0.6.0 |
| **RAG Evals** | RAGAS | >=0.2.0 |
| **OCR** | Pytesseract + pdf2image (in Docker, Tesseract pre-installed) | — |
| **Database** | SQLite via `aiosqlite` | >=0.20.0 |
| **SQL Safety** | `sqlglot` for statement type parsing | >=23.0.0 |
| **File Safety** | `python-magic` (magic bytes, not extension) | >=0.4.27 |
| **Backend** | FastAPI + Uvicorn (single worker — ChromaDB safe) | >=0.115.0 |
| **Streaming** | SSE via `StreamingResponse` + `astream_events` | — |
| **Frontend** | React (Vite) + vanilla CSS | — |
| **Async SQL** | aiosqlite | >=0.20.0 |
| **Memory** | LangGraph `MemorySaver` (MemoryCheckpointer) | built-in |
| **Logging** | structlog | >=24.0.0 |
| **Package Mgr** | `uv` | latest |
| **Containers** | Docker multi-stage + docker-compose | — |
| **Testing** | pytest + pytest-asyncio + httpx | — |
| **Auth** (Phase 9) | passlib[bcrypt] + python-jose | latest |

---

## Phase 10: Performance Optimization & UI Polish ✅ COMPLETE

**Goal**: Optimize memory context window to prevent LLM hallucination and out-of-scope errors on long conversations. Polish the frontend to render rich text and animations.

### Problems Encountered & Fixes Implemented

#### 1. Context Overload & "Out of Scope" Hallucinations
- **Problem**: The original architecture used `recent_turns` and `conversation_summary`, accumulating vast amounts of text over multiple conversational turns. When querying dense domains (like the entire "Alcohol Policy"), the prompt size ballooned. This overwhelmed the LLM, causing degraded intent classification and triggering false-positive "Out-of-Scope" fallback errors for valid follow-up questions.
- **Fix**: Replaced the infinitely growing context model with a strictly bounded 1-turn sliding window (`previous_turn`). The system drops older conversation history entirely, returning the LLM to a highly predictable and token-efficient state on every turn.

#### 2. Pronoun Resolution in Follow-ups
- **Problem**: In a 1-turn memory system, if a user asks "What is the alcohol policy?" followed by "What are its aims?", dropping the older history strips the Planner of the core entity ("alcohol policy"), breaking its ability to route the query.
- **Fix**: Injected the `previous_turn` (the exact preceding question and answer) directly into the Planner's system prompt. This allows the LLM to seamlessly resolve pronouns (mapping "its" to "alcohol policy") and rewrite the user's query into a standalone sentence *before* invoking downstream agents.

#### 3. State Bloat from Parallel Agent Reducers
- **Problem**: To support parallel execution branches, LangGraph requires the `agent_results` state field to use an `operator.add` reducer. Because of this, the array would grow infinitely over hundreds of conversational turns, leading to heavy checkpoint sizes and serialization lag.
- **Fix**: Designed a "memory-safe" slice inside the Combiner node. By using `agent_results[-num_agents:]`, the Combiner isolates strictly the N results generated in the *current* turn, entirely ignoring historical agent traces left in the background state ledger.

#### 4. Unnecessary Routing & Combiner Latency
- **Problem**: If 0 agents were selected (an out-of-scope query) or only 1 agent was selected, the LangGraph flow pointlessly pushed the state through the Router and Combiner, wasting time and resources.
- **Fix**: Configured dynamic graph branching. If 0 agents are needed, the Planner provides the `final_answer` directly. If 1 agent is needed, the Router records the `previous_turn` on exit and bypasses the Combiner. The Combiner is strictly reserved for synthesis when `num_agents > 1`.

#### 5. ChatPromptTemplate KeyError
- **Problem**: Encountered a server crash `KeyError: "Input to ChatPromptTemplate is missing variables {'context_str'}"` during stream generation because the `previous_turn` context string wasn't properly passed through the LangChain invocation.
- **Fix**: Updated the `chain.invoke()` payload dictionary in `planner.py` to correctly map the required `context_str` parameter.

#### 6. Silent React Frontend Crashes (Blank Page)
- **Problem**: After migrating to `react-markdown` v9 for rich text rendering, the frontend crashed with a completely blank screen because v9 introduced a major breaking change that removed the `className` prop entirely from the `<ReactMarkdown>` component.
- **Fix**: Implemented a React `ErrorBoundary` component to catch the silent rendering crash and surface the stack trace to the UI. Fixed the underlying bug by stripping the prop and wrapping the markdown component in a native `<div className="markdown-body">`.

#### 7. Horizontal Scrollbars on Long SQL Queries
- **Problem**: The SQL agent frequently generates complex single-line `SELECT` statements. Inside markdown `<pre>` code blocks, these rigid lines extended far beyond the chat container's width, introducing an ugly horizontal scrollbar (a "sidebar").
- **Fix**: Updated the vanilla CSS for `.markdown-body pre` and `.markdown-body code` to aggressively enforce `white-space: pre-wrap; word-break: break-word`, ensuring long queries gracefully wrap onto the next line to fit the screen.

### Dead Code Cleanup
- Deleted `backend/agents/rag_agent/guardrails.py` and stripped legacy guardrail initializations. 
- Completely deleted the `backend/memory/` directory (`summarizer.py`, `context_builder.py`, `entity_store.py`) as they were rendered obsolete by the 1-turn architecture. 
- Removed legacy token budgets and limits from `config.py`.

### UI Polish
- **React Markdown**: Replaced raw string output with properly formatted markdown rendering (tables, bold, lists, code blocks).
- **Claude-Like Animations**: Added a pulsating CSS `@keyframes` animation (`.bot-pulsate` with `cubic-bezier(0.4, 0, 0.2, 1)`) to the Bot icon during the `isTyping` state, closely mimicking Claude's elegant loading state.

---

## Resolved Questions

- **Models**: Configured all agents and orchestrators to use the models defined in `.env` (Deepseek via NVIDIA NIM).
- **SQLite Seed Data**: Generated 50 realistic employees covering Engineering, Sales, Support, HR, Marketing, and Finance.
- **Database Relationship**: `users` table is independent of `employees` table. Users act as HR Administrators.
- **Auth State Management**: Used native React Context `AuthContext` instead of Zustand to minimize external dependencies.
