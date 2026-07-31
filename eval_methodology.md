# RAG Evaluation — Methodology & Metrics

## How the Q&A Dataset Was Created

> [!WARNING]
> **Important limitation to understand before trusting any eval scores.**

The 33 questions and ground truth answers in `eval_dataset.py` were written **based on document filenames and general UK HR law knowledge** — not by actually reading the documents.

For example:
- I knew a file named `Maternity-Policy.docx` would contain maternity leave information
- I knew UK statutory maternity leave is 52 weeks (26 OML + 26 AML) from UK employment law knowledge
- So I wrote the ground truth answer from that knowledge

**What this means:**
- Questions are well-formed and realistic ✅
- Ground truth answers *may* diverge from what the actual documents say ⚠️
- If your `Maternity-Policy.docx` says 30 weeks instead of 52, the eval will penalise the RAG for correctly quoting the document

**The right fix (before running evals):**
For each question, the ground truth should be extracted from the actual document. The eval runner can help with this — when you run it, you can review `eval_results.json` and compare RAG answers vs. ground truths, then manually correct the ground truths.

---

## RAGAS Metrics Explained

RAGAS evaluates four dimensions of RAG quality. All four require LLM calls.

### 1. Faithfulness (`0.0 – 1.0`, higher = better)
> **"Does the answer contain only facts present in the retrieved context?"**

- Detects **hallucination** — facts in the answer that can't be traced to the retrieved chunks
- Method: LLM decomposes the answer into atomic claims, then checks each claim against the context
- Example failure: RAG answers "maternity leave is 26 weeks" but context says 52 weeks → score drops

### 2. Answer Relevancy (`0.0 – 1.0`, higher = better)
> **"Is the answer actually addressing the question asked?"**

- Detects **off-topic or incomplete** answers
- Method: LLM generates N reverse-questions from the answer, then measures similarity to the original question
- Example failure: User asks about probationary period; RAG answers about notice periods → low score

### 3. Context Recall (`0.0 – 1.0`, higher = better)
> **"Does the retrieved context contain enough information to answer the question?"**

- Measures **retrieval completeness** — did we find the right chunks?
- Method: Each sentence in the ground truth is checked against retrieved context for attributability
- Example failure: Ground truth mentions 3 key policy points; retrieved context only covers 1 → score ~0.33
- **This is the metric most affected by retrieval quality (BM25 + dense search)**

### 4. Context Precision (`0.0 – 1.0`, higher = better)
> **"How much of the retrieved context was actually relevant to the answer?"**

- Measures **retrieval noise** — are we injecting irrelevant chunks?
- Method: Each retrieved chunk is classified as relevant or not; relevant chunks ranked higher = better score
- Example failure: Retrieve 8 chunks; only 2 are about maternity leave, 6 are about unrelated topics → 0.25

---

## Metric → Component Mapping

| Metric | What it measures | What to fix if low |
|---|---|---|
| **Faithfulness** | LLM synthesis quality | System prompt, grounding guardrail |
| **Answer Relevancy** | Query understanding | Planner query rewriting, system prompt |
| **Context Recall** | Retrieval completeness | chunk_size, top_k, BM25 + dense weights |
| **Context Precision** | Retrieval noise | RRF k value, final_top_k, chunk overlap |

---

## What the Eval Runner Will Do

1. For each of the 33 Q&A pairs:
   - Run full hybrid retrieval (embed → ChromaDB → BM25 → RRF)
   - Generate an answer via the LLM (one call per question)
   - Record: question, generated answer, retrieved contexts, ground truth
2. Feed all samples to RAGAS
3. Save results to `eval_results/ragas_results_<timestamp>.json`
4. Print a summary table of all four metric scores

**You will run this.** The script is ready at `backend/agents/rag_agent/evals/eval_runner.py`.
