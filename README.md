# Traditional RAG vs Page Index RAG — Comparison Notebook

A self-contained Google Colab notebook that implements and benchmarks two RAG pipelines side by side: **Traditional Vector RAG** and **Page Index RAG** (Vectorless RAG). Every pipeline component is written from scratch inline — no external repos or pre-built frameworks required.

**Author:** Ammar Nasir

---

## What This Notebook Does

It walks through the full lifecycle of both pipelines — indexing, retrieval, generation, and evaluation — on the same document and queries, so you can observe exactly where each approach succeeds and fails.

```
TRADITIONAL RAG               PAGE INDEX RAG
────────────────              ─────────────────────
Document                      Document
   ↓                             ↓
Split into fixed chunks       LLM reads pages
   ↓                             ↓
Embed each chunk              Build TOC tree
   ↓                             ↓
Store in vector DB            TOC stored in memory

User query                    User query
   ↓                             ↓
Embed query                   LLM traverses tree
   ↓                             ↓
Cosine similarity search      Fetch exact source pages
   ↓                             ↓
Top-K chunks → LLM            Source pages → LLM
   ↓                             ↓
Answer                        Answer
```

---

## The Core Problem Being Investigated

Traditional Vector RAG has three well-known failure modes this notebook demonstrates empirically:

**Context Split Problem** — fixed-size chunking slices text at character boundaries, not semantic boundaries. A concept that spans two chunks loses coherence in retrieval.

**Cross-Reference Problem** — legal docs, technical manuals, and textbooks frequently reference content on distant pages. Vector similarity cannot bridge that gap; it retrieves only what is keyword-adjacent to the query.

**Query Formulation Dependency** — retrieval quality is entirely determined by how well the user's words match the document's vocabulary. A vague or high-level query can miss highly relevant content.

Page Index RAG addresses all three by replacing embeddings with LLM structural reasoning over a hierarchical Table of Contents tree.

---

## Notebook Structure

| Cell | Description |
|------|-------------|
| 1 | Install dependencies (`openai`, `anthropic`, `pypdf`) |
| 2 | API key setup (NVIDIA / OpenAI / Anthropic) |
| 3 | Global config — model, chunk size, overlap, Top-K |
| 4 | Shared utilities — `LLMClient`, `EmbeddingClient`, `Timer`, `QueryResult` |
| 5 | Synthetic AI textbook document (embedded, no upload needed) |
| 6 | Traditional RAG: `TextChunker` + `InMemoryVectorStore` |
| 7 | Traditional RAG: Full pipeline class |
| 8 | Page Index RAG: `TOCNode` data structure |
| 9 | Page Index RAG: `TOCTreeBuilder` (LLM-based indexing) |
| 10 | Page Index RAG: `TreeTraversalRetriever` |
| 11 | Page Index RAG: Full pipeline class |
| 12 | Index both pipelines |
| 13 | Query both and print side-by-side comparison |
| 14 | Experiment A — chunk boundary / context split demo |
| 15 | Experiment B — vague vs precise query |
| 16 | Experiment C — cross-reference query spanning two chapters |
| 17 | Batch comparison across 4 queries with latency/token table |
| 18 | Bring your own PDF |
| 19–20 | Custom evaluation scorecard (faithfulness, relevancy, precision, recall) |

---

## Key Components

### Traditional RAG
- `TextChunker` — fixed-size sliding window with configurable overlap
- `InMemoryVectorStore` — pure Python cosine similarity, no external vector DB
- `TraditionalRAGPipeline` — end-to-end index + query

### Page Index RAG
- `TOCNode` — tree node with `node_id`, `title`, `summary`, `start_page`, `end_page`, `children`, `tags`
- `TOCTreeBuilder` — batched LLM calls to extract hierarchical structure; outputs a JSON-serializable tree
- `TreeTraversalRetriever` — LLM reads the compact TOC, selects relevant `node_id`s, fetches exact source pages
- `PageIndexRAGPipeline` — end-to-end index + query with tree save/load to skip re-indexing

---

## Quickstart

1. Open in Google Colab
2. Set your API key in **Cell 2** (NVIDIA NIM is used by default; OpenAI and Anthropic are also supported)
3. Run all cells top to bottom
4. Modify `QUERY` in Cell 13 to test your own questions
5. Upload your own PDF in Cell 18

No folder structure, no cloning, no config files — everything runs from a single notebook.

---

## Configuration

All tunable parameters are in **Cell 3**:

```python
PROVIDER         = 'nvidia'
LLM_MODEL        = 'meta/llama-3.3-70b-instruct'
EMBEDDING_MODEL  = 'nvidia/nv-embedqa-e5-v5'

CHUNK_SIZE       = 800    # chars per chunk
OVERLAP          = 100    # overlap between chunks
TOP_K            = 5      # chunks retrieved per query

CHARS_PER_PAGE   = 2000   # simulated page size for Page Index
BATCH_SIZE       = 3      # pages sent to LLM per indexing call
```

---

## Observed Results

From the batch evaluation run on a Vectorless RAG explanation document:

| Metric | Traditional RAG | Page Index RAG | Winner |
|--------|----------------|---------------|--------|
| Faithfulness | 0.600 | 0.900 | Page Index |
| Answer Relevancy | 0.800 | 1.000 | Page Index |
| Context Precision | 0.700 | 0.800 | Page Index |
| Context Recall | 0.600 | 0.700 | Page Index |

Page Index retrieved 3 semantically targeted nodes vs Traditional RAG's 5 partially irrelevant chunks — using roughly half the context tokens for better answers.

**Known failure case for Page Index:** if the TOC tree misses or mislabels a section during indexing, all future queries targeting that section will fail. One bad node = permanent blind spot.

---

## When to Use Which

| | Traditional RAG | Page Index RAG |
|--|----------------|---------------|
| **Use when** | Large unstructured corpora, cost-sensitive, varied document types | Books, legal docs, technical manuals, structured PDFs |
| **Strengths** | Fast lookup, low per-query cost, works on anything | Handles vague queries, respects structure, cross-references |
| **Weaknesses** | Context splits, keyword dependency, missed cross-refs | Expensive indexing, slower queries, fails on unstructured docs |

---

## Dependencies

```
openai
anthropic
pypdf
```

Install via Cell 1:
```bash
pip install openai anthropic pypdf -q
```

---

## License

MIT
