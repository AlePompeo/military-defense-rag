# Military & Defense RAG

A Retrieval-Augmented Generation (RAG) system for military and defense doctrine, designed to run entirely **offline** using a local LLM via [LM Studio](https://lmstudio.ai).

The pipeline mirrors the **OODA loop** (Observe → Orient → Decide → Act):

| Phase | Module | Role |
|-------|--------|------|
| Observe | `loader.py` | Download, parse, and hierarchically chunk source documents |
| Orient | `store.py` / `embedder.py` | Embed (cached) and index chunks into ChromaDB |
| Decide | `store.py` | Hybrid dense+BM25 retrieval, RRF fusion, cross-encoder reranking |
| Act | `rag.py` | Generate a grounded, cited answer via LM Studio, with a citation sanity check |

---

## Document Corpus

26 military and defense sources are indexed, including:

- US Army Field Manuals (FM 3-0, FM 5-0, FM 2-0, FM 3-06, FM 3-12, FM 3-09, FM 3-24)
- Army Techniques Publications (ATP 2-01.3 IPB, ATP 3-60 Targeting)
- US Army Doctrine Publications (ADP 3-0)
- NATO Allied Joint Publications (AJP-3.3, AJP-3.20)
- Classic military theory: *On War* (Clausewitz), *The Art of War* (Sun Tzu), MCDP 1
- Operations research, game theory, Lanchester warfare models
- Pathfinding, GIS-based military route planning, logistics modeling
- CIA Gateway Process analysis, asymmetric warfighting research

---

## Requirements

- Python 3.10+
- [LM Studio](https://lmstudio.ai) with a model loaded and the local server running on port 1234
- Tesseract OCR and Poppler — only needed as a fallback for scanned/image-only PDFs; most of the 26 sources have a text layer and never touch this path.

```bash
pip install -r requirements.txt
```

The whole dependency stack (`chromadb`, `fastembed`'s ONNX runtime, `rank_bm25`, `diskcache`, etc.) ships prebuilt wheels for Linux, Windows, and macOS — no compiler needed on any platform.

### Linux

```bash
sudo dnf install tesseract poppler-utils   # or: apt install tesseract-ocr poppler-utils
bash install_lmstudio.sh                    # optional Fedora helper for LM Studio itself
```

`install_lmstudio.sh` installs the `lms` CLI, adds it to `PATH`, optionally downloads a recommended model, and starts the local server.

### Windows

No script needed — the Python code itself is cross-platform and runs identically once dependencies are installed:

1. Install [LM Studio for Windows](https://lmstudio.ai), load a model, and start the local server (**Local Server** tab → **Start Server**, default `http://localhost:1234`, same as Linux).
2. `pip install -r requirements.txt` (PowerShell or cmd).
3. Only if you'll ingest scanned/image-only PDFs: install [Tesseract for Windows](https://github.com/UB-Mannheim/tesseract/wiki) and [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases), then either add both `bin` folders to `PATH`, or point at them directly without touching `PATH`:
   ```powershell
   $env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
   $env:POPPLER_PATH  = "C:\poppler\Library\bin"
   ```
4. `python main.py ingest`, then `python main.py query` — identical commands to Linux.

`file://` URLs (for manually-downloaded documents, see `PROGRESS.md`) work with Windows-style paths too, e.g. `file:///C:/docs/manual.pdf`.

---

## Usage

### 1. Ingest documents

Downloads and indexes all 26 sources. Already-indexed sources are skipped on re-runs.

```bash
python main.py ingest
```

### 2. Query

Interactive REPL:

```bash
python main.py query
```

One-shot question:

```bash
python main.py query -q "What is the OODA loop and how does it apply to modern warfare?"
```

Skip reranking (faster, slightly less precise — useful on constrained RAM):

```bash
python main.py query -q "..." --no-rerank
```

### 3. Optional, higher-accuracy but LLM-call-heavy features

Both are off by default because they trade local-LLM latency for accuracy — see the tradeoffs below before enabling.

**HyDE** (query-side, one extra LLM call per uncached query): generates a hypothetical doctrine-style passage from the question and uses it — instead of the raw question — to steer dense retrieval, which often matches indexed passage style better than a short question does.

```bash
python main.py query -q "..." --hyde
```

**Contextual retrieval** (ingest-side, one LLM call per chunk, cached and resumable): generates a short LLM summary situating each chunk within its parent section before embedding, on top of the static title prefix that's already applied to every chunk. This is **not recommended on CPU-only/constrained hardware** — for this 26-source corpus (a few thousand chunks) it's roughly multi-day on the 3.8GB-RAM CPU-only setup this project was built on, versus roughly 1–5 hours on a dedicated consumer GPU (estimate, not measured — actual time depends on GPU tier and the model loaded in LM Studio):

```bash
python main.py ingest --contextualize
```

If interrupted, re-running is safe: already-summarized chunks are read from `CONTEXTUALIZE_CACHE_PATH` instead of re-calling the LLM.

---

## Configuration

All settings are in `config.py` and can be overridden via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `LM_STUDIO_URL` | `http://localhost:1234/v1` | LM Studio OpenAI-compatible endpoint |
| `LM_STUDIO_MODEL` | `local-model` | Model identifier shown in LM Studio |
| `CHROMA_DB_PATH` | `./data/chroma` | Persistent ChromaDB storage path |
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | fastembed ONNX embedding model (768-dim) |
| `EMBEDDING_CACHE_PATH` | `./data/embed_cache` | Disk cache for computed embeddings |
| `RERANKER_MODEL` | `Xenova/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker (lazily loaded) |
| `USE_RERANKER` | `1` | Set to `0` to disable reranking by default |
| `QUERY_CACHE_PATH` | `./data/query_cache` | Disk cache for retrieval results |
| `QUERY_CACHE_TTL` | `3600` | Query cache expiry, in seconds |
| `HNSW_EF_CONSTRUCTION` / `HNSW_EF_SEARCH` / `HNSW_M` | `200` / `100` / `16` | ChromaDB HNSW index tuning |
| `MMR_LAMBDA` | `0.7` | Relevance vs. diversity trade-off in result reordering (`1.0` = pure relevance) |
| `TESSERACT_CMD` | *(unset — use PATH)* | Path to `tesseract`/`tesseract.exe`, if not on `PATH` (mainly Windows) |
| `POPPLER_PATH` | *(unset — use PATH)* | Path to Poppler's `bin` folder, if not on `PATH` (mainly Windows) |
| `USE_HYDE` | `0` | Set to `1` to enable HyDE by default (equivalent to always passing `--hyde`) |
| `HYDE_CACHE_PATH` | `./data/hyde_cache` | Disk cache for HyDE-generated hypothetical passages |
| `CONTEXTUALIZE_CACHE_PATH` | `./data/contextualize_cache` | Disk cache for per-chunk LLM contextual summaries |

All caches live under `data/`, which is gitignored — safe to delete anytime to force recomputation (e.g. after changing `EMBEDDING_MODEL`).

---

## Architecture

```
main.py            CLI entry point (ingest / query commands)
loader.py          Download → parse PDF/text → hierarchical parent/child chunking
embedder.py        fastembed (ONNX-only) embeddings + cross-encoder reranker, both disk-cached
contextualizer.py  Optional (--contextualize): per-chunk LLM contextual summaries, cached
store.py           ChromaDB vector store: hybrid dense+BM25 retrieval, RRF fusion, reranking, query cache
rag.py             Retrieve top-K chunks → build prompt → call LM Studio → citation grounding check; optional HyDE
config.py          Runtime settings via env vars
install_lmstudio.sh  Fedora setup script for LM Studio CLI
```

### Pipeline detail

- **Chunking** (`loader.py`) — text is split on paragraph/sentence boundaries (not blind character cuts), then grouped into ~2700-char **parent** blocks. Each parent is further split into overlapping ~900-char **child** chunks. Children are what gets embedded and matched; on a hit, the parent's full text is what's sent to the LLM (small-to-big retrieval — precise matching, richer context). Each child is also prefixed with its document's human-readable title before embedding (`main.py`'s `TITLES` map) — a zero-cost form of *contextual retrieval*: a generic sentence about "friction" reads very differently once the index knows it's from *On War* vs. a modern field manual. Optionally (`--contextualize`), an LLM-generated situating sentence is added on top of the title (see below).
- **Embedding** (`embedder.py`) — `BAAI/bge-base-en-v1.5` via `fastembed` (768-dim, ONNX, ~210MB, no PyTorch/GPU). BGE is asymmetric: documents and queries get different instruction prefixes (`embed` vs. `query_embed`). Every embedding is cached on disk keyed by `(model, text)`, so re-ingesting or repeating a query never recomputes it.
- **Storage** (`store.py`) — ChromaDB collection `military_docs_v2` (renamed from the old MiniLM-based collection since the vector space/dimensionality changed), with HNSW tuned for cosine similarity (BGE output is L2-normalized).
- **Retrieval** (`store.py`) — the query is first expanded with any recognized military acronyms (`OODA`, `IPB`, `FM`, `ATP`, `C2`, `ISR`... see `_ACRONYMS`), improving both keyword and embedding matches on doctrine-heavy jargon. Dense (bge) + sparse (BM25 via `rank_bm25`) candidate lists are then combined with Reciprocal Rank Fusion, reordered by a cross-encoder reranker (`Xenova/ms-marco-MiniLM-L-6-v2`, lazily loaded — costs nothing if `--no-rerank` is passed) with a confidence cutoff that drops candidates the reranker is confident are irrelevant, then diversified via **MMR** (Maximal Marginal Relevance, `MMR_LAMBDA`) so near-duplicate passages from overlapping doctrine texts don't crowd out other relevant context. Final results are deduplicated by parent. Optionally (`--hyde`), the dense search step is steered by a hypothetical LLM-generated passage instead of the raw question (see below); BM25 and reranking always use the real question.
- **Query caching** (`store.py`) — the full retrieval result (post-fusion/rerank/MMR) is cached on disk, keyed by query text + settings + a corpus version counter that auto-bumps on every `ingest`, so cached results can never go stale after new documents are added.
- **Generation** (`rag.py`) — after the LLM answers, a lightweight grounding check scans for `[n]` citations outside the actual `1..top_k` range and appends a warning if the model cited a passage that didn't exist (a cheap hallucination tripwire).

### Optional LLM-call-heavy features (off by default)

- **HyDE** (`rag.py: _hyde_passage`) — Hypothetical Document Embeddings (Gao et al., 2022). Before dense retrieval, asks the local LLM to write a short hypothetical doctrine-style passage answering the question, then embeds *that* instead of the raw question for the dense-search step only (BM25 and the reranker still see the real question — a cross-encoder is trained on real query/passage pairs, and a fabricated passage would confuse it). Helps most on short or vaguely-phrased questions, whose wording tends to differ more from indexed doctrine prose than a doctrine-shaped hypothetical answer does. Costs one extra local LLM call per uncached query; the generated passage is itself disk-cached (`HYDE_CACHE_PATH`) so repeating a question doesn't re-trigger it.
- **Contextual retrieval** (`contextualizer.py`) — adapted from Anthropic's [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval), which prepends an LLM-generated summary of each chunk's place in the document before embedding, reported to reduce retrieval failures substantially. The original technique conditions on the *whole source document*, relying on a large context window; most models run locally in LM Studio have far smaller windows (a 100+ page field manual won't fit), so this adaptation conditions each summary on the chunk's **parent block** (~2700 chars) instead — the same unit already used for small-to-big retrieval. Costs one local LLM call per chunk at ingest time, cached (`CONTEXTUALIZE_CACHE_PATH`) and resumable if interrupted.

### On search speed

At this corpus's scale (26 sources, a few thousand chunks), retrieval itself is not the bottleneck: warm model loads measure ~0.7s (embedder) + ~0.2s (reranker), BM25 index (re)build is ~0.25s for 6,000 chunks, and per-query dense+BM25 search is single-digit milliseconds. The dominant cost by far is local LLM generation in LM Studio, which this codebase doesn't control. `--no-rerank` and the query cache are the two real speed levers available; further "search speed" work (e.g. persisting the BM25 index, quantized/binary vector search) wouldn't move the needle here and was left out to avoid complexity without payoff — it would matter at a much larger corpus scale.

---

## Answer quality

Answers are grounded exclusively in the indexed documents. The system prompt instructs the model to:

- Cite every passage used with a bracketed number, e.g. `[1]`
- Explicitly state when the context is insufficient rather than speculate

Temperature is set to `0.2` for factual consistency.
