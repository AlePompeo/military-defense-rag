# Military & Defense RAG

A Retrieval-Augmented Generation (RAG) system for military and defense doctrine, designed to run entirely **offline** using a local LLM via [LM Studio](https://lmstudio.ai).

The pipeline mirrors the **OODA loop** (Observe → Orient → Decide → Act):

| Phase | Module | Role |
|-------|--------|------|
| Observe | `loader.py` | Download, parse (text/tables/OCR/optional image captions), and section-aware hierarchically chunk source documents |
| Orient | `store.py` / `embedder.py` | Embed (cached) and index chunks into ChromaDB |
| Decide | `store.py` | Hybrid dense+BM25 retrieval, RRF fusion, cross-encoder reranking |
| Act | `rag.py` | Generate a grounded, cited answer (with a Sources list) via LM Studio, with a citation sanity check |

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
- A vision-capable model in LM Studio (e.g. LLaVA, Qwen2-VL, MiniCPM-V) — only needed for the opt-in `--describe-images` flag; not needed for anything else in this project.

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

**Semantic chunking** (ingest-side, no extra LLM call — reuses the document embedder, cheap on CPU): splits parent blocks into child chunks at embedding-similarity boundaries instead of a fixed-size sliding window, so a chunk boundary tends to land where the topic actually shifts rather than at a fixed character count.

```bash
python main.py ingest --semantic-chunk
```

Off by default so chunk boundaries stay stable/reproducible unless explicitly opted into, and because the similarity threshold (`SEMANTIC_CHUNK_SIMILARITY_THRESHOLD`) is an untuned starting point — see `PROGRESS.md`.

**Image/chart captioning** (ingest-side, one LLM call per extracted image, cached): extracts embedded images/charts from each PDF page and captions them via LM Studio's vision endpoint (axis labels, legend, key data points for charts), indexing the caption alongside the text so it's retrievable and citable like any other passage.

```bash
python main.py ingest --describe-images
```

**Not recommended on this project's reference hardware** — requires a vision-capable model (typically 3B+ params), heavier than the Phi-3 Mini text model recommended below for 3.8GB RAM; see "On image/chart understanding" below before enabling.

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
| `GENERATION_CACHE_PATH` | `./data/generation_cache` | Disk cache for final LLM answers (cache-augmented generation, see below) |
| `SEMANTIC_CHUNK_SIMILARITY_THRESHOLD` | `0.5` | Cosine similarity below which `--semantic-chunk` starts a new chunk |
| `VISION_MODEL` | same as `LM_STUDIO_MODEL` | Vision-capable model identifier for `--describe-images` (usually a *different* model than the text one — set explicitly) |
| `VISION_CACHE_PATH` | `./data/vision_cache` | Disk cache for LLM-generated image/chart captions |

All caches live under `data/`, which is gitignored — safe to delete anytime to force recomputation (e.g. after changing `EMBEDDING_MODEL`).

---

## Architecture

```
main.py            CLI entry point (ingest / query commands)
loader.py          Download → parse PDF/text (+ tables, + optional image captions) →
                   section-aware hierarchical parent/child chunking (fixed-window or semantic)
embedder.py        fastembed (ONNX-only) embeddings + cross-encoder reranker, both disk-cached
contextualizer.py  Optional (--contextualize): per-chunk LLM contextual summaries, cached
vision.py          Optional (--describe-images): per-image LLM captions via LM Studio's vision endpoint, cached
store.py           ChromaDB vector store: hybrid dense+BM25 retrieval, RRF fusion, reranking, query cache
rag.py             Retrieve top-K chunks → build prompt → call LM Studio → citation grounding check +
                   Sources list; optional HyDE; final answer disk-cached
config.py          Runtime settings via env vars
install_lmstudio.sh  Fedora setup script for LM Studio CLI
```

### Pipeline detail

- **Chunking** (`loader.py`) — text is split on paragraph/sentence boundaries (not blind character cuts), then grouped into ~2700-char **parent** blocks. Each parent is further split into overlapping ~900-char **child** chunks (fixed-size sliding window by default, or embedding-similarity boundaries with `--semantic-chunk`, see below). Children are what gets embedded and matched; on a hit, the parent's full text is what's sent to the LLM (small-to-big retrieval — precise matching, richer context). Each child is also prefixed with its document's human-readable title before embedding (`main.py`'s `TITLES` map) — a zero-cost form of *contextual retrieval*: a generic sentence about "friction" reads very differently once the index knows it's from *On War* vs. a modern field manual. **Section-aware**: a cheap heuristic (no LLM call) detects "Chapter N"/"Section N" markers and short ALL-CAPS lines as section headings — a line-scanning pre-pass, since PDF text extraction commonly loses the blank-line paragraph breaks that would otherwise isolate a heading — and prefixes every chunk under that heading with its section title, on top of the document title. Optionally (`--contextualize`), an LLM-generated situating sentence is added on top of both (see below). **Table/image-aware**: extracted tables and (optional) image captions are treated as atomic blocks that are never split mid-content, regardless of size.
- **Table extraction** (`loader.py`) — `pdfplumber` detects ruled tables per page and renders them as markdown (for embedding/citation) plus the raw rows as JSON (`table_json` chunk metadata, for structured access without re-parsing markdown). This is additive to the existing `pypdf` text path, not a replacement: the flattened inline text still contributes surrounding-paragraph context, while the markdown block gives a clean, citable representation of the actual data. Known limitation: because tables are appended after a page's full text rather than interleaved at their exact original position, a table's `section` tag reflects whichever heading was current at the end of that page, which can be off by one section on pages that both end a table and start a new chapter.
- **Embedding** (`embedder.py`) — `BAAI/bge-base-en-v1.5` via `fastembed` (768-dim, ONNX, ~210MB, no PyTorch/GPU). BGE is asymmetric: documents and queries get different instruction prefixes (`embed` vs. `query_embed`). Every embedding is cached on disk keyed by `(model, text)`, so re-ingesting or repeating a query never recomputes it.
- **Storage** (`store.py`) — ChromaDB collection `military_docs_v2` (renamed from the old MiniLM-based collection since the vector space/dimensionality changed), with HNSW tuned for cosine similarity (BGE output is L2-normalized).
- **Retrieval** (`store.py`) — the query is first expanded with any recognized military acronyms (`OODA`, `IPB`, `FM`, `ATP`, `C2`, `ISR`... see `_ACRONYMS`), improving both keyword and embedding matches on doctrine-heavy jargon. Dense (bge) + sparse (BM25 via `rank_bm25`) candidate lists are then combined with Reciprocal Rank Fusion, reordered by a cross-encoder reranker (`Xenova/ms-marco-MiniLM-L-6-v2`, lazily loaded — costs nothing if `--no-rerank` is passed) with a confidence cutoff that drops candidates the reranker is confident are irrelevant, then diversified via **MMR** (Maximal Marginal Relevance, `MMR_LAMBDA`) so near-duplicate passages from overlapping doctrine texts don't crowd out other relevant context. Final results are deduplicated by parent. Optionally (`--hyde`), the dense search step is steered by a hypothetical LLM-generated passage instead of the raw question (see below); BM25 and reranking always use the real question.
- **Query caching** (`store.py`) — the full retrieval result (post-fusion/rerank/MMR) is cached on disk, keyed by query text + settings + a corpus version counter that auto-bumps on every `ingest`, so cached results can never go stale after new documents are added.
- **Generation** (`rag.py`) — after the LLM answers, a lightweight grounding check scans for `[n]` citations outside the actual `1..top_k` range and appends a warning if the model cited a passage that didn't exist (a cheap hallucination tripwire). The final answer is cached keyed on `(model, retrieved context, question)`: a repeated or identical query returns instantly from disk, skipping the LM Studio call entirely — the dominant latency cost per query at this corpus scale (see "On cache-augmented generation" below). The key includes the literal retrieved context text, so it never needs manual invalidation — different retrieval settings or a changed corpus naturally produce a different key. A **Sources** list is then appended (not cached — cheap to rebuild, always fresh) mapping every `[n]` back to its source document, section (if any), and URL, using metadata already tracked per chunk.

### Optional LLM-call-heavy features (off by default)

- **HyDE** (`rag.py: _hyde_passage`) — Hypothetical Document Embeddings (Gao et al., 2022). Before dense retrieval, asks the local LLM to write a short hypothetical doctrine-style passage answering the question, then embeds *that* instead of the raw question for the dense-search step only (BM25 and the reranker still see the real question — a cross-encoder is trained on real query/passage pairs, and a fabricated passage would confuse it). Helps most on short or vaguely-phrased questions, whose wording tends to differ more from indexed doctrine prose than a doctrine-shaped hypothetical answer does. Costs one extra local LLM call per uncached query; the generated passage is itself disk-cached (`HYDE_CACHE_PATH`) so repeating a question doesn't re-trigger it.
- **Contextual retrieval** (`contextualizer.py`) — adapted from Anthropic's [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval), which prepends an LLM-generated summary of each chunk's place in the document before embedding, reported to reduce retrieval failures substantially. The original technique conditions on the *whole source document*, relying on a large context window; most models run locally in LM Studio have far smaller windows (a 100+ page field manual won't fit), so this adaptation conditions each summary on the chunk's **parent block** (~2700 chars) instead — the same unit already used for small-to-big retrieval. Costs one local LLM call per chunk at ingest time, cached (`CONTEXTUALIZE_CACHE_PATH`) and resumable if interrupted.
- **Image/chart captioning** (`vision.py`) — extracts embedded images from each PDF page (`pdfplumber`, cropped from a rendered page image) and sends each to LM Studio's vision-capable chat completions endpoint (`image_url` content parts), asking for a description that calls out chart/graph axis labels, legend entries, and the key trend or data shown. The caption is indexed as its own atomic `[IMAGE pN#i]` chunk, retrievable and citable like any text passage. Costs one local LLM call per extracted image at ingest time, cached (`VISION_CACHE_PATH`) and resumable if interrupted — but see "On image/chart understanding" below before enabling.

### On cache-augmented generation (CAG)

"Textbook" CAG (Chan et al., 2024/2025) preloads an entire corpus into the model's context so a
KV-cache can be reused across queries, skipping retrieval altogether. That isn't viable here: this
project targets small local models (e.g. Phi-3 Mini, 4k-token context — see `PROGRESS.md`) against
a 26-source, multi-thousand-chunk corpus that doesn't come close to fitting in a 4k window, let
alone leaving room for a question and answer. Retrieval isn't optional at this corpus-to-context
ratio.

What's implemented instead is the response-level equivalent: the final LLM answer is cached (see
"Generation" above), so an identical or repeated question — a common pattern in interactive/FAQ-
style use — skips the LM Studio call entirely, which is the actual bottleneck (see "On search
speed" below). This complements the embedding, retrieval, HyDE, and contextualization caches
already in place, rather than duplicating any of them.

### On image/chart understanding

`--describe-images` is implemented (`vision.py`, image extraction in `loader.py`) but comes with
the same kind of hardware caveat as CAG above: it needs a **vision-capable** model loaded in LM
Studio, and those start around 3B parameters even for lightweight ones (LLaVA, Qwen2-VL,
MiniCPM-V) — meaningfully heavier than the Phi-3 Mini *text* model this project recommends for
3.8GB-RAM hardware, which likely can't load a vision model at all alongside everything else running.
The feature is real and does the useful thing (chart axis labels, legend entries, and key data
points get indexed as text, so a chart becomes as searchable/citable as a paragraph) — enable it
only if you have more RAM/a GPU than this project's reference setup, or run the vision model on a
separate machine/instance than LM Studio's main text model.

### On section detection and table placement (known limitations)

Both are cheap, heuristic, no-LLM-call techniques, so they trade some precision for being free:

- **Heading detection** (`loader.py: _looks_like_heading`) is a regex/heuristic, not layout-aware —
  it can misfire on short ALL-CAPS boilerplate that isn't really a heading (e.g. a
  "FOR OFFICIAL USE ONLY" banner line), silently treating it as a (harmless but meaningless)
  section title. It can also miss real headings that don't match the "Chapter/Section N" or
  short-ALL-CAPS pattern (e.g. title-case headings). No LLM call, so essentially free either way.
- **Table section tagging** — see the "Table extraction" bullet above: a table's `section` metadata
  reflects the heading active at the *end* of its page's text stream, not its true visual position,
  since tables are appended after a page's full text rather than interleaved at their exact
  original position. The table's actual *content* (markdown + `table_json`) is unaffected — only
  the section label can be off by one on pages that both end a table and start a new chapter.

### On search speed

At this corpus's scale (26 sources, a few thousand chunks), retrieval itself is not the bottleneck: warm model loads measure ~0.7s (embedder) + ~0.2s (reranker), BM25 index (re)build is ~0.25s for 6,000 chunks, and per-query dense+BM25 search is single-digit milliseconds. The dominant cost by far is local LLM generation in LM Studio, which this codebase doesn't control. `--no-rerank` and the query cache are the two real speed levers available; further "search speed" work (e.g. persisting the BM25 index, quantized/binary vector search) wouldn't move the needle here and was left out to avoid complexity without payoff — it would matter at a much larger corpus scale.

---

## Answer quality

Answers are grounded exclusively in the indexed documents. The system prompt instructs the model to:

- Cite every passage used with a bracketed number, e.g. `[1]`
- Explicitly state when the context is insufficient rather than speculate

Temperature is set to `0.2` for factual consistency. Every answer ends with a **Sources** list
resolving each `[n]` to its source document, section (if detected), and URL.
