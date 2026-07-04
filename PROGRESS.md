# Military & Defense RAG — Progress

## Purpose
Build a RAG system for military and defense documents, running against a local LLM via LM Studio.

## Architecture
- **loader.py** — Download + parse (PDF/TXT) + chunk documents
- **store.py** — ChromaDB vector store with sentence-transformers embeddings
- **rag.py** — Retrieval + LM Studio generation pipeline
- **main.py** — CLI (`ingest` / `query` subcommands)
- **config.py** — Runtime parameters (URL, model, chunk size, top-k)

## Steps

| # | Step | Status |
|---|------|--------|
| 1 | Project setup (`requirements.txt`, `config.py`) | ✅ Done |
| 2 | Document loader (`loader.py`) | ✅ Done |
| 3 | Vector store (`store.py`) | ✅ Done |
| 4 | RAG pipeline (`rag.py`) | ✅ Done |
| 5 | CLI entry point (`main.py`) | ✅ Done |
| 6 | Ingestion run (all 26 sources) | ⬜ Pending — run `python main.py ingest` |
| 7 | End-to-end query test | ⬜ Pending — requires LM Studio running |

## Documents (26 sources)
All URLs listed in CLAUDE.md §7. Mix of PDFs (FMs, AJPs, ATPs, papers) and plain TXT (Gutenberg classics).

## How to Use
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start LM Studio, load a model, enable the local server

# 3. Ingest all documents (one-time, takes several minutes)
python main.py ingest

# 4. Query interactively
python main.py query

# 4b. One-shot query
python main.py query -q "What is the OODA loop and how does it apply to urban operations?"
```

## Notes
- LM Studio must be running at `http://localhost:1234` (configurable via env var `LM_STUDIO_URL`)
- First ingest downloads all 26 documents and builds the local ChromaDB vector index
- Vector index is persisted in `./data/chroma` — subsequent runs skip already-indexed sources
- Failed document downloads are logged and skipped; partial ingestion is safe to re-run

## Advanced pipeline upgrade (2026-07-04)
Upgraded every stage per user request, keeping the 3.8GB-RAM constraint in mind (chose
fastembed's quantized ONNX models over full sentence-transformers/torch):
- **Chunking**: paragraph/sentence-aware hierarchical chunking (small child chunks for
  retrieval, larger parent blocks sent to the LLM — small-to-big retrieval), replacing the
  old blind fixed-char sliding window.
- **Embedding**: `BAAI/bge-base-en-v1.5` via `fastembed` (768-dim, ~210MB, ONNX-only, no
  torch), replacing the old 384-dim `all-MiniLM-L6-v2`. Disk-cached by (model, text).
- **Storage**: new Chroma collection `military_docs_v2` (old `military_docs`/384-dim data
  left untouched in `data/chroma` — different vector space, can't be reused in place).
  HNSW tuned for cosine space.
- **Retrieval**: hybrid dense + BM25 search fused via Reciprocal Rank Fusion, then
  reranked by a lightweight cross-encoder (`Xenova/ms-marco-MiniLM-L-6-v2`, lazily loaded,
  toggle with `query --no-rerank`).
- **Caching**: query results (post-fusion/rerank) cached on disk with a corpus-version key
  that auto-invalidates on every `ingest`.
- **Generation**: post-hoc citation grounding check flags `[n]` references outside the
  actual retrieved range.
- **Action required**: re-run `python main.py ingest` — the new 768-dim collection starts empty.

## Accuracy + cross-platform pass (2026-07-04, same session)
- **Contextual retrieval (cheap version)**: each chunk is now prefixed with its document's
  human-readable title (`main.py: TITLES`) before embedding/indexing, so generic passages
  (e.g. "friction in war") get disambiguated by source without paying for an LLM call per
  chunk at ingest time.
- **Domain acronym expansion**: queries containing known military acronyms (OODA, IPB, FM,
  ATP, ADP, MCDP, AJP, C2, ISR, EW, ROE) get expanded before dense/BM25 search
  (`store.py: _ACRONYMS`).
- **MMR diversification**: final results are reordered via Maximal Marginal Relevance
  (`MMR_LAMBDA`, default 0.7) so near-duplicate passages from overlapping doctrine texts
  don't crowd out other relevant context.
- **Adaptive reranker cutoff**: when reranking is on, candidates the cross-encoder is
  confident (sigmoid prob < 0.05) are irrelevant get dropped, instead of always forcing
  exactly `top_k` results regardless of quality (always keeps at least the single best match).
- **Speed**: benchmarked — retrieval itself isn't the bottleneck at this corpus scale (warm
  model loads ~0.7s+0.2s, BM25 rebuild ~0.25s/6k chunks, per-query search single-digit ms).
  LM Studio generation dominates; didn't add speed-only complexity (e.g. persisted BM25,
  quantized vectors) since it wouldn't pay off here.
- **Cross-platform fix**: `loader.py`'s `file://` handling used `Path(urlparse(url).path)`
  directly, which mis-parses Windows drive letters (`file:///C:/x.pdf` → path `\C:\x.pdf`,
  wrong). Fixed via `urllib.request.url2pathname`. Added `TESSERACT_CMD` / `POPPLER_PATH` env
  vars so Windows users can point at OCR tools without editing `PATH`. Verified every
  dependency (new and old) ships prebuilt Windows wheels — no compiler needed.
- **Implemented as opt-in flags (same session, after discussing GPU-vs-CPU tradeoffs)**:
  - `python main.py query --hyde` — HyDE (`rag.py: _hyde_passage`): generates a hypothetical
    doctrine-style passage via the local LLM and uses it (cached) to steer dense search only;
    BM25/reranking still use the real question. One extra LLM call per uncached query.
  - `python main.py ingest --contextualize` — per-chunk LLM contextual summaries
    (`contextualizer.py`), adapted from Anthropic's Contextual Retrieval to condition on the
    **parent block** rather than the whole source document (most local models' context
    windows can't fit a 100+ page FM). One LLM call per chunk, cached/resumable. Explicitly
    **not recommended on CPU-only/constrained hardware** (multi-day estimate for this corpus)
    — reasonable on a dedicated GPU (~1-5h estimate, untested). Off by default since it's
    unsuitable for typical/common hardware, with a clear warning in `--help`.


##lm studio installation: 
  LM Studio is distributed as an AppImage on Linux. Download it from the official site:                                                     
                                                                                                                                            
  ▎ https://lmstudio.ai → Download → Linux                                                                                                  
                                                                                                                                            
  Then set it up:                                                                                                                           
                  
  # 1. Move to a predictable location
  mkdir -p ~/apps                                                                                                                           
  mv ~/Downloads/LM-Studio-*.AppImage ~/apps/lmstudio.AppImage
                                                                                                                                            
  # 2. Make executable
  chmod +x ~/apps/lmstudio.AppImage                                                                                                         
                  
  # 3. If SELinux blocks it (common on Fedora)                                                                                              
  chcon -t bin_t ~/apps/lmstudio.AppImage
                                                                                                                                            
  # 4. Launch     
  ~/apps/lmstudio.AppImage

  If it fails with a FUSE error despite fuse-libs being installed, use the extract-and-run fallback:                                        
  
  ~/apps/lmstudio.AppImage --appimage-extract-and-run                                                                                       
                                                                                                                                            
  ---
  After LM Studio is open:                                                                                                                  
                          
  1. Go to the Discover tab → search for "Phi-3 Mini" (Microsoft) — it's the most RAM-efficient useful model (~2.4 GB)
  2. Download it                                                                                                                            
  3. Go to Local Server tab (left sidebar) → click Start Server
  4. Default address: http://localhost:1234                                                                                                 
                                                                                                                                            
  Then your RAG will work as-is. To verify the server is up:                                                                                
                                                                                                                                            
  curl http://localhost:1234/v1/models                                                                                                      
                  
  ---
  Model recommendation given your 3.8 GB RAM:
                                                                                                                                            
  ┌─────────────────┬───────────┬────────────────┐
  │      Model      │ Size (Q4) │  Fits in RAM?  │                                                                                          
  ├─────────────────┼───────────┼────────────────┤
  │ Phi-3 Mini 3.8B │ ~2.4 GB   │ Yes (barely)   │                                                                                          
  ├─────────────────┼───────────┼────────────────┤
  │ Gemma 2B        │ ~1.5 GB   │ Yes            │                                                                                          
  ├─────────────────┼───────────┼────────────────┤
  │ Mistral 7B      │ ~4.1 GB   │ No (swap only) │                                                                                          
  └─────────────────┴───────────┴────────────────┘                                                                                          
   
  Set the model name in your RAG before querying:                                                                                           
                  
  export LM_STUDIO_MODEL="phi-3-mini-4k-instruct"                                                                                           
  python main.py query                                                                                                                      
                                                                                                                                            
  The exact model ID string is shown in LM Studio's Local Server tab once a model is loaded.  
  
  
 ## error to fix 
    New error analysis
  
  Good news first: lanchester_warfare_models and military_ops_research_1994 are gone from the failure list — the browser-like User-Agent
  fixed the DTIC 403. Those 2 sources are now indexed.

  Remaining 7 failures, root causes confirmed:

  Source: cia_gateway_process
  Old error: 0 chunks
  New error: Redirect loop
  Root cause: Browser UA → 302 to homepage (loop)
  Fix: Fixed: plain UA set per-source via headers option
  ────────────────────────────────────────
  Source: on_war_clausewitz
  Old error: SSL error
  New error: 404
  Root cause: Gutenberg geo-blocked in Italy (court order)
  Fix: Manual download required
  ────────────────────────────────────────
  Source: art_of_war_sun_tzu
  Old error: SSL error
  New error: 404
  Root cause: Same
  Fix: Manual download required
  ────────────────────────────────────────
  Source: atp_3_60_targeting
  Old error: empty file
  New error: not a PDF
  Root cause: irp.fas.org serving JS challenge HTML
  Fix: Manual download required
  ────────────────────────────────────────
  Source: fm_3_06_urban
  Old error: empty file
  New error: not a PDF
  Root cause: Same
  Fix: Manual download required
  ────────────────────────────────────────
  Source: fm_3_12_cyber_ew
  Old error: empty file
  New error: not a PDF
  Root cause: Same
  Fix: Manual download required
  ────────────────────────────────────────
  Source: fm_3_24_insurgencies
  Old error: empty file
  New error: not a PDF
  Root cause: Same
  Fix: Manual download required

  For the 6 sources that need manual download:
  1. Open each URL in your browser (browser handles geo-blocks and JS challenges naturally)
  2. Save the file to ./data/local/
  3. Replace the URL in DOCUMENTS with "file:///absolute/path/to/file.txt" (or .pdf)

