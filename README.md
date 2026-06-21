# Military & Defense RAG

A Retrieval-Augmented Generation (RAG) system for military and defense doctrine, designed to run entirely **offline** using a local LLM via [LM Studio](https://lmstudio.ai).

The pipeline mirrors the **OODA loop** (Observe → Orient → Decide → Act):

| Phase | Module | Role |
|-------|--------|------|
| Observe | `loader.py` | Download, parse, and chunk source documents |
| Orient | `store.py` | Embed and index chunks into ChromaDB |
| Decide | `rag.py` | Retrieve the most relevant passages for a query |
| Act | `rag.py` | Generate a grounded, cited answer via LM Studio |

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
- Tesseract OCR (for scanned PDFs): `sudo dnf install tesseract` / `sudo apt install tesseract-ocr`
- Poppler (for `pdf2image`): `sudo dnf install poppler-utils` / `sudo apt install poppler-utils`

```bash
pip install -r requirements.txt
```

### Quick LM Studio setup (Fedora)

```bash
bash install_lmstudio.sh
```

The script installs the `lms` CLI, adds it to `PATH`, optionally downloads a recommended model (Llama 3.1 8B or Mistral 7B), and starts the local server.

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

---

## Configuration

All settings are in `config.py` and can be overridden via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `LM_STUDIO_URL` | `http://localhost:1234/v1` | LM Studio OpenAI-compatible endpoint |
| `LM_STUDIO_MODEL` | `local-model` | Model identifier shown in LM Studio |
| `CHROMA_DB_PATH` | `./data/chroma` | Persistent ChromaDB storage path |

---

## Architecture

```
main.py          CLI entry point (ingest / query commands)
loader.py        Download → parse PDF/text → chunk with overlap
store.py         ChromaDB vector store + all-MiniLM-L6-v2 embeddings (ONNX, no GPU needed)
rag.py           Retrieve top-K chunks → build prompt → call LM Studio
config.py        Runtime settings via env vars
install_lmstudio.sh  Fedora setup script for LM Studio CLI
```

Embeddings use ChromaDB's built-in `DefaultEmbeddingFunction` (`all-MiniLM-L6-v2` via ONNX) — no PyTorch or GPU required for the retrieval stage.

---

## Answer quality

Answers are grounded exclusively in the indexed documents. The system prompt instructs the model to:

- Cite every passage used with a bracketed number, e.g. `[1]`
- Explicitly state when the context is insufficient rather than speculate

Temperature is set to `0.2` for factual consistency.
