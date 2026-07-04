"""
Document downloader, parser, and chunker.

Observe phase of OODA: pull raw intelligence from source URLs,
normalize it into clean text, then split into retrieval-sized chunks.
"""

import io
import os
import re
import warnings
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname
import requests
import pytesseract
from pdf2image import convert_from_bytes
from pypdf import PdfReader
from typing import List, Dict


TIMEOUT = 120  # seconds; some military PDFs are large

# Optional Windows-specific overrides — on Linux both tools are normally
# already on PATH via the system package manager (tesseract-ocr / poppler-utils).
# On Windows they're separate installers, so point at them explicitly if needed.
_TESSERACT_CMD = os.getenv("TESSERACT_CMD")
_POPPLER_PATH = os.getenv("POPPLER_PATH") or None
if _TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = _TESSERACT_CMD

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) "
        "Gecko/20100101 Firefox/125.0"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def _read_local(url: str) -> bytes:
    # url2pathname (not a raw Path() on urlparse().path) is required for
    # Windows file:// URIs: "file:///C:/docs/x.pdf" parses to "/C:/docs/x.pdf",
    # which Path() would treat as a folder literally named "C:" under the
    # current drive's root rather than as drive C:.
    path = Path(url2pathname(urlparse(url).path))
    if not path.exists():
        raise FileNotFoundError(f"Local file not found: {path}")
    return path.read_bytes()


def download(
    url: str,
    verify_ssl: bool = True,
    headers: dict | None = None,
) -> bytes:
    if url.startswith("file://"):
        return _read_local(url)
    hdrs = headers if headers is not None else _BROWSER_HEADERS
    with warnings.catch_warnings():
        if not verify_ssl:
            warnings.simplefilter("ignore")  # suppress urllib3 InsecureRequestWarning
        response = requests.get(url, headers=hdrs, timeout=TIMEOUT, verify=verify_ssl)
    response.raise_for_status()
    if not response.content:
        raise ValueError(
            f"Server returned empty response (HTTP {response.status_code}) — "
            "the host may be blocking automated downloads"
        )
    return response.content


def _ocr_pdf(content: bytes) -> str:
    images = convert_from_bytes(content, dpi=300, poppler_path=_POPPLER_PATH)
    pages = [pytesseract.image_to_string(img, lang="eng") for img in images]
    return "\n".join(pages)


def _parse_pdf(content: bytes) -> str:
    if not content.startswith(b"%PDF"):
        raise ValueError(
            "Downloaded content is not a PDF (missing %%PDF header) — "
            "server may have returned an HTML page or redirect"
        )
    reader = PdfReader(io.BytesIO(content))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages).strip()
    if not text:
        # No text layer — fall back to OCR (handles scanned/image-based PDFs)
        text = _ocr_pdf(content).strip()
    if not text:
        raise ValueError(f"PDF has {len(reader.pages)} page(s): text extraction and OCR both returned empty")
    return text


def _parse_text(content: bytes) -> str:
    return content.decode("utf-8", errors="ignore")


def parse(url: str, content: bytes) -> str:
    lower = url.lower().split("?")[0]
    if lower.endswith(".pdf"):
        return _parse_pdf(content)
    return _parse_text(content)


_PARA_RE = re.compile(r"\n\s*\n+")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_HSPACE_RE = re.compile(r"[ \t]+")


def _normalize(text: str) -> str:
    return _HSPACE_RE.sub(" ", text).strip()


def _split_into_blocks(text: str) -> List[str]:
    """Prefer paragraph breaks; fall back to sentence splitting when the
    source has none (common after PDF text extraction collapses newlines)."""
    blocks = [b.strip() for b in _PARA_RE.split(text) if b.strip()]
    if len(blocks) > 1:
        return [_normalize(b) for b in blocks]
    return [s.strip() for s in _SENT_RE.split(_normalize(text)) if s.strip()]


def _group_into_parents(blocks: List[str], parent_size: int) -> List[str]:
    """Group paragraph/sentence blocks into ~parent_size-char parent chunks."""
    parents: List[str] = []
    current: List[str] = []
    current_len = 0
    for block in blocks:
        if current and current_len + len(block) > parent_size:
            parents.append(" ".join(current))
            current, current_len = [], 0
        current.append(block)
        current_len += len(block) + 1
    if current:
        parents.append(" ".join(current))
    return parents


def _slide(text: str, size: int, overlap: int) -> List[str]:
    segments: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        segment = text[start:end]
        if len(segment.strip()) > 60:  # drop near-empty trailing slices
            segments.append(segment)
        start += size - overlap
    return segments


def chunk(text: str, size: int, overlap: int, parent_size: int) -> List[Dict]:
    """Hierarchical chunking: small "child" chunks (embedded and retrieved,
    on paragraph/sentence boundaries rather than blind char cuts) are grouped
    under larger "parent" blocks, which is what actually gets sent to the LLM
    once a child chunk matches (small-to-big retrieval)."""
    blocks = _split_into_blocks(text)
    parents = _group_into_parents(blocks, parent_size)
    result: List[Dict] = []
    for parent_id, parent in enumerate(parents):
        for child in _slide(parent, size, overlap):
            result.append({"text": child, "parent_text": parent, "parent_id": parent_id})
    return result


def load_document(
    url: str,
    source_name: str,
    chunk_size: int,
    chunk_overlap: int,
    parent_chunk_size: int,
    verify_ssl: bool = True,
    headers: dict | None = None,
) -> List[Dict]:
    """Download, parse, and chunk one document. Returns list of chunk dicts.

    `text` and `parent_text` are returned clean (no title/context prefix) —
    composing the final indexed text (title, optional LLM-generated context,
    then the chunk) is the caller's job (see `main.py: cmd_ingest`), since
    that composition depends on presentation metadata this module doesn't own.
    """
    content = download(url, verify_ssl=verify_ssl, headers=headers)
    text = parse(url, content)
    segments = chunk(text, chunk_size, chunk_overlap, parent_chunk_size)
    return [
        {
            "text": seg["text"],
            "parent_text": seg["parent_text"],
            "parent_id": f"{source_name}::p{seg['parent_id']}",
            "source": source_name,
            "url": url,
            "chunk_id": i,
        }
        for i, seg in enumerate(segments)
    ]
