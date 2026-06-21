"""
Document downloader, parser, and chunker.

Observe phase of OODA: pull raw intelligence from source URLs,
normalize it into clean text, then split into retrieval-sized chunks.
"""

import io
import re
import warnings
from pathlib import Path
from urllib.parse import urlparse
import requests
import pytesseract
from pdf2image import convert_from_bytes
from pypdf import PdfReader
from typing import List, Dict


TIMEOUT = 120  # seconds; some military PDFs are large

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) "
        "Gecko/20100101 Firefox/125.0"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def _read_local(url: str) -> bytes:
    path = Path(urlparse(url).path)
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
    images = convert_from_bytes(content, dpi=300)
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


def chunk(text: str, size: int, overlap: int) -> List[str]:
    # Collapse whitespace runs to single spaces so chunks are dense
    text = re.sub(r"\s+", " ", text).strip()
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        segment = text[start:end]
        if len(segment.strip()) > 60:  # drop near-empty trailing slices
            chunks.append(segment)
        start += size - overlap
    return chunks


def load_document(
    url: str,
    source_name: str,
    chunk_size: int,
    chunk_overlap: int,
    verify_ssl: bool = True,
    headers: dict | None = None,
) -> List[Dict]:
    """Download, parse, and chunk one document. Returns list of chunk dicts."""
    content = download(url, verify_ssl=verify_ssl, headers=headers)
    text = parse(url, content)
    segments = chunk(text, chunk_size, chunk_overlap)
    return [
        {
            "text": seg,
            "source": source_name,
            "url": url,
            "chunk_id": i,
        }
        for i, seg in enumerate(segments)
    ]
