"""
Document downloader, parser, and chunker.

Observe phase of OODA: pull raw intelligence from source URLs,
normalize it into clean text, then split into retrieval-sized chunks.
"""

import io
import json
import os
import re
import warnings
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname
import requests
import pdfplumber
import pytesseract
from pdf2image import convert_from_bytes
from pypdf import PdfReader
from typing import Callable, List, Dict, Optional, Tuple


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


def _rows_to_markdown(rows: List[List[Optional[str]]]) -> str:
    """Render pdfplumber table rows (list of lists, cells may be None) as a
    markdown table. First row is treated as the header."""
    clean = [[("" if c is None else str(c).strip().replace("\n", " ")) for c in row] for row in rows]
    clean = [r for r in clean if any(c for c in r)]
    if not clean:
        return ""
    width = max(len(r) for r in clean)

    def _fmt(row: List[str]) -> str:
        padded = row + [""] * (width - len(row))
        return "| " + " | ".join(padded) + " |"

    lines = [_fmt(clean[0]), "| " + " | ".join(["---"] * width) + " |"]
    lines += [_fmt(r) for r in clean[1:]]
    return "\n".join(lines)


def _extract_tables(content: bytes) -> Dict[int, List[Dict]]:
    """Best-effort table extraction via pdfplumber: page index (0-based) ->
    list of {"rows": [[...]], "markdown": "..."}. Additive to the existing
    pypdf text path — never raises, so a pdfplumber failure (e.g. a PDF it
    can't parse) can't break ingestion that already works via pypdf/OCR."""
    try:
        result: Dict[int, List[Dict]] = {}
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for i, page in enumerate(pdf.pages):
                tables = [t for t in (page.extract_tables() or []) if t]
                entries = [
                    {"rows": rows, "markdown": md}
                    for rows in tables
                    if (md := _rows_to_markdown(rows))
                ]
                if entries:
                    result[i] = entries
        return result
    except Exception:
        return {}


def _extract_images(content: bytes) -> Dict[int, List[bytes]]:
    """Best-effort image extraction via pdfplumber: page index (0-based) ->
    list of cropped-image PNG bytes. Skips tiny artifacts (bullets, rule
    lines). Never raises — mirrors `_extract_tables`."""
    try:
        result: Dict[int, List[bytes]] = {}
        resolution = 150
        scale = resolution / 72
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for i, page in enumerate(pdf.pages):
                if not page.images:
                    continue
                rendered = page.to_image(resolution=resolution).original
                crops: List[bytes] = []
                for im in page.images:
                    bbox = (im["x0"] * scale, im["top"] * scale, im["x1"] * scale, im["bottom"] * scale)
                    try:
                        cropped = rendered.crop(bbox)
                    except Exception:
                        continue
                    if cropped.width < 30 or cropped.height < 30:
                        continue
                    buf = io.BytesIO()
                    cropped.convert("RGB").save(buf, format="PNG")
                    crops.append(buf.getvalue())
                if crops:
                    result[i] = crops
        return result
    except Exception:
        return {}


def _split_table_block(text: str) -> Tuple[str, Optional[str]]:
    """A `[TABLE pN#i]` block is stored as marker / JSON rows / markdown on
    the first three lines (see `_parse_pdf`). Split back into
    (display_text = marker + markdown, json_rows_or_None)."""
    if not text.startswith("[TABLE"):
        return text, None
    parts = text.split("\n", 2)
    if len(parts) != 3:
        return text, None
    marker, json_rows, markdown = parts
    return f"{marker}\n{markdown}", json_rows


def _parse_pdf(content: bytes, image_captioner: Optional[Callable[[bytes], str]] = None) -> str:
    if not content.startswith(b"%PDF"):
        raise ValueError(
            "Downloaded content is not a PDF (missing %%PDF header) — "
            "server may have returned an HTML page or redirect"
        )
    reader = PdfReader(io.BytesIO(content))
    pages = [page.extract_text() or "" for page in reader.pages]

    # Tables: appended as their own atomic `[TABLE pN#i]` blocks per page,
    # alongside (not replacing) pypdf's flattened text — the flattened form
    # still contributes surrounding-paragraph context, while the markdown
    # form gives a clean, citable representation of the actual data.
    for i, tables in _extract_tables(content).items():
        if i >= len(pages):
            continue
        blocks = "\n\n".join(
            f"[TABLE p{i + 1}#{j}]\n{json.dumps(t['rows'])}\n{t['markdown']}"
            for j, t in enumerate(tables)
        )
        if blocks:
            pages[i] = f"{pages[i]}\n\n{blocks}"

    # Images/charts: opt-in (`ingest --describe-images`) since captioning
    # needs an LLM call per image and a vision-capable model in LM Studio.
    if image_captioner is not None:
        for i, images in _extract_images(content).items():
            if i >= len(pages):
                continue
            blocks = []
            for j, img_bytes in enumerate(images):
                try:
                    caption = image_captioner(img_bytes)
                except Exception:
                    caption = None
                if caption:
                    # Collapse any blank lines the LLM caption might contain —
                    # atomic blocks are blank-line delimited (see `_extract_markers`).
                    caption = _PARA_RE.sub(" ", caption.strip())
                    blocks.append(f"[IMAGE p{i + 1}#{j}]\n{caption}")
            if blocks:
                pages[i] = f"{pages[i]}\n\n" + "\n\n".join(blocks)

    text = "\n".join(pages).strip()
    if not text:
        # No text layer — fall back to OCR (handles scanned/image-based PDFs)
        text = _ocr_pdf(content).strip()
    if not text:
        raise ValueError(f"PDF has {len(reader.pages)} page(s): text extraction and OCR both returned empty")
    return text


def _parse_text(content: bytes) -> str:
    return content.decode("utf-8", errors="ignore")


def parse(url: str, content: bytes, image_captioner: Optional[Callable[[bytes], str]] = None) -> str:
    lower = url.lower().split("?")[0]
    if lower.endswith(".pdf"):
        return _parse_pdf(content, image_captioner)
    return _parse_text(content)


_PARA_RE = re.compile(r"\n\s*\n+")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_HSPACE_RE = re.compile(r"[ \t]+")

# Section-aware chunking: cheap heuristics for doctrine/field-manual style
# headings (no LLM call) — "Chapter 3", "Section 3-2", or a short ALL-CAPS
# line, all common in the FM/ATP/AJP corpus this project targets.
_HEADING_KEYWORD_RE = re.compile(r"^(chapter|section)\s+[\divxlc]+([.\-:].*)?$", re.IGNORECASE)


def _normalize(text: str) -> str:
    return _HSPACE_RE.sub(" ", text).strip()


def _looks_like_heading(line: str) -> bool:
    line = line.strip()
    if not (3 <= len(line) <= 80):
        return False
    if _HEADING_KEYWORD_RE.match(line):
        return True
    if line[-1] in ".?!,;:":
        return False
    words = line.split()
    if not (1 <= len(words) <= 8):
        return False
    letters = [c for c in line if c.isalpha()]
    if len(letters) < 3:
        return False
    upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    return upper_ratio > 0.9


def _extract_markers(text: str) -> List[Tuple[str, str]]:
    """Line-scanning pre-pass that pulls out heading lines and `[TABLE]`/
    `[IMAGE]` atomic blocks from the raw text. Needed because PDF text
    extraction commonly collapses the blank-line paragraph breaks that would
    otherwise isolate a heading from the prose around it (pypdf routinely
    emits a whole page as one blank-line-free run) — so headings/markers are
    found by scanning individual lines rather than relying on `_PARA_RE`
    having already separated them. Returns (text, kind) segments in order;
    "prose" segments are further split into paragraphs/sentences by
    `_annotate_blocks`."""
    lines = text.split("\n")
    segments: List[Tuple[str, str]] = []
    current: List[str] = []

    def _flush() -> None:
        if current:
            block = "\n".join(current).strip()
            if block:
                segments.append((block, "prose"))
            current.clear()

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("[TABLE") or stripped.startswith("[IMAGE"):
            _flush()
            kind = "table" if stripped.startswith("[TABLE") else "image"
            block_lines = [lines[i]]
            i += 1
            while i < len(lines) and lines[i].strip():  # atomic blocks are blank-line delimited
                block_lines.append(lines[i])
                i += 1
            segments.append(("\n".join(block_lines).strip(), kind))
            continue
        if _looks_like_heading(stripped):
            _flush()
            segments.append((stripped, "heading"))
            i += 1
            continue
        current.append(lines[i])
        i += 1
    _flush()
    return segments


def _annotate_blocks(text: str) -> List[Tuple[str, str]]:
    """Split into (block_text, kind) pairs — kind is "table"/"image" (atomic,
    never split further), "heading" (becomes a section marker, not indexed on
    its own), or "prose" (grouped/sliding-window/semantic split as before).
    Prose runs between headings/markers are split on paragraph breaks where
    present, falling back to sentence splitting when they aren't (common
    after PDF text extraction collapses newlines)."""
    annotated: List[Tuple[str, str]] = []
    for segment, kind in _extract_markers(text):
        if kind != "prose":
            annotated.append((segment, kind))
            continue
        raw_blocks = [b.strip() for b in _PARA_RE.split(segment) if b.strip()]
        if len(raw_blocks) > 1:
            annotated.extend((_normalize(b), "prose") for b in raw_blocks)
        else:
            annotated.extend(
                (s.strip(), "prose") for s in _SENT_RE.split(_normalize(segment)) if s.strip()
            )
    return annotated


def _group_into_parents(annotated: List[Tuple[str, str]], parent_size: int) -> List[Dict]:
    """Group prose blocks into ~parent_size-char parent chunks, same as
    before, but: a heading flushes the current parent and starts a new
    section (the heading text itself isn't indexed as its own low-value
    chunk); a table/image block flushes the current parent and becomes its
    own atomic parent, so it's never merged with unrelated prose or split by
    the child-level chunker."""
    parents: List[Dict] = []
    current: List[str] = []
    current_len = 0
    section: Optional[str] = None

    def _flush() -> None:
        nonlocal current, current_len
        if current:
            parents.append({"text": " ".join(current), "section": section, "atomic": False})
            current, current_len = [], 0

    for block, kind in annotated:
        if kind == "heading":
            _flush()
            section = block
            continue
        if kind in ("table", "image"):
            _flush()
            parents.append({"text": block, "section": section, "atomic": True})
            continue
        if current and current_len + len(block) > parent_size:
            _flush()
        current.append(block)
        current_len += len(block) + 1
    _flush()
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


def _cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def _semantic_split(
    text: str,
    embed_fn: Callable[[List[str]], List[List[float]]],
    max_chars: int,
    min_chars: int,
    threshold: float,
) -> List[str]:
    """Split a parent block into child chunks at embedding-similarity
    boundaries instead of a fixed-size sliding window: sentences are embedded
    (reusing the already-loaded document embedder — cheap on CPU, unlike an
    LLM call), and a new chunk starts wherever consecutive-sentence cosine
    similarity drops below `threshold` (a topic shift), subject to a minimum
    chunk size so near-identical adjacent sentences don't fragment into
    one-sentence chunks. `max_chars` is a hard safety cap for runs of
    consistently-similar sentences that would otherwise grow unbounded."""
    sentences = [s.strip() for s in _SENT_RE.split(_normalize(text)) if s.strip()]
    if len(sentences) <= 1:
        return [text] if text.strip() else []

    embeddings = embed_fn(sentences)
    chunks: List[str] = []
    current = [sentences[0]]
    current_len = len(sentences[0])
    for i in range(1, len(sentences)):
        sentence = sentences[i]
        sim = _cosine_sim(embeddings[i - 1], embeddings[i])
        would_overflow = current_len + len(sentence) > max_chars
        topic_shift = sim < threshold and current_len >= min_chars
        if would_overflow or topic_shift:
            chunks.append(" ".join(current))
            current = [sentence]
            current_len = len(sentence)
        else:
            current.append(sentence)
            current_len += len(sentence) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk(
    text: str,
    size: int,
    overlap: int,
    parent_size: int,
    embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    semantic_threshold: float = 0.5,
) -> List[Dict]:
    """Hierarchical chunking: small "child" chunks (embedded and retrieved)
    are grouped under larger "parent" blocks, which is what actually gets
    sent to the LLM once a child chunk matches (small-to-big retrieval).

    Child chunks are produced by a fixed-size sliding window by default, or
    by embedding-similarity boundaries when `embed_fn` is given (opt-in
    `ingest --semantic-chunk`, see `_semantic_split`). Table/image parents are
    always atomic — never split at the child level regardless of size."""
    annotated = _annotate_blocks(text)
    parents = _group_into_parents(annotated, parent_size)
    result: List[Dict] = []
    for parent_id, parent in enumerate(parents):
        if parent["atomic"]:
            children = [parent["text"]]
        elif embed_fn is not None:
            children = _semantic_split(parent["text"], embed_fn, size, max(size // 3, 1), semantic_threshold)
        else:
            children = _slide(parent["text"], size, overlap)
        for child in children:
            result.append({
                "text": child,
                "parent_text": parent["text"],
                "parent_id": parent_id,
                "section": parent["section"],
            })
    return result


def load_document(
    url: str,
    source_name: str,
    chunk_size: int,
    chunk_overlap: int,
    parent_chunk_size: int,
    verify_ssl: bool = True,
    headers: dict | None = None,
    embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    image_captioner: Optional[Callable[[bytes], str]] = None,
    semantic_threshold: float = 0.5,
) -> List[Dict]:
    """Download, parse, and chunk one document. Returns list of chunk dicts.

    `text` and `parent_text` are returned clean (no title/context prefix) —
    composing the final indexed text (title, optional section/LLM-generated
    context, then the chunk) is the caller's job (see `main.py: cmd_ingest`),
    since that composition depends on presentation metadata this module
    doesn't own. `table_json` is present only on chunks that are a table
    (the raw pdfplumber rows, JSON-encoded, for structured access without
    re-parsing the markdown).
    """
    content = download(url, verify_ssl=verify_ssl, headers=headers)
    text = parse(url, content, image_captioner=image_captioner)
    segments = chunk(text, chunk_size, chunk_overlap, parent_chunk_size, embed_fn, semantic_threshold)
    docs = []
    for i, seg in enumerate(segments):
        text_, table_json = _split_table_block(seg["text"])
        parent_text_, _ = _split_table_block(seg["parent_text"])
        doc = {
            "text": text_,
            "parent_text": parent_text_,
            "parent_id": f"{source_name}::p{seg['parent_id']}",
            "source": source_name,
            "url": url,
            "chunk_id": i,
            "section": seg["section"],
        }
        if table_json:
            doc["table_json"] = table_json
        docs.append(doc)
    return docs
