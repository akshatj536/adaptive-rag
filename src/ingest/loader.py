from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


@dataclass
class Page:
    """A page-sized unit of a source document. Real pages for PDFs; synthetic
    fixed-size blocks for plain text, which has no page structure."""

    page_no: int
    text: str


@dataclass
class SourceDoc:
    path: Path
    content_hash: str
    pages: list[Page] = field(default_factory=list)

    @property
    def source(self) -> str:
        return self.path.name

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)


def load_documents(data_dir: Path, fallback_page_chars: int = 3000) -> list[SourceDoc]:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    docs: list[SourceDoc] = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            logger.warning("Skipping unsupported file type: %s", path.name)
            continue
        doc = load_document(path, fallback_page_chars)
        if doc.pages:
            docs.append(doc)
        else:
            logger.warning("No extractable text in %s; skipping", path.name)
    return docs


def load_document(path: Path, fallback_page_chars: int = 3000) -> SourceDoc:
    raw = path.read_bytes()
    content_hash = hashlib.sha256(raw).hexdigest()
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        pages = _load_pdf(path)
    else:
        text = raw.decode("utf-8", errors="replace")
        pages = _paginate(text, fallback_page_chars)

    pages = [p for p in pages if p.text.strip()]
    return SourceDoc(path=path, content_hash=content_hash, pages=pages)


def _load_pdf(path: Path) -> list[Page]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # a single malformed page shouldn't kill ingest
            logger.warning("Could not extract page %d of %s: %s", i, path.name, exc)
            text = ""
        pages.append(Page(page_no=i, text=text))
    return pages


def _paginate(text: str, page_chars: int) -> list[Page]:
    """Split plain text into synthetic pages on paragraph boundaries, so a page
    never cuts mid-sentence."""
    if len(text) <= page_chars:
        return [Page(page_no=1, text=text)]

    pages: list[Page] = []
    buffer: list[str] = []
    size = 0
    for paragraph in text.split("\n\n"):
        chunk = paragraph + "\n\n"
        if size + len(chunk) > page_chars and buffer:
            pages.append(Page(page_no=len(pages) + 1, text="".join(buffer).strip()))
            buffer, size = [], 0
        buffer.append(chunk)
        size += len(chunk)
    if buffer:
        pages.append(Page(page_no=len(pages) + 1, text="".join(buffer).strip()))
    return pages
