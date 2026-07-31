"""Idempotent indexing.

Re-running ingestion must not duplicate or drift. Two mechanisms:
  1. Deterministic child ids (source + parent + child index), so an unchanged
     chunk upserts onto itself.
  2. A per-file content hash stored on every chunk. Unchanged file -> skip
     entirely; changed file -> delete its old chunks before writing new ones,
     which also removes chunks that no longer exist after an edit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.config import Config
from src.embeddings.base import Embedder
from src.ingest.chunker import chunk_document
from src.ingest.loader import SourceDoc
from src.vectorstore.base import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class FileResult:
    source: str
    status: str  # "indexed" | "updated" | "skipped-unchanged"
    children: int
    parents: int


def index_documents(
    docs: list[SourceDoc],
    store: VectorStore,
    embedder: Embedder,
    cfg: Config,
) -> list[FileResult]:
    results: list[FileResult] = []
    for doc in docs:
        results.append(_index_one(doc, store, embedder, cfg))
    return results


def _index_one(doc: SourceDoc, store: VectorStore, embedder: Embedder, cfg: Config) -> FileResult:
    existing = store.get(where={"source": doc.source}, limit=1)
    already_current = bool(existing) and existing[0].metadata.get("content_hash") == doc.content_hash

    if already_current:
        count = len(store.get(where={"source": doc.source}))
        logger.info("%s unchanged; skipping", doc.source)
        return FileResult(doc.source, "skipped-unchanged", count, 0)

    status = "indexed"
    if existing:
        removed = store.delete({"source": doc.source})
        logger.info("%s changed; removed %d stale chunks", doc.source, removed)
        status = "updated"

    children = chunk_document(doc, cfg.chunking, cfg.embeddings.model)
    if not children:
        logger.warning("%s produced no chunks", doc.source)
        return FileResult(doc.source, status, 0, 0)

    batch_size = max(1, cfg.embeddings.batch_size)
    for start in range(0, len(children), batch_size):
        batch = children[start : start + batch_size]
        vectors = embedder.embed_documents([c.text for c in batch])
        store.upsert(
            ids=[c.id for c in batch],
            texts=[c.text for c in batch],
            embeddings=vectors,
            metadatas=[c.metadata for c in batch],
        )

    parents = len({c.metadata["parent_id"] for c in children})
    logger.info("%s: %d children across %d parents", doc.source, len(children), parents)
    return FileResult(doc.source, status, len(children), parents)
