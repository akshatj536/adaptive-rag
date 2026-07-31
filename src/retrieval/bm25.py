"""Sparse BM25 retrieval, fused with dense results via RRF.

BM25 matches literal terms, so it catches exactly what embeddings miss: rare
identifiers, acronyms, product codes, surnames. The two are complementary,
which is why hybrid search usually beats either alone.
"""

from __future__ import annotations

import logging
import re

from src.config import Bm25Config
from src.retrieval.base import RETRIEVAL, RetrievalComponent, RetrievalContext
from src.retrieval.fusion import reciprocal_rank_fusion
from src.vectorstore.base import Hit, VectorStore

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Component(RetrievalComponent):
    """Runs its own sparse search, then fuses with whatever dense candidates
    are already in the context.

    The index is built in memory from the whole collection on first use. That
    is fine at this scale; if the corpus grows past memory, this class is the
    only thing that needs to change.
    """

    name = "bm25"
    stage = RETRIEVAL

    def __init__(self, store: VectorStore, cfg: Bm25Config) -> None:
        self._store = store
        self._cfg = cfg
        self._index = None
        self._hits: list[Hit] = []
        self._built_for_count = -1

    def _build_index(self) -> None:
        from rank_bm25 import BM25Okapi

        self._hits = self._store.get()
        if not self._hits:
            self._index = None
            self._built_for_count = 0
            return

        logger.info("Building BM25 index over %d chunks", len(self._hits))
        corpus = [tokenize(h.text) for h in self._hits]
        self._index = BM25Okapi(corpus, k1=self._cfg.k1, b=self._cfg.b)
        self._built_for_count = len(self._hits)

    def _ensure_index(self) -> None:
        # Rebuild if the collection changed size (e.g. a re-ingest happened
        # while a long-lived Streamlit session was open).
        if self._index is None or self._store.count() != self._built_for_count:
            self._build_index()

    def run(self, ctx: RetrievalContext) -> RetrievalContext:
        self._ensure_index()
        if self._index is None:
            ctx.log("bm25: index empty, skipped")
            return ctx

        tokens = tokenize(ctx.effective_query)
        if not tokens:
            ctx.log("bm25: query had no indexable terms, skipped")
            return ctx

        scores = self._index.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        sparse: list[Hit] = []
        for i in ranked[: ctx.fetch_k]:
            if scores[i] <= 0:
                break  # no term overlap at all; not a candidate
            source = self._hits[i]
            sparse.append(
                Hit(
                    id=source.id,
                    text=source.text,
                    metadata=source.metadata,
                    score=float(scores[i]),
                    source_component="bm25",
                )
            )

        if not ctx.candidates:
            ctx.candidates = sparse
            ctx.log(f"bm25: {len(sparse)} candidates (no dense results to fuse)")
            return ctx

        before = len(ctx.candidates)
        ctx.candidates = reciprocal_rank_fusion(
            [ctx.candidates, sparse], k=self._cfg.rrf_k, limit=ctx.fetch_k
        )
        ctx.log(
            f"bm25: {len(sparse)} sparse + {before} dense -> "
            f"{len(ctx.candidates)} fused (RRF k={self._cfg.rrf_k})"
        )
        return ctx
