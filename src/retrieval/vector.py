from __future__ import annotations

from src.embeddings.base import Embedder
from src.retrieval.base import RETRIEVAL, RetrievalComponent, RetrievalContext
from src.retrieval.fusion import DEFAULT_K, reciprocal_rank_fusion
from src.vectorstore.base import VectorStore


class VectorComponent(RetrievalComponent):
    """Dense retrieval. The baseline every other component refines."""

    name = "vector"
    stage = RETRIEVAL

    def __init__(self, store: VectorStore, embedder: Embedder, rrf_k: int = DEFAULT_K) -> None:
        self._store = store
        self._embedder = embedder
        self._rrf_k = rrf_k

    def run(self, ctx: RetrievalContext) -> RetrievalContext:
        embedding = self._embedder.embed_query(ctx.effective_query)
        hits = self._store.query(embedding, top_k=ctx.fetch_k)

        if not ctx.candidates:
            ctx.candidates = hits
            ctx.log(f"vector: {len(hits)} candidates for {ctx.effective_query[:60]!r}")
            return ctx

        # Another retrieval component already ran. Fuse rather than overwrite,
        # so retrieval-stage components compose in any order.
        before = len(ctx.candidates)
        ctx.candidates = reciprocal_rank_fusion(
            [ctx.candidates, hits], k=self._rrf_k, limit=ctx.fetch_k
        )
        ctx.log(
            f"vector: {len(hits)} dense + {before} existing -> "
            f"{len(ctx.candidates)} fused (RRF k={self._rrf_k})"
        )
        return ctx
