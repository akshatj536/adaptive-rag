"""Cross-encoder reranking.

The bi-encoder used for retrieval embeds query and passage separately, which is
what makes it fast enough to search thousands of chunks - but it never lets the
two texts interact. A cross-encoder reads (query, passage) jointly and is far
more accurate, at a cost that only makes sense on a shortlist.

So: retrieve wide and cheap, then rerank narrow and precise. Runs locally, so
it costs no API quota.
"""

from __future__ import annotations

import logging

from src.config import RerankConfig
from src.retrieval.base import POST_RETRIEVAL, RetrievalComponent, RetrievalContext

logger = logging.getLogger(__name__)


class RerankComponent(RetrievalComponent):
    name = "rerank"
    stage = POST_RETRIEVAL

    def __init__(self, cfg: RerankConfig) -> None:
        self._cfg = cfg
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading cross-encoder %s (first run downloads it)", self._cfg.model)
            self._model = CrossEncoder(self._cfg.model, device="cpu")
        return self._model

    def run(self, ctx: RetrievalContext) -> RetrievalContext:
        if not ctx.candidates:
            ctx.log("rerank: nothing to rerank")
            return ctx

        # Score against the user's real question, not effective_query: a HyDE
        # draft is synthetic text, and cross-encoders are trained on genuine
        # query-passage pairs.
        pairs = [(ctx.query, hit.text) for hit in ctx.candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)

        before = [h.id for h in ctx.candidates[: ctx.top_k]]
        for hit, score in zip(ctx.candidates, scores):
            hit.score = float(score)
            hit.source_component = f"{hit.source_component}+rerank".lstrip("+")

        ctx.candidates.sort(key=lambda h: h.score, reverse=True)
        ctx.candidates = ctx.candidates[: ctx.top_k]

        moved = sum(1 for i, h in enumerate(ctx.candidates) if i >= len(before) or h.id != before[i])
        ctx.log(
            f"rerank: scored {len(pairs)} candidates -> kept {len(ctx.candidates)}, "
            f"{moved} position(s) changed"
        )
        return ctx
