"""Reciprocal rank fusion.

Dense and sparse retrievers produce scores on incompatible scales - cosine
similarity versus BM25 term weights - so their scores cannot be added directly.
RRF sidesteps this by discarding the scores and combining *ranks*, which is why
it is the standard way to fuse hybrid search results.

    score(d) = sum over lists of 1 / (k + rank(d))
"""

from __future__ import annotations

from src.vectorstore.base import Hit

DEFAULT_K = 60


def reciprocal_rank_fusion(
    rankings: list[list[Hit]], k: int = DEFAULT_K, limit: int | None = None
) -> list[Hit]:
    """Fuse ranked lists. Documents appearing in several lists rank highest."""
    scores: dict[str, float] = {}
    best: dict[str, Hit] = {}
    origins: dict[str, set[str]] = {}

    for ranking in rankings:
        for rank, hit in enumerate(ranking):
            scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (k + rank + 1)
            origins.setdefault(hit.id, set()).add(hit.source_component or "?")
            # Keep the richest copy of the hit; any list's text/metadata will do.
            if hit.id not in best or (not best[hit.id].text and hit.text):
                best[hit.id] = hit

    fused: list[Hit] = []
    for hit_id, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        hit = best[hit_id]
        fused.append(
            Hit(
                id=hit.id,
                text=hit.text,
                metadata=hit.metadata,
                score=score,
                source_component="+".join(sorted(origins[hit_id])),
            )
        )
    return fused[:limit] if limit else fused
