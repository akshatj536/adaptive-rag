#!/usr/bin/env python3
"""Retrieval-only smoke test. Runs the pipeline and prints what it found,
without calling any LLM - so it works with no API keys and burns no quota.

    python scripts/smoke_retrieval.py "how hot is venus"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bootstrap import build_stack, setup_logging  # noqa: E402
from src.config import get_config  # noqa: E402
from src.generation.context import expand_to_parents  # noqa: E402


def main() -> int:
    query = " ".join(sys.argv[1:]) or "what is the tallest volcano"
    setup_logging()
    cfg = get_config()
    stack = build_stack(cfg)

    if stack.store.count() == 0:
        print("Nothing indexed. Run: python scripts/ingest.py")
        return 1

    print(f"Query:     {query}")
    print(f"Pipeline:  {stack.retrieval.describe()}\n")

    ctx = stack.retrieval.run(query)
    print(f"Children retrieved: {len(ctx.candidates)}")
    for hit in ctx.candidates:
        preview = hit.text.replace("\n", " ")[:90]
        print(f"  {hit.score:+.3f}  {hit.metadata.get('source')}  {preview}…")

    parents = expand_to_parents(ctx.candidates, cfg)
    print(f"\nExpanded to {len(parents)} unique parent(s):")
    for i, parent in enumerate(parents, start=1):
        print(f"  [{i}] {parent.label}  {len(parent.text)} chars, "
              f"{len(parent.child_scores)} matching child(ren)")

    for line in ctx.trace:
        print(f"\ntrace: {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
