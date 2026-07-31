#!/usr/bin/env python3
"""Ingest everything in data/ into the vector store.

    python scripts/ingest.py
    python scripts/ingest.py --reset          # rebuild from scratch
    python scripts/ingest.py --data-dir path/to/docs

Runs entirely offline: embeddings are local, so no API key is needed here.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bootstrap import build_stack, setup_logging  # noqa: E402
from src.config import get_config  # noqa: E402
from src.ingest.index import index_documents  # noqa: E402
from src.ingest.loader import load_documents  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store.")
    parser.add_argument("--data-dir", default="data", help="Folder of source documents")
    parser.add_argument("--reset", action="store_true", help="Drop the collection first")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    cfg = get_config()
    stack = build_stack(cfg)

    if args.reset:
        stack.store.reset()
        print("Collection reset.")

    data_dir = cfg.resolve_path(args.data_dir)
    docs = load_documents(data_dir, cfg.chunking.parent.fallback_page_chars)
    if not docs:
        print(f"No supported documents found in {data_dir}. Drop .txt/.md/.pdf files there.")
        return 1

    print(f"Loading {len(docs)} document(s) from {data_dir}")
    results = index_documents(docs, stack.store, stack.embedder, cfg)

    width = max((len(r.source) for r in results), default=10)
    print()
    for r in results:
        detail = f"{r.children} children" + (f", {r.parents} parents" if r.parents else "")
        print(f"  {r.source:<{width}}  {r.status:<18} {detail}")

    print(f"\nCollection '{cfg.vectorstore.collection}' now holds {stack.store.count()} chunks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
