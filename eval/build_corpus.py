#!/usr/bin/env python3
"""Build the evaluation corpus from the selected questions and index it.

Writes one file per unique context paragraph, so that after chunking
one paragraph == one page == one parent. That makes retrieval ground truth an
exact check ("was a gold-titled paragraph retrieved?") rather than a fuzzy
span-overlap heuristic.

Indexes into its own collection so the working corpus is left untouched.

    python eval/build_corpus.py
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.bootstrap import setup_logging            # noqa: E402
from src.config import get_config                  # noqa: E402
from src.embeddings.local import build_embedder    # noqa: E402
from src.ingest.index import index_documents       # noqa: E402
from src.ingest.loader import load_documents       # noqa: E402
from src.vectorstore.chroma_store import build_vector_store  # noqa: E402

EVAL_DIR = ROOT / "eval"
CORPUS_DIR = EVAL_DIR / "corpus"
QUESTIONS = EVAL_DIR / "selected_questions.jsonl"
TITLE_MAP = EVAL_DIR / "corpus_titles.json"
COLLECTION = "eval_hotpot"

# Comfortably above the longest HotpotQA paragraph, so a paragraph is never
# split across two synthetic "pages" and therefore never across two parents.
PAGE_CHARS = 8000


def slug(title: str) -> str:
    # Strip leading dots as well as separators: a title like "...And Found"
    # would otherwise produce a dotfile, which the loader skips as hidden.
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", title).strip("._-")[:80] or "para"
    return f"{base}__{hashlib.sha1(title.encode()).hexdigest()[:8]}.txt"


def main() -> int:
    setup_logging()
    records = [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]
    print(f"questions: {len(records)}")

    paragraphs: dict[str, str] = {}
    for rec in records:
        for title, sentences in zip(rec["context"]["title"], rec["context"]["sentences"]):
            paragraphs.setdefault(title, " ".join(sentences).strip())

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    for stale in CORPUS_DIR.glob("*.txt"):
        stale.unlink()

    mapping = {}
    longest = 0
    for title, text in sorted(paragraphs.items()):
        name = slug(title)
        (CORPUS_DIR / name).write_text(text, encoding="utf-8")
        mapping[name] = title
        longest = max(longest, len(text))
    TITLE_MAP.write_text(json.dumps(mapping, indent=2, ensure_ascii=False))
    print(f"paragraphs written: {len(mapping)} (longest {longest} chars, page cap {PAGE_CHARS})")

    cfg = get_config()
    cfg = dataclasses.replace(
        cfg,
        vectorstore=dataclasses.replace(cfg.vectorstore, collection=COLLECTION),
        chunking=dataclasses.replace(
            cfg.chunking,
            parent=dataclasses.replace(cfg.chunking.parent, fallback_page_chars=PAGE_CHARS),
        ),
    )

    embedder = build_embedder(cfg.embeddings)
    store = build_vector_store(cfg.vectorstore, cfg)
    store.reset()

    docs = load_documents(CORPUS_DIR, cfg.chunking.parent.fallback_page_chars)
    results = index_documents(docs, store, embedder, cfg)

    multi = [r for r in results if r.parents > 1]
    print(f"\nindexed {len(results)} file(s) -> {store.count()} chunks in '{COLLECTION}'")
    print(f"files that produced more than one parent: {len(multi)} (want 0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
