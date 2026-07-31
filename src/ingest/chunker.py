"""Parent-child chunking.

Children are small (512 tokens) and are what gets embedded and retrieved -
precision. Parents are page-sized and are what gets fed to the LLM - context.
Every child records its parent's id and text so generation can expand back up.
"""

from __future__ import annotations

import functools
import hashlib
import logging
from dataclasses import dataclass

from src.config import ChunkingConfig
from src.ingest.loader import SourceDoc

logger = logging.getLogger(__name__)

# Rough chars-per-token, used only if the real tokenizer can't be loaded.
_CHARS_PER_TOKEN = 4


@dataclass
class Parent:
    id: str
    text: str
    page_no: int
    index: int


@dataclass
class Child:
    id: str
    text: str
    metadata: dict


def chunk_document(doc: SourceDoc, cfg: ChunkingConfig, model_name: str) -> list[Child]:
    parents = _build_parents(doc, cfg)
    splitter = _child_splitter(cfg, model_name)
    max_meta = cfg.parent.max_metadata_chars

    children: list[Child] = []
    for parent in parents:
        parent_text = parent.text[:max_meta]
        if len(parent.text) > max_meta:
            logger.warning(
                "Parent %s truncated to %d chars for metadata storage", parent.id, max_meta
            )
        for child_index, text in enumerate(splitter.split_text(parent.text)):
            if not text.strip():
                continue
            child_id = _hash(f"{doc.source}:{parent.id}:{child_index}")
            children.append(
                Child(
                    id=child_id,
                    text=text,
                    metadata={
                        "parent_id": parent.id,
                        "parent_text": parent_text,
                        "source": doc.source,
                        "source_path": str(doc.path),
                        "page_no": parent.page_no,
                        "content_hash": doc.content_hash,
                        "child_index": child_index,
                    },
                )
            )
    return children


def _build_parents(doc: SourceDoc, cfg: ChunkingConfig) -> list[Parent]:
    granularity = cfg.parent.granularity

    if granularity == "page":
        return [
            Parent(
                id=_parent_id(doc.source, i),
                text=page.text,
                page_no=page.page_no,
                index=i,
            )
            for i, page in enumerate(doc.pages)
        ]

    if granularity == "document":
        return [Parent(id=_parent_id(doc.source, 0), text=doc.full_text, page_no=0, index=0)]

    if granularity == "block":
        splitter = _block_splitter(cfg)
        return [
            Parent(id=_parent_id(doc.source, i), text=text, page_no=0, index=i)
            for i, text in enumerate(splitter.split_text(doc.full_text))
        ]

    raise ValueError(
        f"Unknown parent granularity {granularity!r}; expected page | block | document"
    )


def _parent_id(source: str, index: int) -> str:
    return _hash(f"{source}:parent:{index}")


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


def _child_splitter(cfg: ChunkingConfig, model_name: str):
    return _make_splitter(cfg.child_chunk_size, cfg.child_chunk_overlap, model_name)


def _block_splitter(cfg: ChunkingConfig):
    return _make_splitter(cfg.parent.block_size, cfg.parent.block_overlap, model_name=None)


@functools.lru_cache(maxsize=8)
def _make_splitter(chunk_size: int, overlap: int, model_name: str | None):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    if model_name:
        tokenizer = _load_tokenizer(model_name)
        if tokenizer is not None:
            # Token-accurate: 512 means 512 embedding-model tokens, matching
            # the model's real context window rather than a character guess.
            return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                tokenizer, chunk_size=chunk_size, chunk_overlap=overlap
            )

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size * _CHARS_PER_TOKEN,
        chunk_overlap=overlap * _CHARS_PER_TOKEN,
        length_function=len,
    )


@functools.lru_cache(maxsize=4)
def _load_tokenizer(model_name: str):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_name)
    except Exception as exc:
        logger.warning(
            "Could not load tokenizer for %s (%s); falling back to ~%d chars/token",
            model_name, exc, _CHARS_PER_TOKEN,
        )
        return None
