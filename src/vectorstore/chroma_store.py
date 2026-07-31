from __future__ import annotations

import logging
from pathlib import Path

from src.vectorstore.base import Hit, Metadata, Vector, VectorStore

logger = logging.getLogger(__name__)

# Cosine space: embeddings are normalized upstream, and it makes
# similarity = 1 - distance, which is comparable across backends.
_COLLECTION_METADATA = {"hnsw:space": "cosine"}


class ChromaStore(VectorStore):
    def __init__(self, path: str | Path, collection: str) -> None:
        import chromadb
        from chromadb.config import Settings

        self._path = str(path)
        self._collection_name = collection
        self._client = chromadb.PersistentClient(
            path=self._path,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._get_or_create()

    def _get_or_create(self):
        # No embedding_function is registered: we always pass vectors in
        # explicitly, so Chroma never silently downloads a model of its own.
        return self._client.get_or_create_collection(
            name=self._collection_name,
            metadata=_COLLECTION_METADATA,
            embedding_function=None,
        )

    def upsert(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[Vector],
        metadatas: list[Metadata],
    ) -> None:
        if not ids:
            return
        self._collection.upsert(
            ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas
        )

    def query(self, embedding: Vector, top_k: int, where: Metadata | None = None) -> list[Hit]:
        if self.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self.count()),
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[Hit] = []
        for cid, doc, meta, dist in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            hits.append(
                Hit(
                    id=cid,
                    text=doc or "",
                    metadata=dict(meta or {}),
                    score=1.0 - float(dist),  # cosine distance -> similarity
                    source_component="vector",
                )
            )
        return hits

    def get(self, where: Metadata | None = None, limit: int | None = None) -> list[Hit]:
        result = self._collection.get(
            where=where or None,
            limit=limit,
            include=["documents", "metadatas"],
        )
        return [
            Hit(id=cid, text=doc or "", metadata=dict(meta or {}))
            for cid, doc, meta in zip(
                result["ids"], result["documents"], result["metadatas"]
            )
        ]

    def delete(self, where: Metadata) -> int:
        existing = self._collection.get(where=where, include=[])
        ids = existing["ids"]
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def count(self) -> int:
        return int(self._collection.count())

    def reset(self) -> None:
        logger.info("Resetting collection %s", self._collection_name)
        self._client.delete_collection(self._collection_name)
        self._collection = self._get_or_create()


def build_vector_store(cfg, project_config) -> VectorStore:
    """cfg: VectorStoreConfig. Path is resolved against the project root so the
    store is the same no matter which directory the process was launched from."""
    if cfg.provider != "chroma":
        raise ValueError(f"Unsupported vectorstore provider: {cfg.provider!r}")
    return ChromaStore(project_config.resolve_path(cfg.path), cfg.collection)
