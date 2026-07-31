from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

Vector = list[float]
Metadata = dict[str, Any]


@dataclass
class Hit:
    """One retrieved child chunk. Common currency across every retrieval
    component, so components can be reordered without conversion."""

    id: str
    text: str
    metadata: Metadata = field(default_factory=dict)
    score: float = 0.0  # higher is better, regardless of backend
    source_component: str = ""


class VectorStore(ABC):
    """Swap seam: callers depend on this, never on ChromaDB directly."""

    @abstractmethod
    def upsert(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[Vector],
        metadatas: list[Metadata],
    ) -> None:
        """Insert or replace by id. Must be idempotent for identical input."""

    @abstractmethod
    def query(self, embedding: Vector, top_k: int, where: Metadata | None = None) -> list[Hit]:
        ...

    @abstractmethod
    def get(self, where: Metadata | None = None, limit: int | None = None) -> list[Hit]:
        """Fetch by metadata filter without a vector. Used for idempotency
        checks and (in slice 2) building the BM25 index."""

    @abstractmethod
    def delete(self, where: Metadata) -> int:
        """Delete everything matching the filter; returns how many were removed."""

    @abstractmethod
    def count(self) -> int:
        ...

    @abstractmethod
    def reset(self) -> None:
        """Drop and recreate the collection."""
