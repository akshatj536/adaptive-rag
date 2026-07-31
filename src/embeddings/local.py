from __future__ import annotations

import logging

from src.config import EmbeddingsConfig
from src.embeddings.base import Vector

logger = logging.getLogger(__name__)


class LocalEmbedder:
    """sentence-transformers on CPU. No API, no cost, no rate limit.

    The model is loaded lazily: importing this module must stay cheap so the
    ingest CLI and Streamlit can import freely without paying ~5s of model load.
    """

    def __init__(self, cfg: EmbeddingsConfig) -> None:
        self._cfg = cfg
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model %s (first run downloads it)", self._cfg.model)
            self._model = SentenceTransformer(self._cfg.model, device="cpu")
        return self._model

    @property
    def dimension(self) -> int:
        # Renamed in sentence-transformers 5.x; keep working on both.
        getter = getattr(self.model, "get_embedding_dimension", None) or \
            self.model.get_sentence_embedding_dimension
        return int(getter())

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            batch_size=self._cfg.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> Vector:
        # bge-* is asymmetric; the prefix belongs on the query side only.
        prefixed = f"{self._cfg.query_prefix}{text}"
        vector = self.model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vector.tolist()


def build_embedder(cfg: EmbeddingsConfig) -> LocalEmbedder:
    if cfg.provider != "local":
        raise ValueError(f"Unsupported embeddings provider: {cfg.provider!r} (only 'local')")
    return LocalEmbedder(cfg)
