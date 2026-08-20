"""Loads config.yaml + .env into a typed, frozen config tree.

Secrets are deliberately absent from this tree: providers read their API keys
from the environment at call time so keys never sit on an object that might be
logged or reprd.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass(frozen=True)
class ParentConfig:
    granularity: str = "page"
    block_size: int = 2048
    block_overlap: int = 128
    fallback_page_chars: int = 3000
    max_metadata_chars: int = 12000


@dataclass(frozen=True)
class ChunkingConfig:
    strategy: str = "recursive"
    child_chunk_size: int = 512
    child_chunk_overlap: int = 64
    parent: ParentConfig = field(default_factory=ParentConfig)


@dataclass(frozen=True)
class EmbeddingsConfig:
    provider: str = "local"
    model: str = "BAAI/bge-small-en-v1.5"
    query_prefix: str = ""
    batch_size: int = 64


@dataclass(frozen=True)
class VectorStoreConfig:
    provider: str = "chroma"
    path: str = "./.chroma"
    collection: str = "rag_children"


@dataclass(frozen=True)
class Bm25Config:
    k1: float = 1.5
    b: float = 0.75
    rrf_k: int = 60


@dataclass(frozen=True)
class HydeConfig:
    role: str = "reasoning"
    max_tokens: int = 256


@dataclass(frozen=True)
class RerankConfig:
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 5
    fetch_multiplier: int = 4
    # Insertion order is preserved; it orders components within a stage.
    components: dict[str, bool] = field(default_factory=lambda: {"vector": True})
    bm25: Bm25Config = field(default_factory=Bm25Config)
    hyde: HydeConfig = field(default_factory=HydeConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)

    def enabled_components(self) -> list[str]:
        return [name for name, on in self.components.items() if on]

    def is_enabled(self, name: str) -> bool:
        return bool(self.components.get(name))


@dataclass(frozen=True)
class GenerationConfig:
    max_context_chars: int = 24000
    max_answer_tokens: int = 4096


@dataclass(frozen=True)
class Deployment:
    """One place a role can be served from.

    `model` is a LiteLLM model string ("provider/model", sometimes with a
    nested path like "groq/openai/gpt-oss-120b"). `order` is priority: the
    router exhausts order 1 before trying order 2, which preserves the
    primary-then-fallback behaviour while allowing chains deeper than two.
    """

    model: str
    order: int = 1

    @property
    def provider(self) -> str:
        return self.model.split("/", 1)[0]


@dataclass(frozen=True)
class RouterSettings:
    """Passed straight to litellm.Router."""

    routing_strategy: str = "simple-shuffle"
    num_retries: int = 2
    allowed_fails: int = 2
    cooldown_time: float = 60.0
    timeout: float = 60.0


@dataclass(frozen=True)
class LLMConfig:
    roles: dict[str, list[Deployment]] = field(default_factory=dict)
    router: RouterSettings = field(default_factory=RouterSettings)
    # Only needed for models missing from litellm's cost map (NIM chat models
    # have no entries), otherwise reported cost is silently zero.
    cost_overrides: dict[str, dict] = field(default_factory=dict)

    def role(self, name: str) -> list[Deployment]:
        try:
            return self.roles[name]
        except KeyError:
            known = ", ".join(sorted(self.roles)) or "<none>"
            raise KeyError(f"Unknown LLM role {name!r}. Configured roles: {known}") from None

    def primary(self, name: str) -> Deployment:
        return self.role(name)[0]


@dataclass(frozen=True)
class RouterConfig:
    enabled: bool = False
    default_path: str = "naive"
    paths: list[str] = field(default_factory=lambda: ["naive"])


@dataclass(frozen=True)
class AgenticConfig:
    max_loops: int = 3
    max_sub_questions: int = 4
    enable_decomposition: bool = True
    enable_self_check: bool = True
    max_self_checks: int = 1


@dataclass(frozen=True)
class Config:
    chunking: ChunkingConfig
    embeddings: EmbeddingsConfig
    vectorstore: VectorStoreConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig
    llm: LLMConfig
    router: RouterConfig
    agentic: AgenticConfig
    project_root: Path = PROJECT_ROOT

    def resolve_path(self, value: str) -> Path:
        """Resolve a config path relative to the project root, not the CWD.

        Without this, running `streamlit run app/streamlit_app.py` from a
        different directory would silently create a second, empty Chroma store.
        """
        p = Path(value).expanduser()
        return p if p.is_absolute() else (self.project_root / p).resolve()


def _build(raw: dict[str, Any]) -> Config:
    chunking_raw = dict(raw.get("chunking") or {})
    parent_raw = dict(chunking_raw.pop("parent", None) or {})

    llm_raw = dict(raw.get("llm") or {})
    router_settings = RouterSettings(**(llm_raw.pop("router", None) or {}))
    cost_overrides = dict(llm_raw.pop("cost_overrides", None) or {})
    roles = {
        role: [
            # Order defaults to list position, so config reads as a priority
            # list without repeating an explicit order on every entry.
            Deployment(order=i, **dep) if "order" not in dep else Deployment(**dep)
            for i, dep in enumerate(deployments, start=1)
        ]
        for role, deployments in (llm_raw.pop("roles", None) or {}).items()
    }

    retrieval_raw = dict(raw.get("retrieval") or {})
    retrieval = RetrievalConfig(
        bm25=Bm25Config(**(retrieval_raw.pop("bm25", None) or {})),
        hyde=HydeConfig(**(retrieval_raw.pop("hyde", None) or {})),
        rerank=RerankConfig(**(retrieval_raw.pop("rerank", None) or {})),
        **retrieval_raw,
    )

    return Config(
        chunking=ChunkingConfig(parent=ParentConfig(**parent_raw), **chunking_raw),
        embeddings=EmbeddingsConfig(**(raw.get("embeddings") or {})),
        vectorstore=VectorStoreConfig(**(raw.get("vectorstore") or {})),
        retrieval=retrieval,
        generation=GenerationConfig(**(raw.get("generation") or {})),
        llm=LLMConfig(roles=roles, router=router_settings,
                      cost_overrides=cost_overrides),
        router=RouterConfig(**(raw.get("router") or {})),
        agentic=AgenticConfig(**(raw.get("agentic") or {})),
    )


def load_config(path: Path | str | None = None) -> Config:
    """Load config.yaml and .env. Unknown YAML keys raise TypeError by design."""
    load_dotenv(PROJECT_ROOT / ".env")
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text()) or {}
    return _build(raw)


_cached: Config | None = None


def get_config(path: Path | str | None = None) -> Config:
    global _cached
    if _cached is None or path is not None:
        _cached = load_config(path)
    return _cached
