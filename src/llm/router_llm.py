"""Role -> deployment routing, backed by litellm.Router.

This is a thin facade. Call sites ask for a *role* ("routing", "reasoning",
"generation") and never name a model, which is what let the whole Llama family
be swapped out for gpt-oss with a config edit. LiteLLM supplies the parts that
used to be hand-rolled here: ordered fallbacks deeper than two, per-deployment
cooldowns, retry with backoff, and per-call cost.

Keys are read from the environment by litellm at call time and never stored on
a config object or logged.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from src.config import LLMConfig
from src.llm.base import LLMError, LLMResponse, Message

logger = logging.getLogger(__name__)

# litellm looks for specific variable names; accept the ones already in .env.
_KEY_ALIASES = {"NVIDIA_NIM_API_KEY": "NVIDIA_API_KEY", "GEMINI_API_KEY": "GOOGLE_API_KEY"}


def _alias_env_keys() -> None:
    for expected, existing in _KEY_ALIASES.items():
        if not os.environ.get(expected) and os.environ.get(existing):
            os.environ[expected] = os.environ[existing]


class AllProvidersExhausted(LLMError):
    """Every deployment for a role failed."""


@dataclass
class CallOutcome:
    """What actually served the call. Surfaced in the UI."""

    provider: str
    model: str
    fell_back: bool = False
    cost: float | None = None
    attempts: list[str] = field(default_factory=list)


class LLMRouter:
    def __init__(self, cfg: LLMConfig) -> None:
        _alias_env_keys()
        self._cfg = cfg
        self.last_outcome: CallOutcome | None = None
        self._primary = {role: deps[0].model for role, deps in cfg.roles.items() if deps}

        import litellm
        from litellm import Router

        litellm.suppress_debug_info = True
        # Params like reasoning_effort are OpenAI/Groq-specific and are rejected
        # outright by Mistral, Cohere and NIM. Without this, those deployments
        # fail, trip allowed_fails, get cooled down, and the role silently
        # collapses onto whichever provider happens to accept the param.
        litellm.drop_params = True
        model_list = [
            {
                "model_name": role,
                "litellm_params": {"model": dep.model, "order": dep.order},
                "model_info": cfg.cost_overrides.get(dep.model, {}),
            }
            for role, deps in cfg.roles.items()
            for dep in deps
        ]
        if not model_list:
            raise ValueError("No LLM deployments configured under llm.roles")

        r = cfg.router
        self._router = Router(
            model_list=model_list,
            routing_strategy=r.routing_strategy,
            num_retries=r.num_retries,
            allowed_fails=r.allowed_fails,
            cooldown_time=r.cooldown_time,
            timeout=r.timeout,
        )
        logger.info(
            "LLM gateway ready: %s",
            ", ".join(f"{role}={len(deps)} deployment(s)" for role, deps in cfg.roles.items()),
        )

    # -- public API --------------------------------------------------------
    def for_role(self, role: str) -> "RoleClient":
        return RoleClient(self, role)

    def complete(
        self,
        role: str,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        self._cfg.role(role)  # raises a clear error for an unknown role
        payload = [{"role": m.role, "content": m.content} for m in messages]
        extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}

        try:
            response = self._router.completion(
                model=role, messages=payload, temperature=temperature,
                max_tokens=max_tokens, **extra,
            )
        except Exception as exc:  # litellm raises OpenAI-shaped errors
            raise AllProvidersExhausted(
                f"role={role} exhausted all deployments: {type(exc).__name__}: {exc}"
            ) from exc

        served = response.model or ""
        cost = (response._hidden_params or {}).get("response_cost")
        # Not tracked during retry any more: derived by comparing what served
        # the call against the role's first-choice deployment.
        fell_back = bool(served) and not self._primary.get(role, "").endswith(served)
        if fell_back:
            logger.warning("role=%s served by %s (not primary %s)",
                           role, served, self._primary.get(role))

        self.last_outcome = CallOutcome(
            provider=served.split("/", 1)[0], model=served,
            fell_back=fell_back, cost=cost,
        )
        choice = response.choices[0]
        return LLMResponse(
            text=(choice.message.content or "").strip(),
            model=served,
            provider=served.split("/", 1)[0],
            finish_reason=str(choice.finish_reason or ""),
            cost=cost,
            raw=response,
        )


@dataclass
class RoleClient:
    """So call sites read as llm.for_role('generation').complete(...)."""

    router: LLMRouter
    role: str

    def complete(self, messages: list[Message], **kw) -> LLMResponse:
        return self.router.complete(self.role, messages, **kw)

    def ask(self, system: str, user: str, **kw) -> str:
        return self.complete([Message("system", system), Message("user", user)], **kw).text
