"""Role -> provider resolution, retry with backoff, and cross-provider fallback.

Every LLM call in the system goes through here. Design rules:
  * No rate-limit numbers are hardcoded. Quota state comes from response
    headers (x-ratelimit-*) and Retry-After only.
  * A long Retry-After means a daily cap, not a burst - fail over instead of
    sleeping, so one provider's exhausted quota never stalls the app.
  * API keys are never logged; log lines carry provider and model names only.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field

from src.config import LLMConfig, RoleConfig
from src.llm.base import (
    LLMError,
    LLMProvider,
    LLMResponse,
    Message,
    MissingCredentialsError,
    ProviderError,
    RateLimitError,
    RequestTooLargeError,
    TransientError,
    parse_retry_after,
)

logger = logging.getLogger(__name__)

# Sleeping longer than this means waiting out a daily cap. Fail over instead.
MAX_SLEEP_S = 30.0

# Floor for the shrink-on-413 retry; below this an answer is not worth having.
_MIN_MAX_TOKENS = 512

# Used only when a role has no explicit fallback_provider/fallback_model.
# These are model *names*, not limits - nothing here encodes a quota.
DEFAULT_FALLBACK_MODELS: dict[str, dict[str, str]] = {
    "groq": {
        "routing": "openai/gpt-oss-20b",
        "reasoning": "openai/gpt-oss-120b",
        # The small model, not the large one: it has far more daily headroom,
        # and a fallback that is also capped is not a fallback.
        "generation": "openai/gpt-oss-20b",
    },
    "gemini": {
        "routing": "gemini-3.5-flash-lite",
        "reasoning": "gemini-3.5-flash",
        "generation": "gemini-3.6-flash",
    },
}

_OTHER_PROVIDER = {"groq": "gemini", "gemini": "groq"}


class AllProvidersExhausted(LLMError):
    """Both the primary and fallback provider failed for a role."""


@dataclass
class QuotaState:
    """Last known remaining quota for a provider, straight from its headers."""

    remaining_requests: float | None = None
    remaining_tokens: float | None = None
    blocked_until: float = 0.0  # monotonic timestamp

    def is_blocked(self) -> bool:
        return time.monotonic() < self.blocked_until

    def block_for(self, seconds: float) -> None:
        self.blocked_until = max(self.blocked_until, time.monotonic() + seconds)


@dataclass
class Attempt:
    provider: str
    model: str
    error: str


@dataclass
class CallOutcome:
    """Returned alongside the response so the UI can surface what happened."""

    provider: str
    model: str
    fell_back: bool = False
    attempts: list[Attempt] = field(default_factory=list)


class LLMRouter:
    def __init__(self, cfg: LLMConfig) -> None:
        self._cfg = cfg
        self._providers: dict[str, LLMProvider] = {}
        self._quota: dict[str, QuotaState] = {}
        self.last_outcome: CallOutcome | None = None

    # -- provider registry -------------------------------------------------
    def provider(self, name: str) -> LLMProvider:
        if name not in self._providers:
            if name == "groq":
                from src.llm.groq_provider import GroqProvider

                self._providers[name] = GroqProvider()
            elif name == "gemini":
                from src.llm.gemini_provider import GeminiProvider

                self._providers[name] = GeminiProvider()
            else:
                raise ProviderError(f"Unknown LLM provider: {name!r}")
        return self._providers[name]

    def quota(self, provider: str) -> QuotaState:
        return self._quota.setdefault(provider, QuotaState())

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
        role_cfg = self._cfg.role(role)
        targets = self._targets(role, role_cfg)
        attempts: list[Attempt] = []

        for index, (provider_name, model) in enumerate(targets):
            is_fallback = index > 0
            if is_fallback:
                logger.warning(
                    "role=%s falling back %s -> %s (model=%s)",
                    role, targets[0][0], provider_name, model,
                )
            try:
                response = self._call_with_retry(
                    provider_name, model, messages, temperature, max_tokens,
                    attempts, reasoning_effort,
                )
            except LLMError:
                continue
            self.last_outcome = CallOutcome(
                provider=provider_name, model=model, fell_back=is_fallback, attempts=attempts
            )
            return response

        detail = "; ".join(f"{a.provider}/{a.model}: {a.error}" for a in attempts) or "no attempts"
        raise AllProvidersExhausted(f"role={role} exhausted all providers. {detail}")

    # -- internals ---------------------------------------------------------
    def _targets(self, role: str, role_cfg: RoleConfig) -> list[tuple[str, str]]:
        targets = [(role_cfg.provider, role_cfg.model)]
        if not self._cfg.retry.fallback:
            return targets
        provider = role_cfg.fallback_provider or _OTHER_PROVIDER.get(role_cfg.provider)
        if not provider:
            return targets
        model = role_cfg.fallback_model or DEFAULT_FALLBACK_MODELS.get(provider, {}).get(role)
        if model:
            targets.append((provider, model))
        return targets

    def _call_with_retry(
        self,
        provider_name: str,
        model: str,
        messages: list[Message],
        temperature: float,
        max_tokens: int | None,
        attempts: list[Attempt],
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        quota = self.quota(provider_name)
        if quota.is_blocked():
            wait = quota.blocked_until - time.monotonic()
            attempts.append(
                Attempt(provider_name, model, f"quota blocked for another {wait:.0f}s")
            )
            raise RateLimitError(f"{provider_name} quota blocked")

        provider = self.provider(provider_name)
        max_attempts = max(1, self._cfg.retry.max_attempts)
        base_delay = self._cfg.retry.base_delay_s

        for attempt in range(max_attempts):
            try:
                response = provider.complete(
                    messages, model, temperature=temperature, max_tokens=max_tokens,
                    reasoning_effort=reasoning_effort,
                )
            except RateLimitError as exc:
                delay = exc.retry_after
                self._record_rate_limit(provider_name, delay)
                attempts.append(Attempt(provider_name, model, f"429 (retry_after={delay})"))
                # A long Retry-After is a daily cap - don't sit on it.
                if delay is not None and delay > MAX_SLEEP_S:
                    logger.warning(
                        "%s asked for %.0fs backoff on %s; failing over instead",
                        provider_name, delay, model,
                    )
                    raise
                if attempt == max_attempts - 1:
                    raise
                self._sleep(delay, base_delay, attempt)
            except TransientError as exc:
                attempts.append(Attempt(provider_name, model, f"transient: {exc}"))
                if attempt == max_attempts - 1:
                    raise
                self._sleep(None, base_delay, attempt)
            except RequestTooLargeError as exc:
                # The reserved output budget is counted against the provider's
                # per-minute allowance. Shrinking it can make the same prompt
                # fit, which beats failing the query outright.
                if max_tokens and max_tokens > _MIN_MAX_TOKENS and attempt < max_attempts - 1:
                    max_tokens = max(_MIN_MAX_TOKENS, max_tokens // 2)
                    logger.warning(
                        "%s rejected the request as too large; retrying with "
                        "max_tokens=%d", provider_name, max_tokens,
                    )
                    attempts.append(Attempt(provider_name, model, "413, shrank max_tokens"))
                    continue
                attempts.append(Attempt(provider_name, model, f"too large: {exc}"))
                raise
            except MissingCredentialsError as exc:
                attempts.append(Attempt(provider_name, model, str(exc)))
                raise
            except ProviderError as exc:
                # Permanent for this provider; retrying changes nothing.
                attempts.append(Attempt(provider_name, model, f"permanent: {exc}"))
                raise
            else:
                self._record_headers(provider_name, response.headers)
                return response

        raise AllProvidersExhausted(f"{provider_name}/{model} exhausted retry budget")

    def _sleep(self, retry_after: float | None, base_delay: float, attempt: int) -> None:
        delay = retry_after if retry_after is not None else base_delay * (2**attempt)
        delay = min(delay, MAX_SLEEP_S)
        delay += random.uniform(0, min(1.0, delay * 0.25))  # jitter
        logger.info("Backing off %.2fs (attempt %d)", delay, attempt + 1)
        time.sleep(delay)

    def _record_rate_limit(self, provider_name: str, retry_after: float | None) -> None:
        if retry_after and retry_after > MAX_SLEEP_S:
            # Remember the cap so subsequent calls skip straight to fallback
            # instead of each paying a 429 round trip.
            self.quota(provider_name).block_for(retry_after)

    def _record_headers(self, provider_name: str, headers: dict[str, str]) -> None:
        """Treat the provider's headers as the source of truth for quota."""
        if not headers:
            return
        quota = self.quota(provider_name)
        requests_left = _as_float(headers.get("x-ratelimit-remaining-requests"))
        tokens_left = _as_float(headers.get("x-ratelimit-remaining-tokens"))
        if requests_left is not None:
            quota.remaining_requests = requests_left
        if tokens_left is not None:
            quota.remaining_tokens = tokens_left

        # Exhausted: block until the provider's own reset window elapses.
        for value, reset_key in (
            (requests_left, "x-ratelimit-reset-requests"),
            (tokens_left, "x-ratelimit-reset-tokens"),
        ):
            if value is not None and value <= 0:
                reset = parse_retry_after(headers.get(reset_key)) or 60.0
                logger.warning(
                    "%s reports 0 remaining (%s); blocking it for %.0fs",
                    provider_name, reset_key, reset,
                )
                quota.block_for(reset)


def _as_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


@dataclass
class RoleClient:
    """Thin handle so call sites read as llm.for_role('generation').complete(...)."""

    router: LLMRouter
    role: str

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        return self.router.complete(
            self.role, messages, temperature=temperature, max_tokens=max_tokens
        )

    def ask(
        self, system: str, user: str, *, temperature: float = 0.0, max_tokens: int | None = None
    ) -> str:
        response = self.complete(
            [Message("system", system), Message("user", user)],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.text
