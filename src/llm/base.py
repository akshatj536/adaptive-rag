"""Provider-agnostic LLM interface.

The point of this module is error normalization: every provider maps its own
SDK exceptions onto the hierarchy below, so router_llm can implement retry and
fallback once without knowing which vendor it is talking to.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    # Normalized across providers. "length" means the output hit max_tokens and
    # was cut mid-sentence - callers must be able to tell that apart from a
    # complete answer.
    finish_reason: str = ""
    # Raw response headers, lowercased. Carries Retry-After and x-ratelimit-*
    # so quota decisions come from the provider, never from constants here.
    headers: dict[str, str] = field(default_factory=dict)
    cost: float | None = None      # USD for this call, from litellm's cost map
    raw: Any = None

    @property
    def truncated(self) -> bool:
        return self.finish_reason.lower() in {"length", "max_tokens"}


class LLMError(Exception):
    """Base for all provider errors."""


class RateLimitError(LLMError):
    """HTTP 429 / quota exhausted."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class TransientError(LLMError):
    """5xx, timeout, connection reset — worth retrying."""


class ProviderError(LLMError):
    """Permanent: bad request, auth failure, unknown model. Do not retry, but
    it may still be worth failing over to the other provider."""


class RequestTooLargeError(ProviderError):
    """HTTP 413. The prompt plus the reserved max_tokens exceeds what this
    provider allows per minute. Retrying identically fails identically, but
    retrying smaller can succeed - so this is worth distinguishing."""


class MissingCredentialsError(ProviderError):
    """API key absent from the environment."""


class LLMProvider(ABC):
    name: str

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        """reasoning_effort is a hint ("low"/"medium"/"high") for models that
        spend output budget on hidden reasoning. Providers that do not support
        it must ignore it rather than fail."""


def parse_retry_after(value: str | None) -> float | None:
    """Retry-After is seconds, but providers sometimes send '2.5s' or '1m30s'."""
    if not value:
        return None
    value = value.strip().lower()
    try:
        return float(value)
    except ValueError:
        pass
    total, number = 0.0, ""
    for ch in value:
        if ch.isdigit() or ch == ".":
            number += ch
        elif ch in {"m", "s", "h"} and number:
            total += float(number) * {"h": 3600, "m": 60, "s": 1}[ch]
            number = ""
    return total or None
