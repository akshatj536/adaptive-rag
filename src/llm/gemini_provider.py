from __future__ import annotations

import logging
import os

from src.llm.base import (
    LLMError,
    LLMProvider,
    LLMResponse,
    Message,
    MissingCredentialsError,
    ProviderError,
    RateLimitError,
    TransientError,
    parse_retry_after,
)

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self) -> None:
        self._client = None
        # Models observed to reject thinking_config. Learned at runtime.
        self._no_thinking_config: set[str] = set()

    @property
    def client(self):
        if self._client is None:
            key = os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise MissingCredentialsError("GOOGLE_API_KEY is not set")
            from google import genai

            self._client = genai.Client(api_key=key)
        return self._client

    def complete(
        self,
        messages: list[Message],
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        from google.genai import errors, types

        system = "\n\n".join(m.content for m in messages if m.role == "system") or None
        contents = [
            types.Content(
                role="model" if m.role == "assistant" else "user",
                parts=[types.Part(text=m.content)],
            )
            for m in messages
            if m.role != "system"
        ]

        config_kwargs = {
            "system_instruction": system,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        # Disabling thinking saves free-tier output tokens on grounded
        # extraction, but only some models accept it - 2.5 does, 3.x rejects it
        # with a 400. Rather than hardcode a model list that will go stale, try
        # it once and remember the answer per model.
        if model not in self._no_thinking_config:
            try:
                config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            except (AttributeError, TypeError):
                self._no_thinking_config.add(model)
                config_kwargs.pop("thinking_config", None)

        try:
            response = self._generate(model, contents, config_kwargs, types, errors)
        except errors.APIError as exc:
            raise _map_api_error(exc, model) from exc
        except LLMError:
            # Already classified (e.g. missing credentials). Re-wrapping it as
            # transient would burn the whole retry budget on a permanent error.
            raise
        except Exception as exc:  # transport-level failures surface untyped
            raise TransientError(f"gemini transport error on {model}: {exc}") from exc

        text = (getattr(response, "text", None) or "").strip()
        if not text:
            reason = _finish_reason(response)
            raise ProviderError(f"gemini returned no text on {model} (finish_reason={reason})")

        return LLMResponse(
            text=text,
            model=model,
            provider=self.name,
            finish_reason=_finish_reason(response),
            headers=_headers(response),
            raw=response,
        )


    def _generate(self, model: str, contents, config_kwargs: dict, types, errors):
        """Send the request, retrying once without thinking_config if the model
        rejects it. Keeps the unsupported-parameter case from looking like a
        real failure and triggering a pointless provider failover."""
        try:
            return self.client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )
        except errors.APIError as exc:
            code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
            if code != 400 or "thinking_config" not in config_kwargs:
                raise
            logger.info("%s rejects thinking_config; retrying without it", model)
            self._no_thinking_config.add(model)
            config_kwargs.pop("thinking_config")
            return self.client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )


def _map_api_error(exc, model: str):
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    message = getattr(exc, "message", str(exc))
    if code == 429:
        return RateLimitError(
            f"gemini rate limited on {model}: {message}",
            retry_after=_retry_after_from_error(exc),
        )
    if isinstance(code, int) and code >= 500:
        return TransientError(f"gemini {code} on {model}: {message}")
    return ProviderError(f"gemini {code} on {model}: {message}")


def _retry_after_from_error(exc) -> float | None:
    """Gemini reports backoff either as a header or inside error details as a
    RetryInfo entry like {'retryDelay': '31s'}."""
    headers = _headers(getattr(exc, "response", None))
    if delay := parse_retry_after(headers.get("retry-after")):
        return delay
    details = getattr(exc, "details", None) or {}
    if isinstance(details, dict):
        for entry in details.get("error", {}).get("details", []) or []:
            if isinstance(entry, dict) and "retryDelay" in entry:
                return parse_retry_after(entry["retryDelay"])
    return None


def _headers(obj) -> dict[str, str]:
    http = getattr(obj, "sdk_http_response", None) or getattr(obj, "response", None)
    headers = getattr(http, "headers", None) or getattr(obj, "headers", None)
    if not headers:
        return {}
    try:
        return {str(k).lower(): str(v) for k, v in dict(headers).items()}
    except (TypeError, ValueError):
        return {}


def _finish_reason(response) -> str:
    """Normalize to a bare token ("MAX_TOKENS", "STOP") - the SDK enum stringifies
    as "FinishReason.MAX_TOKENS", which no caller should have to parse."""
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return "no candidates"
    reason = getattr(candidates[0], "finish_reason", None)
    if reason is None:
        return ""
    return str(getattr(reason, "name", reason)).rsplit(".", 1)[-1]
