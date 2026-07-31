from __future__ import annotations

import os

from src.llm.base import (
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


class GroqProvider(LLMProvider):
    name = "groq"

    def __init__(self) -> None:
        self._client = None

    @property
    def client(self):
        if self._client is None:
            key = os.environ.get("GROQ_API_KEY")
            if not key:
                raise MissingCredentialsError("GROQ_API_KEY is not set")
            from groq import Groq

            self._client = Groq(api_key=key, max_retries=0)  # we own retry policy
        return self._client

    def complete(
        self,
        messages: list[Message],
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        import groq

        payload = [{"role": m.role, "content": m.content} for m in messages]
        try:
            # with_raw_response keeps the HTTP headers, which is where the
            # authoritative rate-limit state lives.
            raw = self.client.chat.completions.with_raw_response.create(
                model=model,
                messages=payload,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except groq.RateLimitError as exc:
            headers = _headers(getattr(exc, "response", None))
            raise RateLimitError(
                f"groq rate limited on {model}",
                retry_after=parse_retry_after(headers.get("retry-after")),
            ) from exc
        except (groq.APIConnectionError, groq.APITimeoutError) as exc:
            raise TransientError(f"groq connection error: {exc}") from exc
        except groq.APIStatusError as exc:
            if exc.status_code >= 500:
                raise TransientError(f"groq {exc.status_code} on {model}") from exc
            if exc.status_code == 413:
                raise RequestTooLargeError(f"groq 413 on {model}: {exc.message}") from exc
            raise ProviderError(f"groq {exc.status_code} on {model}: {exc.message}") from exc
        except groq.APIError as exc:
            raise ProviderError(f"groq error on {model}: {exc}") from exc

        headers = {k.lower(): v for k, v in raw.headers.items()}
        completion = raw.parse()
        choice = completion.choices[0]
        text = (choice.message.content or "").strip()
        return LLMResponse(
            text=text,
            model=model,
            provider=self.name,
            finish_reason=str(getattr(choice, "finish_reason", "") or ""),
            headers=headers,
            raw=completion,
        )


def _headers(response) -> dict[str, str]:
    if response is None or not getattr(response, "headers", None):
        return {}
    return {k.lower(): v for k, v in response.headers.items()}
