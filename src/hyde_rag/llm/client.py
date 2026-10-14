"""Provider-agnostic LLM client (D-LLM-1).

One interface, :class:`LLMClient`. The default implementation talks to any
OpenAI-compatible endpoint: Groq's free plan by default, or local Ollama.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from hyde_rag.llm.quota import assert_free_tier

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def default_base_url() -> str:
    """Endpoint used for cache keys when reading results without a client (no key needed)."""
    return os.environ.get("OPENAI_COMPAT_BASE_URL") or GROQ_BASE_URL


@dataclass(frozen=True)
class LLMRequest:
    model: str
    messages: tuple[tuple[str, str], ...]  # ((role, content), ...) - hashable and order-stable
    max_tokens: int = 512
    temperature: float | None = None
    top_p: float | None = None
    stop: tuple[str, ...] | None = None
    seed: int | None = None
    reasoning_effort: str | None = None
    json_schema: dict[str, Any] | None = None  # structured output (judge)

    def params(self) -> dict[str, Any]:
        """Canonical, JSON-serialisable request parameters (used for cache keys)."""
        return {
            "model": self.model,
            "messages": [list(m) for m in self.messages],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stop": list(self.stop) if self.stop else None,
            "seed": self.seed,
            "reasoning_effort": self.reasoning_effort,
            "json_schema": self.json_schema,
        }


@dataclass
class LLMResponse:
    text: str
    finish_reason: str
    model: str
    prompt_tokens: int
    completion_tokens: int  # includes reasoning tokens where the backend counts them
    latency_s: float
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMError(RuntimeError):
    """Base class for backend-neutral LLM errors raised by clients."""


class RateLimited(LLMError):
    """HTTP 429: per-minute or daily free-tier limit reached."""

    def __init__(self, retry_after_s: float | None, message: str = ""):
        super().__init__(message or f"rate limited (retry after {retry_after_s}s)")
        self.retry_after_s = retry_after_s


class TransientLLMError(LLMError):
    """Network problem, timeout or 5xx: safe to retry."""


class PermanentLLMError(LLMError):
    """4xx other than 429 (bad parameters, unknown model, auth): don't retry."""


class LLMClient(Protocol):
    base_url: str

    def complete(self, request: LLMRequest) -> LLMResponse: ...


def _retry_after(headers: Any) -> float | None:
    try:
        value = headers.get("retry-after")
        return float(value) if value is not None else None
    except (AttributeError, TypeError, ValueError):
        return None


class OpenAICompatClient:
    """Chat completions against Groq (free plan) or any OpenAI-compatible server."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None, timeout: float = 120.0):
        from openai import OpenAI

        self.base_url = base_url or default_base_url()
        key = api_key or os.environ.get("OPENAI_COMPAT_API_KEY") or os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "No API key. Put a free Groq key in .env as GROQ_API_KEY "
                "(or set OPENAI_COMPAT_BASE_URL/OPENAI_COMPAT_API_KEY for Ollama)."
            )
        # Retries are handled by the bulk runner, which knows about free-tier limits.
        self._client = OpenAI(base_url=self.base_url, api_key=key, timeout=timeout, max_retries=0)
        self._checked_free_tier = False

    def complete(self, request: LLMRequest) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": [{"role": r, "content": c} for r, c in request.messages],
            "max_completion_tokens": request.max_tokens,
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.stop:
            kwargs["stop"] = list(request.stop)
        if request.seed is not None:
            kwargs["seed"] = request.seed
        extra: dict[str, Any] = {}
        if request.reasoning_effort is not None:
            # Passed through untouched: Groq accepts values (e.g. "none" for Qwen3) the SDK
            # doesn't enumerate.
            extra["reasoning_effort"] = request.reasoning_effort
        if request.json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": request.json_schema, "strict": True},
            }
        if extra:
            kwargs["extra_body"] = extra

        import openai

        start = time.perf_counter()
        try:
            raw = self._client.chat.completions.with_raw_response.create(**kwargs)
        except openai.RateLimitError as e:
            raise RateLimited(_retry_after(e.response.headers), str(e)) from e
        except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError) as e:
            raise TransientLLMError(str(e)) from e
        except openai.APIStatusError as e:
            if e.status_code >= 500:
                raise TransientLLMError(str(e)) from e
            raise PermanentLLMError(f"{e.status_code}: {e.message}") from e
        latency = time.perf_counter() - start
        headers = {k.lower(): v for k, v in raw.headers.items()}
        if not self._checked_free_tier:
            assert_free_tier(headers)  # zero-spend guard (D-LLM-4)
            self._checked_free_tier = True
        completion = raw.parse()
        choice = completion.choices[0]
        usage = completion.usage
        return LLMResponse(
            text=(choice.message.content or "").strip(),
            finish_reason=choice.finish_reason or "",
            model=completion.model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_s=latency,
            headers={k: v for k, v in headers.items() if k.startswith(("x-ratelimit", "retry-after"))},
        )
