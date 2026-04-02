"""Async LLM client — thin wrapper around openai.AsyncOpenAI.

Features
--------
* Semaphore-based concurrency limiting so we don't flood the vLLM server.
* Exponential back-off with jitter on transient failures.
* Convenience method :meth:`LLMClient.batch_complete` for sending many
  requests concurrently and collecting results in order.

Example usage::

    client = LLMClient(base_url="http://localhost:8000/v1", api_key="dummy")
    responses = await client.batch_complete(
        model="gpt-oss-120b",
        messages_list=[messages1, messages2, ...],
        temperature=0.8,
        max_tokens=4096,
    )
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import random
import time
from typing import Any

import openai

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class CompletionResult:
    """Rich result from a single chat completion request.

    Attributes:
        content: The assistant message text.
        reasoning_content: Chain-of-thought reasoning produced before the
            final answer (empty string when not available).
            vLLM exposes this as the ``reasoning`` field.
        prompt_tokens: Number of input tokens consumed.
        reasoning_tokens: Number of tokens spent on reasoning.
            When the API provides ``completion_tokens_details.reasoning_tokens``
            (e.g. OpenAI o-series models) the exact value is used.
            For vLLM, which does not break down completion tokens, this is
            estimated from the character-length ratio of ``reasoning_content``
            to total output text (``None`` when no reasoning content present).
        completion_tokens: Total output tokens including reasoning.
        total_tokens: Sum of prompt + completion tokens.
    """

    content: str
    reasoning_content: str = ""
    prompt_tokens: int | None = None
    reasoning_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class LLMClient:
    """Async OpenAI-compatible client with batching and retry logic.

    Parameters
    ----------
    base_url:
        The vLLM server base URL (e.g. ``http://localhost:8000/v1``).
    api_key:
        API key; vLLM typically accepts any non-empty string.
    max_concurrency:
        Maximum number of simultaneous in-flight requests.
    max_retries:
        Number of retry attempts on transient errors before giving up.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "dummy",
        max_concurrency: int = 8,
        max_retries: int = 3,
    ) -> None:
        self._client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._max_retries = max_retries

    async def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.8,
        max_tokens: int = 4096,
        top_p: float = 0.95,
        **kwargs: Any,
    ) -> CompletionResult:
        """Send a single chat completion request and return a :class:`CompletionResult`.

        Retries on transient failures with exponential back-off and jitter.

        Parameters
        ----------
        model:
            Model identifier (passed verbatim to vLLM).
        messages:
            OpenAI-format message list.
        temperature:
            Sampling temperature.
        max_tokens:
            Maximum tokens to generate.
        top_p:
            Top-p nucleus sampling parameter.
        **kwargs:
            Additional kwargs forwarded to the completions API.
            If ``reasoning_effort`` is present it is sent via
            ``extra_body`` rather than as a top-level parameter.

        Returns
        -------
        CompletionResult
            The assistant message content, reasoning content, and token usage.

        Raises
        ------
        RuntimeError
            If all retry attempts are exhausted.
        """
        # Pop reasoning_effort and pass via extra_body for vLLM compatibility
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        extra_body = kwargs.pop("extra_body", None) or {}
        if reasoning_effort is not None:
            extra_body["reasoning_effort"] = reasoning_effort

        api_kwargs: dict[str, Any] = dict(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs,
        )
        if extra_body:
            api_kwargs["extra_body"] = extra_body

        async with self._semaphore:
            last_exc: Exception | None = None
            for attempt in range(self._max_retries + 1):
                try:
                    response = await self._client.chat.completions.create(**api_kwargs)
                    message = response.choices[0].message
                    content = message.content or ""
                    # vLLM uses "reasoning" while OpenAI uses "reasoning_content"
                    reasoning_content = (
                        getattr(message, "reasoning_content", None)
                        or getattr(message, "reasoning", None)
                        or ""
                    )

                    # Extract token usage
                    prompt_tokens: int | None = None
                    reasoning_tokens: int | None = None
                    completion_tokens: int | None = None
                    total_tokens: int | None = None
                    usage = getattr(response, "usage", None)
                    if usage is not None:
                        prompt_tokens = getattr(usage, "prompt_tokens", None)
                        completion_tokens = getattr(usage, "completion_tokens", None)
                        total_tokens = getattr(usage, "total_tokens", None)
                        details = getattr(usage, "completion_tokens_details", None)
                        if details is not None:
                            reasoning_tokens = getattr(details, "reasoning_tokens", None)

                    # vLLM does not populate completion_tokens_details, so
                    # reasoning_tokens will be None.  Estimate it from the
                    # character-length ratio when reasoning content is present.
                    if reasoning_tokens is None and reasoning_content and completion_tokens:
                        total_chars = len(reasoning_content) + len(content)
                        if total_chars > 0:
                            reasoning_tokens = round(
                                completion_tokens * len(reasoning_content) / total_chars
                            )

                    return CompletionResult(
                        content=content,
                        reasoning_content=reasoning_content,
                        prompt_tokens=prompt_tokens,
                        reasoning_tokens=reasoning_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                    )
                except openai.RateLimitError as exc:
                    last_exc = exc
                    wait = _backoff(attempt)
                    logger.warning("Rate limit hit (attempt %d/%d); retrying in %.1fs", attempt + 1, self._max_retries, wait)
                    await asyncio.sleep(wait)
                except openai.APIStatusError as exc:
                    last_exc = exc
                    wait = _backoff(attempt)
                    logger.warning(
                        "API error %s (attempt %d/%d); retrying in %.1fs",
                        exc.status_code,
                        attempt + 1,
                        self._max_retries,
                        wait,
                    )
                    await asyncio.sleep(wait)
                except Exception as exc:
                    last_exc = exc
                    logger.error("Unexpected error on attempt %d: %s", attempt + 1, exc)
                    if attempt >= self._max_retries:
                        break
                    await asyncio.sleep(_backoff(attempt))
            raise RuntimeError(f"LLM request failed after {self._max_retries + 1} attempts") from last_exc

    async def batch_complete(
        self,
        model: str,
        messages_list: list[list[dict[str, str]]],
        temperature: float = 0.8,
        max_tokens: int = 4096,
        top_p: float = 0.95,
        **kwargs: Any,
    ) -> list[CompletionResult | Exception]:
        """Send many completion requests concurrently and return results in order.

        Results corresponding to failed requests contain the :class:`Exception`
        rather than a :class:`CompletionResult` — callers should check with
        ``isinstance(r, Exception)``.

        Parameters
        ----------
        model:
            Model identifier.
        messages_list:
            List of message lists, one per request.
        temperature, max_tokens, top_p:
            Forwarded to each request.
        **kwargs:
            Additional kwargs forwarded to the completions API.

        Returns
        -------
        list[CompletionResult | Exception]
            Results in the same order as ``messages_list``.
        """
        tasks = [
            self.complete(model, msgs, temperature=temperature, max_tokens=max_tokens, top_p=top_p, **kwargs)
            for msgs in messages_list
        ]
        # gather preserves order and captures exceptions as values
        raw = await asyncio.gather(*tasks, return_exceptions=True)
        return list(raw)

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()

    async def __aenter__(self) -> "LLMClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()


def _backoff(attempt: int, base: float = 1.0, cap: float = 30.0) -> float:
    """Compute exponential back-off with full jitter.

    Parameters
    ----------
    attempt:
        Zero-based attempt index.
    base:
        Base delay in seconds.
    cap:
        Maximum delay in seconds.

    Returns
    -------
    float
        Seconds to sleep before the next attempt.
    """
    ceiling = min(cap, base * (2**attempt))
    return random.uniform(0, ceiling)
