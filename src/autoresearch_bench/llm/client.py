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
import logging
import random
import time
from typing import Any

import openai

logger = logging.getLogger(__name__)


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
    ) -> str:
        """Send a single chat completion request and return the response text.

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

        Returns
        -------
        str
            The assistant message content.

        Raises
        ------
        RuntimeError
            If all retry attempts are exhausted.
        """
        async with self._semaphore:
            last_exc: Exception | None = None
            for attempt in range(self._max_retries + 1):
                try:
                    response = await self._client.chat.completions.create(
                        model=model,
                        messages=messages,  # type: ignore[arg-type]
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        **kwargs,
                    )
                    content = response.choices[0].message.content or ""
                    return content
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
    ) -> list[str | Exception]:
        """Send many completion requests concurrently and return results in order.

        Results corresponding to failed requests contain the :class:`Exception`
        rather than a string — callers should check with ``isinstance(r, Exception)``.

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
        list[str | Exception]
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
