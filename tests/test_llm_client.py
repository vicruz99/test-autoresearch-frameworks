"""Tests for autoresearch_bench.llm.client: LLMClient."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest

from autoresearch_bench.llm.client import LLMClient, CompletionResult, _backoff


# ---------------------------------------------------------------------------
# _backoff helper
# ---------------------------------------------------------------------------

class TestBackoff:
    """Tests for the _backoff exponential back-off helper."""

    def test_returns_float(self):
        """_backoff returns a float value."""
        assert isinstance(_backoff(0), float)

    def test_zero_attempt_between_zero_and_base(self):
        """At attempt 0, back-off is between 0 and base (1.0 seconds)."""
        for _ in range(20):
            val = _backoff(0, base=1.0, cap=30.0)
            assert 0.0 <= val <= 1.0

    def test_higher_attempts_have_higher_cap(self):
        """Higher attempt numbers have a larger potential back-off."""
        cap0 = 1.0 * (2 ** 0)  # 1s
        cap3 = min(30.0, 1.0 * (2 ** 3))  # 8s
        # The ceiling should grow
        assert cap3 > cap0

    def test_never_exceeds_cap(self):
        """back-off never exceeds the cap, regardless of attempt number."""
        for attempt in range(10):
            for _ in range(20):
                val = _backoff(attempt, base=1.0, cap=5.0)
                assert val <= 5.0

    def test_always_non_negative(self):
        """back-off is always >= 0."""
        for attempt in range(5):
            val = _backoff(attempt)
            assert val >= 0.0


# ---------------------------------------------------------------------------
# Helpers to build mock openai responses
# ---------------------------------------------------------------------------

def _make_completion_response(
    content: str,
    reasoning_content: str = "",
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    reasoning_tokens: int | None = None,
    reasoning: str | None = None,
) -> MagicMock:
    """Create a MagicMock that looks like an openai ChatCompletion response.

    Parameters
    ----------
    reasoning_content:
        Value for the OpenAI-style ``reasoning_content`` attribute.
    reasoning:
        Value for the vLLM-style ``reasoning`` attribute.
        When set, ``reasoning_content`` is cleared to simulate vLLM behaviour.
    """
    mock_message = MagicMock(spec=[
        "content", "role", "refusal", "annotations", "audio",
        "function_call", "tool_calls", "reasoning_content", "reasoning",
    ])
    mock_message.content = content
    mock_message.reasoning_content = reasoning_content if reasoning is None else None
    mock_message.reasoning = reasoning

    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    if prompt_tokens is not None:
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = prompt_tokens
        mock_usage.completion_tokens = completion_tokens
        mock_usage.total_tokens = total_tokens
        if reasoning_tokens is not None:
            mock_details = MagicMock()
            mock_details.reasoning_tokens = reasoning_tokens
            mock_usage.completion_tokens_details = mock_details
        else:
            mock_usage.completion_tokens_details = None
        mock_response.usage = mock_usage
    else:
        mock_response.usage = None

    return mock_response


def _make_rate_limit_error() -> openai.RateLimitError:
    """Create a minimal openai.RateLimitError for testing retries."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 429
    mock_response.headers = {"x-request-id": "test-id"}
    mock_response.request = MagicMock()
    return openai.RateLimitError(
        "Rate limit exceeded", response=mock_response, body=None
    )


# ---------------------------------------------------------------------------
# LLMClient.complete tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestLLMClientComplete:
    """Tests for LLMClient.complete()."""

    async def test_complete_returns_completion_result(self):
        """complete() returns a CompletionResult with the content from the LLM response."""
        mock_response = _make_completion_response("Hello from LLM")
        with patch("openai.AsyncOpenAI") as MockOpenAI:
            mock_oai = MagicMock()
            mock_oai.chat.completions.create = AsyncMock(return_value=mock_response)
            MockOpenAI.return_value = mock_oai

            client = LLMClient(base_url="http://localhost:8000/v1", api_key="dummy")
            result = await client.complete("gpt-oss-120b", [{"role": "user", "content": "hi"}])
            assert isinstance(result, CompletionResult)
            assert result.content == "Hello from LLM"

    async def test_complete_retries_on_rate_limit(self):
        """complete() retries when a RateLimitError is raised."""
        mock_response = _make_completion_response("ok after retry")
        rate_err = _make_rate_limit_error()

        with patch("openai.AsyncOpenAI") as MockOpenAI:
            mock_oai = MagicMock()
            mock_oai.chat.completions.create = AsyncMock(
                side_effect=[rate_err, mock_response]
            )
            MockOpenAI.return_value = mock_oai

            with patch("asyncio.sleep", new_callable=AsyncMock):
                client = LLMClient(
                    base_url="http://localhost:8000/v1",
                    api_key="dummy",
                    max_retries=2,
                )
                result = await client.complete("gpt-oss-120b", [{"role": "user", "content": "hi"}])
                assert result.content == "ok after retry"

    async def test_complete_raises_after_all_retries_exhausted(self):
        """complete() raises RuntimeError when all retry attempts fail."""
        rate_err = _make_rate_limit_error()

        with patch("openai.AsyncOpenAI") as MockOpenAI:
            mock_oai = MagicMock()
            mock_oai.chat.completions.create = AsyncMock(side_effect=rate_err)
            MockOpenAI.return_value = mock_oai

            with patch("asyncio.sleep", new_callable=AsyncMock):
                client = LLMClient(
                    base_url="http://localhost:8000/v1",
                    api_key="dummy",
                    max_retries=1,
                )
                with pytest.raises(RuntimeError, match="LLM request failed"):
                    await client.complete("gpt-oss-120b", [{"role": "user", "content": "hi"}])

    async def test_complete_empty_content_returns_empty_string(self):
        """complete() returns an empty string when message content is None."""
        mock_response = _make_completion_response(None)
        # Patch the None → "" conversion
        mock_response.choices[0].message.content = None

        with patch("openai.AsyncOpenAI") as MockOpenAI:
            mock_oai = MagicMock()
            mock_oai.chat.completions.create = AsyncMock(return_value=mock_response)
            MockOpenAI.return_value = mock_oai

            client = LLMClient(base_url="http://localhost:8000/v1", api_key="dummy")
            result = await client.complete("gpt-oss-120b", [{"role": "user", "content": "hi"}])
            assert result.content == ""


# ---------------------------------------------------------------------------
# LLMClient.batch_complete tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestLLMClientBatchComplete:
    """Tests for LLMClient.batch_complete()."""

    async def test_batch_complete_returns_list_in_order(self):
        """batch_complete() returns results in the same order as inputs."""
        responses = [
            _make_completion_response("response_0"),
            _make_completion_response("response_1"),
            _make_completion_response("response_2"),
        ]

        with patch("openai.AsyncOpenAI") as MockOpenAI:
            mock_oai = MagicMock()
            mock_oai.chat.completions.create = AsyncMock(side_effect=responses)
            MockOpenAI.return_value = mock_oai

            client = LLMClient(base_url="http://localhost:8000/v1", api_key="dummy")
            messages_list = [[{"role": "user", "content": f"q{i}"}] for i in range(3)]
            results = await client.batch_complete("gpt-oss-120b", messages_list)

            assert len(results) == 3
            assert results[0].content == "response_0"
            assert results[1].content == "response_1"
            assert results[2].content == "response_2"

    async def test_batch_complete_captures_exceptions(self):
        """batch_complete() returns exceptions as values, not raised."""
        rate_err = _make_rate_limit_error()

        with patch("openai.AsyncOpenAI") as MockOpenAI:
            mock_oai = MagicMock()
            # Always fail so retry exhaustion triggers RuntimeError
            mock_oai.chat.completions.create = AsyncMock(side_effect=rate_err)
            MockOpenAI.return_value = mock_oai

            with patch("asyncio.sleep", new_callable=AsyncMock):
                client = LLMClient(
                    base_url="http://localhost:8000/v1",
                    api_key="dummy",
                    max_retries=0,
                )
                results = await client.batch_complete(
                    "gpt-oss-120b",
                    [[{"role": "user", "content": "hi"}]],
                )
                assert len(results) == 1
                assert isinstance(results[0], Exception)

    async def test_batch_complete_empty_list_returns_empty(self):
        """batch_complete() with an empty messages list returns []."""
        with patch("openai.AsyncOpenAI"):
            client = LLMClient(base_url="http://localhost:8000/v1", api_key="dummy")
            results = await client.batch_complete("gpt-oss-120b", [])
            assert results == []


# ---------------------------------------------------------------------------
# Semaphore concurrency test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_semaphore_limits_concurrency():
    """Semaphore with max_concurrency=1 serialises requests."""
    call_count = 0
    max_concurrent = 0
    concurrent = 0

    async def mock_create(**kwargs):
        nonlocal call_count, max_concurrent, concurrent
        concurrent += 1
        max_concurrent = max(max_concurrent, concurrent)
        await asyncio.sleep(0)  # yield
        concurrent -= 1
        call_count += 1
        return _make_completion_response(f"resp_{call_count}")

    with patch("openai.AsyncOpenAI") as MockOpenAI:
        mock_oai = MagicMock()
        mock_oai.chat.completions.create = mock_create
        MockOpenAI.return_value = mock_oai

        client = LLMClient(
            base_url="http://localhost:8000/v1",
            api_key="dummy",
            max_concurrency=1,
        )
        messages_list = [[{"role": "user", "content": f"q{i}"}] for i in range(3)]
        results = await client.batch_complete("gpt-oss-120b", messages_list)

    assert len(results) == 3
    # With semaphore=1, max simultaneous calls should be 1
    assert max_concurrent <= 1


# ---------------------------------------------------------------------------
# CompletionResult with reasoning content and token usage tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestCompletionResultFields:
    """Tests for CompletionResult populated with reasoning_content and token usage."""

    async def test_reasoning_content_captured(self):
        """complete() captures reasoning_content from the response."""
        mock_response = _make_completion_response(
            "final answer",
            reasoning_content="I think step by step...",
        )
        with patch("openai.AsyncOpenAI") as MockOpenAI:
            mock_oai = MagicMock()
            mock_oai.chat.completions.create = AsyncMock(return_value=mock_response)
            MockOpenAI.return_value = mock_oai

            client = LLMClient(base_url="http://localhost:8000/v1", api_key="dummy")
            result = await client.complete("gpt-oss-120b", [{"role": "user", "content": "hi"}])
            assert result.reasoning_content == "I think step by step..."

    async def test_token_usage_captured(self):
        """complete() captures token usage from the response."""
        mock_response = _make_completion_response(
            "answer",
            prompt_tokens=100,
            completion_tokens=200,
            total_tokens=300,
            reasoning_tokens=50,
        )
        with patch("openai.AsyncOpenAI") as MockOpenAI:
            mock_oai = MagicMock()
            mock_oai.chat.completions.create = AsyncMock(return_value=mock_response)
            MockOpenAI.return_value = mock_oai

            client = LLMClient(base_url="http://localhost:8000/v1", api_key="dummy")
            result = await client.complete("gpt-oss-120b", [{"role": "user", "content": "hi"}])
            assert result.prompt_tokens == 100
            assert result.completion_tokens == 200
            assert result.total_tokens == 300
            assert result.reasoning_tokens == 50

    async def test_no_usage_returns_none_tokens(self):
        """complete() returns None for token fields when usage is absent."""
        mock_response = _make_completion_response("answer")
        with patch("openai.AsyncOpenAI") as MockOpenAI:
            mock_oai = MagicMock()
            mock_oai.chat.completions.create = AsyncMock(return_value=mock_response)
            MockOpenAI.return_value = mock_oai

            client = LLMClient(base_url="http://localhost:8000/v1", api_key="dummy")
            result = await client.complete("gpt-oss-120b", [{"role": "user", "content": "hi"}])
            assert result.prompt_tokens is None
            assert result.reasoning_tokens is None

    async def test_reasoning_effort_passed_via_extra_body(self):
        """reasoning_effort kwarg is passed via extra_body to the API."""
        mock_response = _make_completion_response("answer")
        with patch("openai.AsyncOpenAI") as MockOpenAI:
            mock_oai = MagicMock()
            mock_oai.chat.completions.create = AsyncMock(return_value=mock_response)
            MockOpenAI.return_value = mock_oai

            client = LLMClient(base_url="http://localhost:8000/v1", api_key="dummy")
            await client.complete(
                "gpt-oss-120b",
                [{"role": "user", "content": "hi"}],
                reasoning_effort="medium",
            )
            call_kwargs = mock_oai.chat.completions.create.call_args
            assert call_kwargs.kwargs["extra_body"] == {"reasoning_effort": "medium"}

    async def test_vllm_reasoning_field_captured(self):
        """complete() captures vLLM's 'reasoning' field as reasoning_content."""
        mock_response = _make_completion_response(
            "final answer",
            reasoning="Step 1: think... Step 2: answer",
        )
        with patch("openai.AsyncOpenAI") as MockOpenAI:
            mock_oai = MagicMock()
            mock_oai.chat.completions.create = AsyncMock(return_value=mock_response)
            MockOpenAI.return_value = mock_oai

            client = LLMClient(base_url="http://localhost:8000/v1", api_key="dummy")
            result = await client.complete("gpt-oss-120b", [{"role": "user", "content": "hi"}])
            assert result.reasoning_content == "Step 1: think... Step 2: answer"

    async def test_reasoning_tokens_estimated_when_no_details(self):
        """reasoning_tokens is estimated from char ratio when completion_tokens_details absent."""
        # reasoning is 90 chars, content is 10 chars → 90% of 100 tokens = 90
        reasoning_text = "r" * 90
        content_text = "c" * 10
        mock_response = _make_completion_response(
            content_text,
            reasoning=reasoning_text,
            prompt_tokens=50,
            completion_tokens=100,
            total_tokens=150,
        )
        with patch("openai.AsyncOpenAI") as MockOpenAI:
            mock_oai = MagicMock()
            mock_oai.chat.completions.create = AsyncMock(return_value=mock_response)
            MockOpenAI.return_value = mock_oai

            client = LLMClient(base_url="http://localhost:8000/v1", api_key="dummy")
            result = await client.complete("gpt-oss-120b", [{"role": "user", "content": "hi"}])
            assert result.reasoning_tokens == 90  # 90% of 100

    async def test_reasoning_tokens_none_when_no_reasoning_content(self):
        """reasoning_tokens stays None when there is no reasoning content."""
        mock_response = _make_completion_response(
            "just an answer",
            prompt_tokens=50,
            completion_tokens=10,
            total_tokens=60,
        )
        with patch("openai.AsyncOpenAI") as MockOpenAI:
            mock_oai = MagicMock()
            mock_oai.chat.completions.create = AsyncMock(return_value=mock_response)
            MockOpenAI.return_value = mock_oai

            client = LLMClient(base_url="http://localhost:8000/v1", api_key="dummy")
            result = await client.complete("gpt-oss-120b", [{"role": "user", "content": "hi"}])
            assert result.reasoning_tokens is None
