"""Shared pytest fixtures for autoresearch_bench tests.

All fixtures that create mock external dependencies (ProblemSpec, EvalResult,
LLMClient) are centralised here so individual test modules stay focused.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from autoresearch_problems import EvalResult, ProblemSpec

from autoresearch_bench.llm.client import LLMClient, CompletionResult
from autoresearch_bench.prompts.builder import PromptBuilder


# ---------------------------------------------------------------------------
# Problem / evaluation fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_spec() -> ProblemSpec:
    """A minimal ProblemSpec suitable for use in all sampler/prompt tests."""
    return ProblemSpec(
        name="cap_set",
        category="combinatorics",
        description="Find the largest cap set in F_3^n.",
        output_type="list[list[int]]",
        evaluator_code="def evaluate(output, **params): return {'score': len(output), 'valid': True}",
        evaluator_entrypoint="evaluate",
        evaluator_dependencies=[],
        parameters={"n": 6},
        timeout_seconds=10.0,
        maximize=True,
        known_best_score=112.0,
        initial_prompt="Improve the cap set solver.",
        initial_program="def solve(n):\n    return []\n",
        function_name="solve",
        source="test",
        tags=["combinatorics"],
    )


@pytest.fixture
def sample_spec_minimize() -> ProblemSpec:
    """A ProblemSpec where lower scores are better (maximize=False)."""
    return ProblemSpec(
        name="min_problem",
        category="test",
        description="Minimisation test problem.",
        output_type="float",
        evaluator_code="def evaluate(output, **params): return {'score': output, 'valid': True}",
        evaluator_entrypoint="evaluate",
        evaluator_dependencies=[],
        parameters={},
        timeout_seconds=5.0,
        maximize=False,
        initial_program="def solve(): return 1.0\n",
    )


@pytest.fixture
def good_eval_result() -> EvalResult:
    """A valid EvalResult with a positive score."""
    return EvalResult(score=42.0, valid=True, execution_time=0.1, error="", metrics={"size": 42})


@pytest.fixture
def bad_eval_result() -> EvalResult:
    """An invalid EvalResult (execution failure)."""
    return EvalResult(score=0.0, valid=False, execution_time=0.0, error="TimeoutError", metrics={})


# ---------------------------------------------------------------------------
# LLM client fixture
# ---------------------------------------------------------------------------

def _make_completion_result(content: str) -> CompletionResult:
    """Helper to create a CompletionResult with default token fields."""
    return CompletionResult(content=content)


@pytest.fixture
def mock_llm_client() -> MagicMock:
    """A MagicMock standing in for LLMClient with AsyncMock methods."""
    client = MagicMock(spec=LLMClient)
    client.complete = AsyncMock(
        return_value=_make_completion_result("```python\ndef solve(n):\n    return [[0]*n]\n```")
    )
    client.batch_complete = AsyncMock(
        return_value=[_make_completion_result("```python\ndef solve(n):\n    return [[0]*n]\n```")]
    )
    return client


# ---------------------------------------------------------------------------
# PromptBuilder fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def full_rewrite_builder() -> PromptBuilder:
    """A PromptBuilder in full_rewrite mode."""
    return PromptBuilder(mode="full_rewrite")


@pytest.fixture
def edit_builder() -> PromptBuilder:
    """A PromptBuilder in edit mode."""
    return PromptBuilder(mode="edit")
