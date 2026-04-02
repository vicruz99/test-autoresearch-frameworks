"""Tests for autoresearch_bench.samplers.iterative_sampler: IterativeSampler."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from autoresearch_problems import EvalResult

from autoresearch_bench.llm.client import CompletionResult
from autoresearch_bench.prompts.builder import PromptBuilder
from autoresearch_bench.results import RunResult
from autoresearch_bench.samplers.iterative_sampler import IterativeSampler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cr(content: str) -> CompletionResult:
    """Shorthand to create a CompletionResult."""
    return CompletionResult(content=content)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sampler(
    mock_client: MagicMock,
    mode: str = "full_rewrite",
    num_steps: int = 2,
    samples_per_step: int = 2,
) -> IterativeSampler:
    """Construct an IterativeSampler wired up with mock dependencies."""
    return IterativeSampler(
        num_steps=num_steps,
        samples_per_step=samples_per_step,
        client=mock_client,
        model="gpt-oss-120b",
        prompt_builder=PromptBuilder(mode=mode),
        llm_params={"temperature": 0.8, "max_tokens": 256, "top_p": 0.95},
        eval_max_workers=1,
    )


def _make_eval_result(score: float, valid: bool = True) -> EvalResult:
    return EvalResult(score=score, valid=valid, execution_time=0.01, error="", metrics={})


# ---------------------------------------------------------------------------
# IterativeSampler.run tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestIterativeSamplerRun:
    """Tests for IterativeSampler.run()."""

    async def test_returns_run_result(self, sample_spec, mock_llm_client):
        """run() returns a RunResult instance."""
        code_block = _cr("```python\ndef solve(n): return [[0]*n]\n```")
        mock_llm_client.batch_complete = AsyncMock(return_value=[code_block, code_block])
        eval_results = [_make_eval_result(10.0), _make_eval_result(20.0)]

        sampler = _make_sampler(mock_llm_client, num_steps=1, samples_per_step=2)
        with (
            patch("autoresearch_bench.samplers.iterative_sampler.execute_and_evaluate", return_value=_make_eval_result(5.0)),
            patch("autoresearch_bench.samplers.base.execute_and_evaluate_batch", return_value=eval_results),
        ):
            result = await sampler.run(spec=sample_spec, seed=42, config_dict={})

        assert isinstance(result, RunResult)

    async def test_total_steps_is_num_steps_times_samples_per_step(self, sample_spec, mock_llm_client):
        """Total steps = num_steps * samples_per_step."""
        code_block = _cr("```python\ndef solve(n): return []\n```")
        num_steps, sps = 3, 2
        mock_llm_client.batch_complete = AsyncMock(return_value=[code_block] * sps)
        eval_per_step = [_make_eval_result(float(i)) for i in range(sps)]

        sampler = _make_sampler(mock_llm_client, num_steps=num_steps, samples_per_step=sps)
        with (
            patch("autoresearch_bench.samplers.iterative_sampler.execute_and_evaluate", return_value=_make_eval_result(0.0)),
            patch("autoresearch_bench.samplers.base.execute_and_evaluate_batch", return_value=eval_per_step),
        ):
            result = await sampler.run(spec=sample_spec, seed=42, config_dict={})

        assert len(result.steps) == num_steps * sps

    async def test_program_updates_when_better_score_found(self, sample_spec, mock_llm_client):
        """current_program is updated when a step yields a higher score."""
        # Step 1: return a better-scoring program
        better_code = _cr("```python\ndef solve(n): return [[1]*n]\n```")
        worse_code = _cr("```python\ndef solve(n): return []\n```")

        # 2 steps, 1 sample each
        # Step 1 yields score 100 (improvement over initial 5.0)
        # Step 2 yields score 50 (no improvement over 100)
        call_count = 0

        async def side_effect_batch(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [better_code]
            return [worse_code]

        mock_llm_client.batch_complete = side_effect_batch
        step1_eval = [_make_eval_result(100.0)]
        step2_eval = [_make_eval_result(50.0)]

        sampler = _make_sampler(mock_llm_client, num_steps=2, samples_per_step=1)
        eval_call_count = 0

        def side_effect_eval_batch(spec, codes, **kwargs):
            nonlocal eval_call_count
            eval_call_count += 1
            if eval_call_count == 1:
                return step1_eval
            return step2_eval

        with (
            patch("autoresearch_bench.samplers.iterative_sampler.execute_and_evaluate", return_value=_make_eval_result(5.0)),
            patch("autoresearch_bench.samplers.base.execute_and_evaluate_batch", side_effect=side_effect_eval_batch),
        ):
            result = await sampler.run(spec=sample_spec, seed=42, config_dict={})

        # Best overall should be 100
        assert result.best_score == pytest.approx(100.0)

    async def test_best_score_does_not_decrease(self, sample_spec, mock_llm_client):
        """best_score in the RunResult never decreases between steps."""
        code_block = _cr("```python\ndef solve(n): return []\n```")
        # Step 1 yields 100, step 2 yields 10 (should not replace best)
        mock_llm_client.batch_complete = AsyncMock(return_value=[code_block])
        eval_call_count = 0

        def side_effect_eval_batch(spec, codes, **kwargs):
            nonlocal eval_call_count
            eval_call_count += 1
            if eval_call_count == 1:
                return [_make_eval_result(100.0)]
            return [_make_eval_result(10.0)]

        sampler = _make_sampler(mock_llm_client, num_steps=2, samples_per_step=1)
        with (
            patch("autoresearch_bench.samplers.iterative_sampler.execute_and_evaluate", return_value=_make_eval_result(5.0)),
            patch("autoresearch_bench.samplers.base.execute_and_evaluate_batch", side_effect=side_effect_eval_batch),
        ):
            result = await sampler.run(spec=sample_spec, seed=42, config_dict={})

        assert result.best_score == pytest.approx(100.0)

    async def test_sampler_type_label(self, sample_spec, mock_llm_client):
        """RunResult has sampler_type='iterative'."""
        mock_llm_client.batch_complete = AsyncMock(return_value=[])
        sampler = _make_sampler(mock_llm_client, num_steps=1, samples_per_step=0)
        with (
            patch("autoresearch_bench.samplers.iterative_sampler.execute_and_evaluate", return_value=_make_eval_result(5.0)),
            patch("autoresearch_bench.samplers.base.execute_and_evaluate_batch", return_value=[]),
        ):
            result = await sampler.run(spec=sample_spec, seed=42, config_dict={})

        assert result.sampler_type == "iterative"

    async def test_handles_all_none_codes_gracefully(self, sample_spec, mock_llm_client):
        """run() handles steps where all LLM responses contain no code block."""
        mock_llm_client.batch_complete = AsyncMock(return_value=[_cr("no code here"), _cr("also no code")])

        sampler = _make_sampler(mock_llm_client, num_steps=1, samples_per_step=2)
        with (
            patch("autoresearch_bench.samplers.iterative_sampler.execute_and_evaluate", return_value=_make_eval_result(5.0)),
            patch("autoresearch_bench.samplers.base.execute_and_evaluate_batch", return_value=[]),
        ):
            result = await sampler.run(spec=sample_spec, seed=42, config_dict={})

        # All steps should be invalid
        for step in result.steps:
            assert step.valid is False

    async def test_minimize_problem_updates_on_lower_score(self, sample_spec_minimize, mock_llm_client):
        """For minimize problems, best_score decreases over steps."""
        code_block = _cr("```python\ndef solve(): return 0.1\n```")
        mock_llm_client.batch_complete = AsyncMock(return_value=[code_block])
        eval_call_count = 0

        def side_effect_eval_batch(spec, codes, **kwargs):
            nonlocal eval_call_count
            eval_call_count += 1
            # Return a lower score on step 1 (improvement), higher on step 2
            if eval_call_count == 1:
                return [_make_eval_result(0.5)]
            return [_make_eval_result(2.0)]

        sampler = _make_sampler(mock_llm_client, num_steps=2, samples_per_step=1)
        with (
            patch("autoresearch_bench.samplers.iterative_sampler.execute_and_evaluate", return_value=_make_eval_result(1.0)),
            patch("autoresearch_bench.samplers.base.execute_and_evaluate_batch", side_effect=side_effect_eval_batch),
        ):
            result = await sampler.run(spec=sample_spec_minimize, seed=42, config_dict={})

        # Best should be lowest (0.5), not the step-2 value (2.0)
        assert result.best_score == pytest.approx(0.5)
