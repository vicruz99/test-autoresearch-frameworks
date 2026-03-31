"""Tests for autoresearch_bench.samplers.random_sampler: RandomSampler."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from autoresearch_problems import EvalResult

from autoresearch_bench.llm.client import LLMClient
from autoresearch_bench.prompts.builder import PromptBuilder
from autoresearch_bench.results import RunResult
from autoresearch_bench.samplers.random_sampler import RandomSampler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sampler(
    mock_client: MagicMock,
    mode: str = "full_rewrite",
    num_samples: int = 3,
) -> RandomSampler:
    """Construct a RandomSampler wired up with mock dependencies."""
    return RandomSampler(
        num_samples=num_samples,
        client=mock_client,
        model="gpt-oss-120b",
        prompt_builder=PromptBuilder(mode=mode),
        llm_params={"temperature": 0.8, "max_tokens": 256, "top_p": 0.95},
        eval_max_workers=1,
    )


def _make_eval_result(score: float, valid: bool = True) -> EvalResult:
    return EvalResult(score=score, valid=valid, execution_time=0.01, error="", metrics={})


# ---------------------------------------------------------------------------
# RandomSampler.run tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRandomSamplerRun:
    """Tests for RandomSampler.run()."""

    async def test_returns_run_result(self, sample_spec, mock_llm_client):
        """run() returns a RunResult instance."""
        mock_llm_client.batch_complete = AsyncMock(
            return_value=["```python\ndef solve(n): return [[0]*n]\n```"] * 3
        )
        eval_results = [_make_eval_result(10.0), _make_eval_result(20.0), _make_eval_result(15.0)]

        sampler = _make_sampler(mock_llm_client, num_samples=3)
        with (
            patch("autoresearch_bench.samplers.random_sampler.execute_and_evaluate", return_value=_make_eval_result(5.0)),
            patch("autoresearch_bench.samplers.base.execute_and_evaluate_batch", return_value=eval_results),
        ):
            result = await sampler.run(spec=sample_spec, seed=42, config_dict={})

        assert isinstance(result, RunResult)

    async def test_best_score_is_maximum_for_maximize_problem(self, sample_spec, mock_llm_client):
        """best_score equals the highest valid score when maximize=True."""
        mock_llm_client.batch_complete = AsyncMock(
            return_value=["```python\ndef solve(n): return []\n```"] * 3
        )
        eval_results = [_make_eval_result(10.0), _make_eval_result(50.0), _make_eval_result(30.0)]

        sampler = _make_sampler(mock_llm_client, num_samples=3)
        with (
            patch("autoresearch_bench.samplers.random_sampler.execute_and_evaluate", return_value=_make_eval_result(5.0)),
            patch("autoresearch_bench.samplers.base.execute_and_evaluate_batch", return_value=eval_results),
        ):
            result = await sampler.run(spec=sample_spec, seed=42, config_dict={})

        assert result.best_score == pytest.approx(50.0)

    async def test_steps_count_matches_num_samples(self, sample_spec, mock_llm_client):
        """The number of steps in RunResult equals num_samples."""
        n = 4
        mock_llm_client.batch_complete = AsyncMock(
            return_value=["```python\ndef solve(n): return []\n```"] * n
        )
        eval_results = [_make_eval_result(float(i)) for i in range(n)]

        sampler = _make_sampler(mock_llm_client, num_samples=n)
        with (
            patch("autoresearch_bench.samplers.random_sampler.execute_and_evaluate", return_value=_make_eval_result(0.0)),
            patch("autoresearch_bench.samplers.base.execute_and_evaluate_batch", return_value=eval_results),
        ):
            result = await sampler.run(spec=sample_spec, seed=42, config_dict={})

        assert len(result.steps) == n

    async def test_handles_none_code_gracefully(self, sample_spec, mock_llm_client):
        """Candidates with no extracted code get score=None and valid=False."""
        # Return plain text with no code block
        mock_llm_client.batch_complete = AsyncMock(
            return_value=["no code block here"] * 2
        )

        sampler = _make_sampler(mock_llm_client, num_samples=2)
        with (
            patch("autoresearch_bench.samplers.random_sampler.execute_and_evaluate", return_value=_make_eval_result(5.0)),
            patch("autoresearch_bench.samplers.base.execute_and_evaluate_batch", return_value=[]),
        ):
            result = await sampler.run(spec=sample_spec, seed=42, config_dict={})

        # All steps should be invalid
        for step in result.steps:
            assert step.valid is False
            assert step.score is None

    async def test_sampler_type_label(self, sample_spec, mock_llm_client):
        """RunResult has sampler_type='random'."""
        mock_llm_client.batch_complete = AsyncMock(return_value=[])
        sampler = _make_sampler(mock_llm_client, num_samples=0)
        with (
            patch("autoresearch_bench.samplers.random_sampler.execute_and_evaluate", return_value=_make_eval_result(5.0)),
            patch("autoresearch_bench.samplers.base.execute_and_evaluate_batch", return_value=[]),
        ):
            result = await sampler.run(spec=sample_spec, seed=1, config_dict={})

        assert result.sampler_type == "random"

    async def test_problem_id_includes_category(self, sample_spec, mock_llm_client):
        """RunResult.problem is 'category/name' when category is set."""
        mock_llm_client.batch_complete = AsyncMock(return_value=[])
        sampler = _make_sampler(mock_llm_client, num_samples=0)
        with (
            patch("autoresearch_bench.samplers.random_sampler.execute_and_evaluate", return_value=_make_eval_result(0.0)),
            patch("autoresearch_bench.samplers.base.execute_and_evaluate_batch", return_value=[]),
        ):
            result = await sampler.run(spec=sample_spec, seed=42, config_dict={})

        assert result.problem == "combinatorics/cap_set"

    async def test_initial_score_is_set(self, sample_spec, mock_llm_client):
        """initial_score in RunResult reflects the initial program evaluation."""
        mock_llm_client.batch_complete = AsyncMock(return_value=[])
        sampler = _make_sampler(mock_llm_client, num_samples=0)
        with (
            patch("autoresearch_bench.samplers.random_sampler.execute_and_evaluate", return_value=_make_eval_result(7.0)),
            patch("autoresearch_bench.samplers.base.execute_and_evaluate_batch", return_value=[]),
        ):
            result = await sampler.run(spec=sample_spec, seed=42, config_dict={})

        assert result.initial_score == pytest.approx(7.0)

    async def test_config_dict_stored_in_result(self, sample_spec, mock_llm_client):
        """The config_dict passed to run() is stored in the RunResult."""
        mock_llm_client.batch_complete = AsyncMock(return_value=[])
        sampler = _make_sampler(mock_llm_client, num_samples=0)
        cfg = {"output_dir": "test_output"}
        with (
            patch("autoresearch_bench.samplers.random_sampler.execute_and_evaluate", return_value=_make_eval_result(0.0)),
            patch("autoresearch_bench.samplers.base.execute_and_evaluate_batch", return_value=[]),
        ):
            result = await sampler.run(spec=sample_spec, seed=42, config_dict=cfg)

        assert result.config_dict == cfg


# ---------------------------------------------------------------------------
# BaseSampler._best tests
# ---------------------------------------------------------------------------

class TestBaseSamplerBest:
    """Tests for BaseSampler._best helper via RandomSampler."""

    def _make_sampler(self):
        client = MagicMock(spec=LLMClient)
        return RandomSampler(
            num_samples=1,
            client=client,
            model="test",
            prompt_builder=PromptBuilder(mode="full_rewrite"),
            llm_params={},
        )

    def test_best_maximize_returns_highest_score(self):
        """_best returns the highest score when maximize=True."""
        sampler = self._make_sampler()
        candidates = [(10.0, "a"), (50.0, "b"), (30.0, "c")]
        score, code = sampler._best(candidates, maximize=True)
        assert score == pytest.approx(50.0)
        assert code == "b"

    def test_best_minimize_returns_lowest_score(self):
        """_best returns the lowest score when maximize=False."""
        sampler = self._make_sampler()
        candidates = [(10.0, "a"), (50.0, "b"), (5.0, "c")]
        score, code = sampler._best(candidates, maximize=False)
        assert score == pytest.approx(5.0)
        assert code == "c"

    def test_best_ignores_none_scores(self):
        """_best ignores candidates with None scores."""
        sampler = self._make_sampler()
        candidates = [(None, "a"), (20.0, "b"), (None, "c")]
        score, code = sampler._best(candidates, maximize=True)
        assert score == pytest.approx(20.0)
        assert code == "b"

    def test_best_all_none_returns_first_candidate(self):
        """When all scores are None, _best returns the first candidate."""
        sampler = self._make_sampler()
        candidates = [(None, "fallback"), (None, "other")]
        score, code = sampler._best(candidates, maximize=True)
        assert score is None
        assert code == "fallback"

    def test_best_empty_list_returns_none_empty(self):
        """An empty candidate list returns (None, '')."""
        sampler = self._make_sampler()
        score, code = sampler._best([], maximize=True)
        assert score is None
        assert code == ""
