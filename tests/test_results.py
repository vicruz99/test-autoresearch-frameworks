"""Tests for autoresearch_bench.results: StepResult, RunResult, aggregate_results."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoresearch_bench.results import RunResult, StepResult, aggregate_results


# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------

def make_step(step: int = 0, score: float | None = 1.0, valid: bool = True) -> StepResult:
    """Create a minimal StepResult for testing."""
    return StepResult(
        step=step,
        prompt_messages=[{"role": "user", "content": "improve this"}],
        raw_response="```python\ndef solve(): return 1\n```",
        generated_code="def solve(): return 1",
        score=score,
        valid=valid,
        error="" if valid else "timeout",
        execution_time=0.05,
        metrics={"size": 1},
    )


def make_run_result(
    sampler_type: str = "random",
    sampler_mode: str = "full_rewrite",
    model: str = "gpt-oss-120b",
    problem: str = "combinatorics/cap_set",
    seed: int = 42,
    best_score: float | None = 42.0,
) -> RunResult:
    """Create a minimal RunResult for testing."""
    return RunResult(
        sampler_type=sampler_type,
        sampler_mode=sampler_mode,
        model=model,
        problem=problem,
        seed=seed,
        steps=[make_step(0, score=best_score)],
        best_score=best_score,
        best_code="def solve(): return 1",
        initial_score=10.0,
        config_dict={"vllm": {"base_url": "http://localhost:8000/v1"}},
        timestamp="2024-01-01T00:00:00Z",
    )


# ---------------------------------------------------------------------------
# StepResult tests
# ---------------------------------------------------------------------------

class TestStepResult:
    """Tests for StepResult serialisation."""

    def test_to_dict_contains_all_keys(self):
        """to_dict() returns a dict with all expected keys."""
        step = make_step()
        d = step.to_dict()
        expected_keys = {
            "step", "prompt_messages", "raw_response", "generated_code",
            "score", "valid", "error", "execution_time", "metrics",
            "reasoning_content",
        }
        assert expected_keys == set(d.keys())

    def test_to_dict_values_are_correct(self):
        """to_dict() preserves field values."""
        step = make_step(step=3, score=7.5, valid=True)
        d = step.to_dict()
        assert d["step"] == 3
        assert d["score"] == pytest.approx(7.5)
        assert d["valid"] is True

    def test_to_dict_none_score(self):
        """to_dict() handles None score without error."""
        step = make_step(score=None, valid=False)
        d = step.to_dict()
        assert d["score"] is None
        assert d["valid"] is False

    def test_to_dict_is_json_serialisable(self):
        """to_dict() output can be serialised to JSON without error."""
        step = make_step()
        assert json.dumps(step.to_dict())  # Should not raise


# ---------------------------------------------------------------------------
# RunResult tests
# ---------------------------------------------------------------------------

class TestRunResult:
    """Tests for RunResult serialisation and summary."""

    def test_to_dict_contains_all_keys(self):
        """to_dict() returns a dict with all expected top-level keys."""
        rr = make_run_result()
        d = rr.to_dict()
        expected_keys = {
            "sampler_type", "sampler_mode", "model", "problem", "seed",
            "best_score", "best_code", "initial_score", "config", "timestamp", "steps",
        }
        assert expected_keys == set(d.keys())

    def test_to_dict_steps_serialised(self):
        """to_dict() includes steps as a list of dicts."""
        rr = make_run_result()
        d = rr.to_dict()
        assert isinstance(d["steps"], list)
        assert len(d["steps"]) == 1
        assert isinstance(d["steps"][0], dict)

    def test_to_dict_is_json_serialisable(self):
        """to_dict() output can be serialised to JSON without error."""
        rr = make_run_result()
        assert json.dumps(rr.to_dict())

    def test_summary_excludes_code_and_steps(self):
        """summary() omits 'steps', 'best_code', and 'config' keys."""
        rr = make_run_result()
        s = rr.summary()
        assert "steps" not in s
        assert "best_code" not in s
        assert "config" not in s

    def test_summary_contains_key_fields(self):
        """summary() includes the key identity and metric fields."""
        rr = make_run_result(best_score=99.0, seed=7)
        s = rr.summary()
        assert s["best_score"] == pytest.approx(99.0)
        assert s["seed"] == 7
        assert s["sampler_type"] == "random"
        assert s["model"] == "gpt-oss-120b"
        assert s["num_steps"] == 1

    def test_save_creates_json_file(self, tmp_path):
        """save() writes a JSON file to the specified directory."""
        rr = make_run_result()
        path = rr.save(tmp_path)
        assert path.exists()
        assert path.suffix == ".json"

    def test_save_json_content_matches_to_dict(self, tmp_path):
        """The saved JSON content matches to_dict()."""
        rr = make_run_result()
        path = rr.save(tmp_path)
        with open(path) as fh:
            saved = json.load(fh)
        assert saved["sampler_type"] == rr.sampler_type
        assert saved["best_score"] == pytest.approx(rr.best_score)
        assert len(saved["steps"]) == 1

    def test_save_creates_output_dir(self, tmp_path):
        """save() creates nested directories if they don't exist."""
        rr = make_run_result()
        nested = tmp_path / "a" / "b" / "c"
        path = rr.save(nested)
        assert path.exists()

    def test_save_filename_contains_key_fields(self, tmp_path):
        """The filename encodes sampler_type, model, problem, and seed."""
        rr = make_run_result(sampler_type="iterative", seed=123)
        path = rr.save(tmp_path)
        fname = path.name
        assert "iterative" in fname
        assert "123" in fname

    def test_save_creates_programs_folder(self, tmp_path):
        """save() creates a per-experiment folder alongside the JSON file."""
        rr = make_run_result()
        path = rr.save(tmp_path)
        stem = path.stem
        programs_dir = tmp_path / stem
        assert programs_dir.is_dir()

    def test_save_writes_individual_program_files(self, tmp_path):
        """save() writes each step's generated_code as a separate .py file."""
        rr = make_run_result()
        path = rr.save(tmp_path)
        stem = path.stem
        programs_dir = tmp_path / stem
        py_files = list(programs_dir.glob("program_*.py"))
        # make_run_result() has one step with generated_code set
        assert len(py_files) == 1
        assert py_files[0].read_text() == rr.steps[0].generated_code

    def test_save_skips_empty_generated_code(self, tmp_path):
        """save() does not write a .py file for steps with empty generated_code."""
        from autoresearch_bench.results import StepResult
        rr = make_run_result()
        # Replace generated_code with an empty string
        rr.steps[0] = StepResult(
            step=0,
            prompt_messages=[],
            raw_response="",
            generated_code="",
            score=None,
            valid=False,
            error="no code",
            execution_time=0.0,
        )
        path = rr.save(tmp_path)
        stem = path.stem
        programs_dir = tmp_path / stem
        py_files = list(programs_dir.glob("program_*.py"))
        assert len(py_files) == 0


# ---------------------------------------------------------------------------
# aggregate_results tests
# ---------------------------------------------------------------------------

class TestAggregateResults:
    """Tests for aggregate_results()."""

    def test_single_group_mean_max(self):
        """A single group of results produces correct mean and max."""
        results = [
            make_run_result(seed=1, best_score=10.0),
            make_run_result(seed=2, best_score=20.0),
            make_run_result(seed=3, best_score=30.0),
        ]
        agg = aggregate_results(results)
        key = "random/full_rewrite/gpt-oss-120b/combinatorics/cap_set"
        assert key in agg
        assert agg[key]["mean"] == pytest.approx(20.0)
        assert agg[key]["max"] == pytest.approx(30.0)
        assert set(agg[key]["scores"]) == {10.0, 20.0, 30.0}

    def test_stdev_computed_for_multiple_scores(self):
        """stdev is included when there are at least two scores."""
        results = [
            make_run_result(seed=1, best_score=10.0),
            make_run_result(seed=2, best_score=20.0),
        ]
        agg = aggregate_results(results)
        key = "random/full_rewrite/gpt-oss-120b/combinatorics/cap_set"
        assert "stdev" in agg[key]

    def test_none_scores_excluded(self):
        """Results with best_score=None are excluded from aggregation."""
        results = [
            make_run_result(seed=1, best_score=None),
            make_run_result(seed=2, best_score=5.0),
        ]
        agg = aggregate_results(results)
        key = "random/full_rewrite/gpt-oss-120b/combinatorics/cap_set"
        assert agg[key]["scores"] == [5.0]

    def test_separate_groups_for_different_samplers(self):
        """Different sampler types produce separate group keys."""
        results = [
            make_run_result(sampler_type="random", seed=1, best_score=10.0),
            make_run_result(sampler_type="iterative", seed=1, best_score=20.0),
        ]
        agg = aggregate_results(results)
        assert len(agg) == 2

    def test_empty_list_returns_empty_dict(self):
        """An empty input list returns an empty aggregation dict."""
        assert aggregate_results([]) == {}

    def test_all_none_scores_excluded(self):
        """If all results have None scores, their key is absent from output."""
        results = [
            make_run_result(seed=1, best_score=None),
            make_run_result(seed=2, best_score=None),
        ]
        agg = aggregate_results(results)
        assert agg == {}
