"""Tests for autoresearch_bench.runner: Runner, _build_sampler, helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from autoresearch_bench.config import ExperimentConfig, SamplerConfig
from autoresearch_bench.llm.client import LLMClient
from autoresearch_bench.prompts.builder import PromptBuilder
from autoresearch_bench.runner import _build_sampler, _config_to_dict, Runner
from autoresearch_bench.samplers.iterative_sampler import IterativeSampler
from autoresearch_bench.samplers.random_sampler import RandomSampler


# ---------------------------------------------------------------------------
# Minimal ExperimentConfig factory
# ---------------------------------------------------------------------------

def _minimal_config(problems: list[str] | None = None) -> ExperimentConfig:
    """Return an ExperimentConfig with sensible test defaults."""
    if problems is None:
        problems = ["combinatorics/cap_set"]
    return ExperimentConfig._from_dict({
        "models": [{"name": "gpt-oss-120b"}],
        "problems": problems,
        "samplers": [{"type": "random", "num_samples": 2}],
        "runs": {"seeds": [42]},
    })


# ---------------------------------------------------------------------------
# _build_sampler tests
# ---------------------------------------------------------------------------

class TestBuildSampler:
    """Tests for the _build_sampler module-level helper."""

    def _common_kwargs(self):
        return dict(
            client=MagicMock(spec=LLMClient),
            model="gpt-oss-120b",
            prompt_builder=PromptBuilder(mode="full_rewrite"),
            llm_params={"temperature": 0.8, "max_tokens": 256, "top_p": 0.95},
            eval_max_workers=4,
        )

    def test_random_sampler_config_returns_random_sampler(self):
        """A 'random' SamplerConfig produces a RandomSampler."""
        cfg = SamplerConfig(type="random", num_samples=10)
        sampler = _build_sampler(sampler_cfg=cfg, **self._common_kwargs())
        assert isinstance(sampler, RandomSampler)

    def test_random_sampler_has_correct_num_samples(self):
        """RandomSampler is initialised with the configured num_samples."""
        cfg = SamplerConfig(type="random", num_samples=25)
        sampler = _build_sampler(sampler_cfg=cfg, **self._common_kwargs())
        assert sampler.num_samples == 25

    def test_iterative_sampler_config_returns_iterative_sampler(self):
        """An 'iterative' SamplerConfig produces an IterativeSampler."""
        cfg = SamplerConfig(type="iterative", num_steps=5, samples_per_step=3)
        sampler = _build_sampler(sampler_cfg=cfg, **self._common_kwargs())
        assert isinstance(sampler, IterativeSampler)

    def test_iterative_sampler_has_correct_params(self):
        """IterativeSampler is initialised with num_steps and samples_per_step."""
        cfg = SamplerConfig(type="iterative", num_steps=7, samples_per_step=4)
        sampler = _build_sampler(sampler_cfg=cfg, **self._common_kwargs())
        assert sampler.num_steps == 7
        assert sampler.samples_per_step == 4

    def test_unknown_type_raises_value_error(self):
        """An unknown sampler type raises ValueError."""
        cfg = SamplerConfig(type="unknown_type")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown sampler type"):
            _build_sampler(sampler_cfg=cfg, **self._common_kwargs())


# ---------------------------------------------------------------------------
# Runner._resolve_problems tests
# ---------------------------------------------------------------------------

class TestRunnerResolveProblems:
    """Tests for Runner._resolve_problems()."""

    def test_explicit_list_returned_as_is(self):
        """A non-['all'] list of problems is returned unchanged."""
        cfg = _minimal_config(problems=["combinatorics/cap_set", "graphs/clique"])
        runner = Runner(cfg)
        problems = runner._resolve_problems()
        assert problems == ["combinatorics/cap_set", "graphs/clique"]

    def test_all_keyword_calls_registry_list(self):
        """['all'] triggers registry.list_problems() and returns the full list."""
        cfg = _minimal_config(problems=["all"])
        runner = Runner(cfg)
        with patch("autoresearch_bench.runner.registry") as mock_registry:
            mock_registry.list_problems.return_value = ["p1", "p2", "p3"]
            problems = runner._resolve_problems()
        assert problems == ["p1", "p2", "p3"]
        mock_registry.list_problems.assert_called_once()

    def test_empty_problems_list(self):
        """An empty problems list is returned unchanged."""
        cfg = _minimal_config(problems=[])
        runner = Runner(cfg)
        assert runner._resolve_problems() == []


# ---------------------------------------------------------------------------
# Runner.dry_run tests
# ---------------------------------------------------------------------------

class TestRunnerDryRun:
    """Tests for Runner.dry_run()."""

    def test_dry_run_does_not_raise(self, capsys):
        """dry_run() completes without raising an exception."""
        cfg = _minimal_config(problems=["combinatorics/cap_set"])
        runner = Runner(cfg)
        runner.dry_run()  # Should not raise

    def test_dry_run_prints_problem_count(self, capsys):
        """dry_run() prints the number of problems to stdout."""
        cfg = _minimal_config(problems=["p1", "p2"])
        runner = Runner(cfg)
        runner.dry_run()
        out = capsys.readouterr().out
        assert "2" in out

    def test_dry_run_prints_total_runs(self, capsys):
        """dry_run() prints the total number of runs."""
        cfg = _minimal_config(problems=["p1"])
        # 1 problem * 1 model * 1 sampler * 1 seed = 1
        runner = Runner(cfg)
        runner.dry_run()
        out = capsys.readouterr().out
        assert "Total runs" in out

    def test_dry_run_prints_output_dir(self, capsys):
        """dry_run() prints the output directory."""
        cfg = _minimal_config()
        runner = Runner(cfg)
        runner.dry_run()
        out = capsys.readouterr().out
        assert cfg.output_dir in out


# ---------------------------------------------------------------------------
# Runner.from_config_file
# ---------------------------------------------------------------------------

class TestRunnerFromConfigFile:
    """Tests for Runner.from_config_file()."""

    def test_loads_from_yaml(self, tmp_path):
        """from_config_file() creates a Runner from a YAML file."""
        import textwrap
        yaml_content = textwrap.dedent("""\
            models:
              - name: "gpt-oss-120b"
            problems:
              - "combinatorics/cap_set"
            samplers:
              - type: "random"
            runs:
              seeds: [1]
        """)
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text(yaml_content)
        runner = Runner.from_config_file(cfg_file)
        assert isinstance(runner, Runner)
        assert runner.config.models[0].name == "gpt-oss-120b"


# ---------------------------------------------------------------------------
# _config_to_dict tests
# ---------------------------------------------------------------------------

class TestConfigToDict:
    """Tests for the _config_to_dict helper."""

    def test_returns_dict(self):
        """_config_to_dict() returns a plain dict."""
        cfg = _minimal_config()
        d = _config_to_dict(cfg)
        assert isinstance(d, dict)

    def test_contains_models_key(self):
        """The output dict contains a 'models' key."""
        cfg = _minimal_config()
        d = _config_to_dict(cfg)
        assert "models" in d

    def test_is_json_serialisable(self):
        """Output of _config_to_dict() can be serialised to JSON."""
        import json
        cfg = _minimal_config()
        d = _config_to_dict(cfg)
        assert json.dumps(d)  # Should not raise
