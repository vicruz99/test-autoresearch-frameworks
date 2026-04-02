"""Tests for autoresearch_bench.config: ExperimentConfig loading."""

from __future__ import annotations

import textwrap

import pytest
import yaml

from autoresearch_bench.config import (
    EvaluationConfig,
    ExperimentConfig,
    LLMParams,
    ModelConfig,
    RunsConfig,
    SamplerConfig,
    VllmConfig,
)


# ---------------------------------------------------------------------------
# Full config dict
# ---------------------------------------------------------------------------

FULL_CONFIG_DICT = {
    "vllm": {
        "base_url": "http://localhost:8000/v1",
        "api_key": "secret",
        "max_concurrency": 16,
        "max_retries": 5,
    },
    "models": [
        {"name": "gpt-oss-120b"},
        {"name": "Qwen/Qwen3.5-32B"},
    ],
    "problems": ["combinatorics/cap_set", "graphs/max_clique"],
    "samplers": [
        {"type": "random", "mode": "full_rewrite", "num_samples": 20},
        {"type": "iterative", "mode": "edit", "num_steps": 5, "samples_per_step": 3},
    ],
    "runs": {"seeds": [1, 2, 3]},
    "llm_params": {"temperature": 0.7, "max_tokens": 2048, "top_p": 0.9},
    "evaluation": {"max_workers": 4},
    "output_dir": "out/results",
}


class TestExperimentConfigFromDict:
    """Tests for ExperimentConfig._from_dict()."""

    def test_vllm_config_populated(self):
        """VllmConfig fields are parsed from the 'vllm' key."""
        cfg = ExperimentConfig._from_dict(FULL_CONFIG_DICT)
        assert isinstance(cfg.vllm, VllmConfig)
        assert cfg.vllm.base_url == "http://localhost:8000/v1"
        assert cfg.vllm.api_key == "secret"
        assert cfg.vllm.max_concurrency == 16
        assert cfg.vllm.max_retries == 5

    def test_models_populated(self):
        """Models list is parsed into ModelConfig instances."""
        cfg = ExperimentConfig._from_dict(FULL_CONFIG_DICT)
        assert len(cfg.models) == 2
        assert all(isinstance(m, ModelConfig) for m in cfg.models)
        assert cfg.models[0].name == "gpt-oss-120b"
        assert cfg.models[1].name == "Qwen/Qwen3.5-32B"

    def test_problems_populated(self):
        """Problems list is preserved as-is."""
        cfg = ExperimentConfig._from_dict(FULL_CONFIG_DICT)
        assert cfg.problems == ["combinatorics/cap_set", "graphs/max_clique"]

    def test_samplers_populated(self):
        """Samplers are parsed into SamplerConfig instances with correct types."""
        cfg = ExperimentConfig._from_dict(FULL_CONFIG_DICT)
        assert len(cfg.samplers) == 2
        assert all(isinstance(s, SamplerConfig) for s in cfg.samplers)
        rand = cfg.samplers[0]
        assert rand.type == "random"
        assert rand.mode == "full_rewrite"
        assert rand.num_samples == 20
        iterative = cfg.samplers[1]
        assert iterative.type == "iterative"
        assert iterative.mode == "edit"
        assert iterative.num_steps == 5
        assert iterative.samples_per_step == 3

    def test_runs_config_populated(self):
        """RunsConfig seeds are parsed correctly."""
        cfg = ExperimentConfig._from_dict(FULL_CONFIG_DICT)
        assert isinstance(cfg.runs, RunsConfig)
        assert cfg.runs.seeds == [1, 2, 3]

    def test_llm_params_populated(self):
        """LLMParams fields are parsed with correct types."""
        cfg = ExperimentConfig._from_dict(FULL_CONFIG_DICT)
        assert isinstance(cfg.llm_params, LLMParams)
        assert cfg.llm_params.temperature == pytest.approx(0.7)
        assert cfg.llm_params.max_tokens == 2048
        assert cfg.llm_params.top_p == pytest.approx(0.9)
        assert cfg.llm_params.reasoning_effort is None

    def test_evaluation_config_populated(self):
        """EvaluationConfig is parsed correctly."""
        cfg = ExperimentConfig._from_dict(FULL_CONFIG_DICT)
        assert isinstance(cfg.evaluation, EvaluationConfig)
        assert cfg.evaluation.max_workers == 4

    def test_output_dir_populated(self):
        """output_dir is read from the dict."""
        cfg = ExperimentConfig._from_dict(FULL_CONFIG_DICT)
        assert cfg.output_dir == "out/results"


class TestExperimentConfigDefaults:
    """Tests that minimal configs get sensible defaults."""

    def test_empty_dict_uses_defaults(self):
        """An entirely empty dict produces an ExperimentConfig with defaults."""
        cfg = ExperimentConfig._from_dict({})
        assert cfg.vllm.base_url == "http://localhost:8000/v1"
        assert cfg.vllm.api_key == "dummy"
        assert cfg.vllm.max_concurrency == 8
        assert cfg.vllm.max_retries == 3
        assert cfg.models == []
        assert cfg.problems == []
        assert cfg.samplers == []
        assert cfg.runs.seeds == [42]
        assert cfg.llm_params.temperature == pytest.approx(0.8)
        assert cfg.llm_params.max_tokens == 4096
        assert cfg.llm_params.top_p == pytest.approx(0.95)
        assert cfg.llm_params.reasoning_effort is None
        assert cfg.evaluation.max_workers == 8
        assert cfg.output_dir == "experiments/results"

    def test_sampler_defaults(self):
        """A sampler dict with only 'type' gets all other fields defaulted."""
        d = {"samplers": [{"type": "random"}]}
        cfg = ExperimentConfig._from_dict(d)
        s = cfg.samplers[0]
        assert s.mode == "full_rewrite"
        assert s.num_samples == 50
        assert s.num_steps == 10
        assert s.samples_per_step == 5

    def test_partial_vllm_config(self):
        """Partial vllm dict only overrides the provided fields."""
        d = {"vllm": {"api_key": "my-key"}}
        cfg = ExperimentConfig._from_dict(d)
        assert cfg.vllm.api_key == "my-key"
        assert cfg.vllm.base_url == "http://localhost:8000/v1"

    def test_reasoning_effort_parsed_when_set(self):
        """reasoning_effort is parsed from llm_params when provided."""
        d = {"llm_params": {"reasoning_effort": "medium"}}
        cfg = ExperimentConfig._from_dict(d)
        assert cfg.llm_params.reasoning_effort == "medium"

    def test_reasoning_effort_none_when_absent(self):
        """reasoning_effort defaults to None when not in YAML."""
        d = {"llm_params": {"temperature": 0.5}}
        cfg = ExperimentConfig._from_dict(d)
        assert cfg.llm_params.reasoning_effort is None


class TestExperimentConfigFromYaml:
    """Tests for ExperimentConfig.from_yaml()."""

    def test_from_yaml_roundtrip(self, tmp_path):
        """Writing a YAML file and loading it produces the expected config."""
        yaml_content = textwrap.dedent("""\
            vllm:
              base_url: "http://localhost:9000/v1"
              api_key: "test-key"
              max_concurrency: 4
              max_retries: 2
            models:
              - name: "gpt-oss-120b"
            problems:
              - "combinatorics/cap_set"
            samplers:
              - type: "random"
                num_samples: 10
            runs:
              seeds: [42, 99]
            llm_params:
              temperature: 0.5
              max_tokens: 1024
              top_p: 1.0
            evaluation:
              max_workers: 2
            output_dir: "tmp/results"
        """)
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content)

        cfg = ExperimentConfig.from_yaml(config_file)

        assert cfg.vllm.base_url == "http://localhost:9000/v1"
        assert cfg.vllm.api_key == "test-key"
        assert cfg.models[0].name == "gpt-oss-120b"
        assert cfg.problems == ["combinatorics/cap_set"]
        assert cfg.samplers[0].num_samples == 10
        assert cfg.runs.seeds == [42, 99]
        assert cfg.llm_params.temperature == pytest.approx(0.5)
        assert cfg.output_dir == "tmp/results"

    def test_from_yaml_example_config(self):
        """The bundled example.yaml loads without errors."""
        import pathlib
        repo_root = pathlib.Path(__file__).parent.parent
        example = repo_root / "configs" / "example.yaml"
        if not example.exists():
            pytest.skip("configs/example.yaml not found")
        cfg = ExperimentConfig.from_yaml(example)
        assert isinstance(cfg, ExperimentConfig)
        assert len(cfg.models) > 0
        assert len(cfg.samplers) > 0
