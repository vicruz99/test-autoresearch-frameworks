"""Experiment configuration dataclasses, loaded from YAML files.

Example usage::

    config = ExperimentConfig.from_yaml("configs/example.yaml")
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclasses.dataclass
class VllmConfig:
    """Connection settings for the vLLM OpenAI-compatible server."""

    base_url: str = "http://localhost:8000/v1"
    api_key: str = "dummy"
    max_concurrency: int = 8
    max_retries: int = 3


@dataclasses.dataclass
class ModelConfig:
    """A single model entry — just its name (used verbatim as the vLLM model ID)."""

    name: str


@dataclasses.dataclass
class SamplerConfig:
    """Configuration for one sampler/strategy.

    Attributes:
        type: ``"random"`` or ``"iterative"``.
        mode: ``"full_rewrite"`` or ``"edit"``.
        num_samples: Number of independent samples (random sampler).
        num_steps: Number of sequential refinement steps (iterative sampler).
        samples_per_step: LLM calls per step (iterative sampler).
    """

    type: Literal["random", "iterative"]
    mode: Literal["full_rewrite", "edit"] = "full_rewrite"
    num_samples: int = 50
    num_steps: int = 10
    samples_per_step: int = 5


@dataclasses.dataclass
class RunsConfig:
    """Controls statistical reproducibility — one run per seed."""

    seeds: list[int] = dataclasses.field(default_factory=lambda: [42])


@dataclasses.dataclass
class LLMParams:
    """Generation hyper-parameters forwarded to every LLM call."""

    temperature: float = 0.8
    max_tokens: int = 4096
    top_p: float = 0.95


@dataclasses.dataclass
class EvaluationConfig:
    """Evaluation execution settings."""

    max_workers: int = 8


@dataclasses.dataclass
class ExperimentConfig:
    """Root configuration object for a full benchmark run.

    Attributes:
        vllm: vLLM server connection settings.
        models: List of models to evaluate.
        problems: List of problem IDs or ``["all"]``.
        samplers: List of sampler configs to evaluate.
        runs: Seeds for reproducible statistical runs.
        llm_params: LLM generation parameters.
        evaluation: Evaluation execution settings.
        output_dir: Directory where JSON results are written.
    """

    vllm: VllmConfig
    models: list[ModelConfig]
    problems: list[str]
    samplers: list[SamplerConfig]
    runs: RunsConfig
    llm_params: LLMParams
    evaluation: EvaluationConfig
    output_dir: str = "experiments/results"

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load an :class:`ExperimentConfig` from a YAML file."""
        with open(path, "r") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh)
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> "ExperimentConfig":
        vllm_raw = d.get("vllm", {})
        vllm = VllmConfig(
            base_url=vllm_raw.get("base_url", VllmConfig.base_url),
            api_key=vllm_raw.get("api_key", VllmConfig.api_key),
            max_concurrency=int(vllm_raw.get("max_concurrency", VllmConfig.max_concurrency)),
            max_retries=int(vllm_raw.get("max_retries", VllmConfig.max_retries)),
        )

        models = [ModelConfig(name=m["name"]) for m in d.get("models", [])]

        problems = d.get("problems", [])

        samplers: list[SamplerConfig] = []
        for s in d.get("samplers", []):
            samplers.append(
                SamplerConfig(
                    type=s["type"],
                    mode=s.get("mode", "full_rewrite"),
                    num_samples=int(s.get("num_samples", 50)),
                    num_steps=int(s.get("num_steps", 10)),
                    samples_per_step=int(s.get("samples_per_step", 5)),
                )
            )

        runs_raw = d.get("runs", {})
        runs = RunsConfig(seeds=runs_raw.get("seeds", [42]))

        llm_raw = d.get("llm_params", {})
        llm_params = LLMParams(
            temperature=float(llm_raw.get("temperature", 0.8)),
            max_tokens=int(llm_raw.get("max_tokens", 4096)),
            top_p=float(llm_raw.get("top_p", 0.95)),
        )

        eval_raw = d.get("evaluation", {})
        evaluation = EvaluationConfig(
            max_workers=int(eval_raw.get("max_workers", 8)),
        )

        return cls(
            vllm=vllm,
            models=models,
            problems=problems,
            samplers=samplers,
            runs=runs,
            llm_params=llm_params,
            evaluation=evaluation,
            output_dir=d.get("output_dir", "experiments/results"),
        )
