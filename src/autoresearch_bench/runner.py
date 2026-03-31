"""Experiment runner — orchestrates config loading, problem loading, and sampler execution.

The :class:`Runner` is the top-level object used by the CLI.  It:

1. Reads an :class:`~autoresearch_bench.config.ExperimentConfig`.
2. Resolves which problems to run.
3. For each ``(sampler, model, problem, seed)`` combination, instantiates the
   appropriate sampler and calls :meth:`BaseSampler.run`.
4. Saves each :class:`~autoresearch_bench.results.RunResult` as JSON.
5. Prints a summary to stdout when finished.

Example usage::

    runner = Runner.from_config_file("configs/example.yaml")
    await runner.run()
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

from autoresearch_problems import registry, ProblemSpec

from autoresearch_bench.config import ExperimentConfig, SamplerConfig
from autoresearch_bench.llm.client import LLMClient
from autoresearch_bench.llm.models import resolve_model
from autoresearch_bench.prompts.builder import PromptBuilder
from autoresearch_bench.results import RunResult, aggregate_results
from autoresearch_bench.samplers.base import BaseSampler
from autoresearch_bench.samplers.random_sampler import RandomSampler
from autoresearch_bench.samplers.iterative_sampler import IterativeSampler

logger = logging.getLogger(__name__)


class Runner:
    """Orchestrates a full benchmark experiment.

    Parameters
    ----------
    config:
        Parsed experiment configuration.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_config_file(cls, path: str | Path) -> "Runner":
        """Create a :class:`Runner` from a YAML config file.

        Parameters
        ----------
        path:
            Path to the YAML config file.
        """
        config = ExperimentConfig.from_yaml(path)
        return cls(config)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def dry_run(self) -> None:
        """Print a summary of what would be executed without running anything."""
        problems = self._resolve_problems()
        print("=== DRY RUN ===")
        print(f"Problems ({len(problems)}): {problems}")
        print(f"Models ({len(self.config.models)}): {[m.name for m in self.config.models]}")
        print(f"Samplers ({len(self.config.samplers)}): {[(s.type, s.mode) for s in self.config.samplers]}")
        print(f"Seeds: {self.config.runs.seeds}")
        total = len(problems) * len(self.config.models) * len(self.config.samplers) * len(self.config.runs.seeds)
        print(f"Total runs: {total}")
        print(f"Output dir: {self.config.output_dir}")

    async def run(self) -> list[RunResult]:
        """Execute all runs and save results.

        Returns
        -------
        list[RunResult]
            All completed run results.
        """
        _setup_logging()
        problems = self._resolve_problems()
        logger.info("Starting experiment: %d problems, %d models, %d samplers, %d seeds",
                    len(problems), len(self.config.models), len(self.config.samplers), len(self.config.runs.seeds))

        config_dict = _config_to_dict(self.config)
        llm_params = {
            "temperature": self.config.llm_params.temperature,
            "max_tokens": self.config.llm_params.max_tokens,
            "top_p": self.config.llm_params.top_p,
        }

        all_results: list[RunResult] = []

        async with LLMClient(
            base_url=self.config.vllm.base_url,
            api_key=self.config.vllm.api_key,
            max_concurrency=self.config.vllm.max_concurrency,
            max_retries=self.config.vllm.max_retries,
        ) as client:
            for model_cfg in self.config.models:
                model_id = resolve_model(model_cfg.name)
                for sampler_cfg in self.config.samplers:
                    prompt_builder = PromptBuilder(mode=sampler_cfg.mode)
                    for problem_id in problems:
                        spec = registry.load(problem_id)
                        for seed in self.config.runs.seeds:
                            sampler = _build_sampler(
                                sampler_cfg=sampler_cfg,
                                client=client,
                                model=model_id,
                                prompt_builder=prompt_builder,
                                llm_params=llm_params,
                                eval_max_workers=self.config.evaluation.max_workers,
                            )
                            logger.info(
                                "Running: sampler=%s/%s model=%s problem=%s seed=%d",
                                sampler_cfg.type, sampler_cfg.mode, model_id, problem_id, seed,
                            )
                            result = await sampler.run(spec=spec, seed=seed, config_dict=config_dict)
                            result.save(self.config.output_dir)
                            all_results.append(result)
                            logger.info(
                                "  Done: best_score=%s",
                                f"{result.best_score:.4f}" if result.best_score is not None else "N/A",
                            )

        # Print aggregate summary
        _print_summary(all_results)
        return all_results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_problems(self) -> list[str]:
        """Expand ``["all"]`` to the full problem list, otherwise return as-is."""
        if self.config.problems == ["all"]:
            return registry.list_problems()
        return list(self.config.problems)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _build_sampler(
    sampler_cfg: SamplerConfig,
    client: LLMClient,
    model: str,
    prompt_builder: PromptBuilder,
    llm_params: dict[str, Any],
    eval_max_workers: int,
) -> BaseSampler:
    """Instantiate the correct sampler from a :class:`SamplerConfig`."""
    common = dict(
        client=client,
        model=model,
        prompt_builder=prompt_builder,
        llm_params=llm_params,
        eval_max_workers=eval_max_workers,
    )
    if sampler_cfg.type == "random":
        return RandomSampler(num_samples=sampler_cfg.num_samples, **common)
    if sampler_cfg.type == "iterative":
        return IterativeSampler(
            num_steps=sampler_cfg.num_steps,
            samples_per_step=sampler_cfg.samples_per_step,
            **common,
        )
    raise ValueError(f"Unknown sampler type: {sampler_cfg.type!r}")


def _config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    """Recursively serialise an :class:`ExperimentConfig` to a plain dict."""
    return dataclasses.asdict(config)


def _print_summary(results: list[RunResult]) -> None:
    """Print a human-readable summary table to stdout."""
    agg = aggregate_results(results)
    print("\n=== Experiment Summary ===")
    for key, stats in sorted(agg.items()):
        scores_str = ", ".join(f"{s:.4f}" for s in stats["scores"])
        print(f"  {key}")
        print(f"    scores=[{scores_str}]  mean={stats['mean']:.4f}  max={stats['max']:.4f}")
    print(f"Total runs: {len(results)}")


def _setup_logging() -> None:
    """Configure basic logging if not already configured."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
