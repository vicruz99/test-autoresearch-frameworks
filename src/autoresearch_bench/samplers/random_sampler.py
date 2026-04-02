"""Random sampler — all N samples are independent LLM calls from the initial program.

Each call receives the *same* starting program (either the initial program or
the one provided by the caller) and generates an improved version.  There is
no memory between samples.

This is the simplest possible search strategy and acts as a strong baseline.
"""

from __future__ import annotations

import logging
from typing import Any

from autoresearch_problems import ProblemSpec, execute_and_evaluate

from autoresearch_bench.results import RunResult, StepResult
from autoresearch_bench.samplers.base import BaseSampler

logger = logging.getLogger(__name__)


class RandomSampler(BaseSampler):
    """Generate *num_samples* independent improved programs and return the best.

    Parameters
    ----------
    num_samples:
        Total number of independent LLM calls to make.
    See :class:`~autoresearch_bench.samplers.base.BaseSampler` for the
    remaining parameters.
    """

    #: Sampler type label used in result files.
    SAMPLER_TYPE = "random"

    def __init__(self, *, num_samples: int = 50, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.num_samples = num_samples

    async def run(
        self,
        spec: ProblemSpec,
        seed: int,
        config_dict: dict[str, Any],
    ) -> RunResult:
        """Run the random sampler and return a :class:`RunResult`.

        Parameters
        ----------
        spec:
            Problem specification.
        seed:
            Stored for reproducibility (not used internally by the random
            sampler since samples are independent).
        config_dict:
            Serialised experiment configuration.
        """
        logger.info(
            "[RandomSampler] problem=%s model=%s mode=%s num_samples=%d seed=%d",
            spec.name,
            self.model,
            self.prompt_builder.mode,
            self.num_samples,
            seed,
        )

        # Evaluate the initial program first
        initial_result = execute_and_evaluate(spec, spec.initial_program)
        initial_score: float | None = initial_result.score if initial_result.valid else None

        logger.info("[RandomSampler] initial_score=%.4f", initial_score or float("nan"))

        # Generate all candidates from the initial program
        candidates = await self._generate_candidates(spec, spec.initial_program, self.num_samples)

        # Separate valid code strings, keeping track of indices
        codes = [c for _, _, _, c in candidates if c is not None]
        none_mask = [c is None for _, _, _, c in candidates]

        steps: list[StepResult] = []

        if codes:
            eval_results = self._evaluate_candidates(spec, codes)
        else:
            eval_results = []

        # Rebuild per-candidate results
        eval_iter = iter(eval_results)
        all_candidates: list[tuple[float | None, str]] = []

        for idx, (messages, raw, reasoning_content, code) in enumerate(candidates):
            if code is None:
                step = StepResult(
                    step=idx,
                    prompt_messages=messages,
                    raw_response=raw,
                    generated_code="",
                    score=None,
                    valid=False,
                    error="No code extracted from LLM response",
                    execution_time=0.0,
                    reasoning_content=reasoning_content,
                )
                all_candidates.append((None, spec.initial_program))
            else:
                er = next(eval_iter)
                score: float | None = er.score if er.valid else None
                step = StepResult(
                    step=idx,
                    prompt_messages=messages,
                    raw_response=raw,
                    generated_code=code,
                    score=score,
                    valid=er.valid,
                    error=er.error,
                    execution_time=er.execution_time,
                    metrics=er.metrics,
                    reasoning_content=reasoning_content,
                )
                all_candidates.append((score, code))
            steps.append(step)

        best_score, best_code = self._best(all_candidates, spec.maximize)

        return RunResult(
            sampler_type=self.SAMPLER_TYPE,
            sampler_mode=self.prompt_builder.mode,
            model=self.model,
            problem=f"{spec.category}/{spec.name}" if spec.category else spec.name,
            seed=seed,
            steps=steps,
            best_score=best_score,
            best_code=best_code,
            initial_score=initial_score,
            config_dict=config_dict,
        )
