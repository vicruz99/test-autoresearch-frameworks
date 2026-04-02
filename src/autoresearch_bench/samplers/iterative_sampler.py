"""Iterative refinement sampler — sequential improvement of the best-so-far program.

At each step, :attr:`samples_per_step` independent LLM calls are made from
the *best program found so far*.  The best result replaces the current program
and the process repeats for :attr:`num_steps` steps.

This sampler **reuses** all core logic from :class:`RandomSampler` and
:class:`BaseSampler`; only the outer search loop differs.
"""

from __future__ import annotations

import logging
from typing import Any

from autoresearch_problems import ProblemSpec, execute_and_evaluate

from autoresearch_bench.results import RunResult, StepResult
from autoresearch_bench.samplers.base import BaseSampler

logger = logging.getLogger(__name__)


class IterativeSampler(BaseSampler):
    """Iteratively refine the best-so-far program over multiple steps.

    Parameters
    ----------
    num_steps:
        Number of sequential refinement steps.
    samples_per_step:
        Number of independent LLM calls per step (the best among them is
        kept for the next step).
    See :class:`~autoresearch_bench.samplers.base.BaseSampler` for the
    remaining parameters.
    """

    #: Sampler type label used in result files.
    SAMPLER_TYPE = "iterative"

    def __init__(self, *, num_steps: int = 10, samples_per_step: int = 5, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.num_steps = num_steps
        self.samples_per_step = samples_per_step

    async def run(
        self,
        spec: ProblemSpec,
        seed: int,
        config_dict: dict[str, Any],
    ) -> RunResult:
        """Run the iterative refinement loop and return a :class:`RunResult`.

        Parameters
        ----------
        spec:
            Problem specification.
        seed:
            Stored for reproducibility.
        config_dict:
            Serialised experiment configuration.
        """
        logger.info(
            "[IterativeSampler] problem=%s model=%s mode=%s num_steps=%d samples_per_step=%d seed=%d",
            spec.name,
            self.model,
            self.prompt_builder.mode,
            self.num_steps,
            self.samples_per_step,
            seed,
        )

        # Evaluate initial program
        initial_result = execute_and_evaluate(spec, spec.initial_program)
        initial_score: float | None = initial_result.score if initial_result.valid else None

        logger.info("[IterativeSampler] initial_score=%.4f", initial_score or float("nan"))

        current_program = spec.initial_program
        current_best_score = initial_score

        all_steps: list[StepResult] = []
        global_step = 0

        for step_idx in range(self.num_steps):
            logger.info(
                "[IterativeSampler] step %d/%d  current_best=%.4f",
                step_idx + 1,
                self.num_steps,
                current_best_score or float("nan"),
            )

            # Generate samples_per_step candidates from the current best program
            candidates = await self._generate_candidates(spec, current_program, self.samples_per_step)

            codes = [c for _, _, c, _ in candidates if c is not None]

            if codes:
                eval_results = self._evaluate_candidates(spec, codes)
            else:
                eval_results = []

            eval_iter = iter(eval_results)
            step_candidates: list[tuple[float | None, str]] = []

            for _, (messages, raw, code, completion) in enumerate(candidates):
                if code is None:
                    step = StepResult(
                        step=global_step,
                        prompt_messages=messages,
                        raw_response=raw,
                        generated_code="",
                        score=None,
                        valid=False,
                        error="No code extracted from LLM response",
                        execution_time=0.0,
                        reasoning_content=completion.reasoning_content if completion else "",
                        prompt_tokens=completion.prompt_tokens if completion else None,
                        reasoning_tokens=completion.reasoning_tokens if completion else None,
                        completion_tokens=completion.completion_tokens if completion else None,
                        total_tokens=completion.total_tokens if completion else None,
                    )
                    step_candidates.append((None, current_program))
                else:
                    er = next(eval_iter)
                    score: float | None = er.score if er.valid else None
                    step = StepResult(
                        step=global_step,
                        prompt_messages=messages,
                        raw_response=raw,
                        generated_code=code,
                        score=score,
                        valid=er.valid,
                        error=er.error,
                        execution_time=er.execution_time,
                        metrics=er.metrics,
                        reasoning_content=completion.reasoning_content if completion else "",
                        prompt_tokens=completion.prompt_tokens if completion else None,
                        reasoning_tokens=completion.reasoning_tokens if completion else None,
                        completion_tokens=completion.completion_tokens if completion else None,
                        total_tokens=completion.total_tokens if completion else None,
                    )
                    step_candidates.append((score, code))
                all_steps.append(step)
                global_step += 1

            best_score_this_step, best_code_this_step = self._best(step_candidates, spec.maximize)

            # Update current program if we found an improvement (or equal score)
            if best_score_this_step is not None:
                if current_best_score is None or (
                    (best_score_this_step > current_best_score)
                    if spec.maximize
                    else (best_score_this_step < current_best_score)
                ):
                    current_best_score = best_score_this_step
                    current_program = best_code_this_step
                    logger.info("[IterativeSampler] New best score: %.4f", current_best_score)

        return RunResult(
            sampler_type=self.SAMPLER_TYPE,
            sampler_mode=self.prompt_builder.mode,
            model=self.model,
            problem=f"{spec.category}/{spec.name}" if spec.category else spec.name,
            seed=seed,
            steps=all_steps,
            best_score=current_best_score,
            best_code=current_program,
            initial_score=initial_score,
            config_dict=config_dict,
            initial_program=spec.initial_program,
        )
