"""Abstract base class shared by all samplers.

All concrete samplers inherit from :class:`BaseSampler` and override
:meth:`run`, which returns a :class:`~autoresearch_bench.results.RunResult`.

The base class provides two key shared helpers:

* :meth:`_generate_candidates` — send a batch of prompts to the LLM and
  extract code from each response.
* :meth:`_evaluate_candidates` — evaluate a batch of code strings against
  the problem spec and return :class:`~autoresearch_problems.EvalResult` objects.
"""

from __future__ import annotations

import abc
import logging
from typing import Any

from autoresearch_problems import ProblemSpec, execute_and_evaluate_batch, EvalResult

from autoresearch_bench.code_utils import extract_code, apply_edit
from autoresearch_bench.llm.client import LLMClient
from autoresearch_bench.prompts.builder import PromptBuilder
from autoresearch_bench.results import RunResult, StepResult

logger = logging.getLogger(__name__)


class BaseSampler(abc.ABC):
    """Abstract base sampler.

    Subclasses implement :meth:`run` using the helpers provided here.

    Parameters
    ----------
    client:
        The async LLM client to use for generation.
    model:
        Model identifier string.
    prompt_builder:
        Prompt builder for the chosen mode (``full_rewrite`` or ``edit``).
    llm_params:
        Dict with keys ``temperature``, ``max_tokens``, ``top_p``.
    eval_max_workers:
        Number of parallel workers for :func:`execute_and_evaluate_batch`.
    """

    def __init__(
        self,
        client: LLMClient,
        model: str,
        prompt_builder: PromptBuilder,
        llm_params: dict[str, Any],
        eval_max_workers: int = 8,
    ) -> None:
        self.client = client
        self.model = model
        self.prompt_builder = prompt_builder
        self.llm_params = llm_params
        self.eval_max_workers = eval_max_workers

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    async def run(
        self,
        spec: ProblemSpec,
        seed: int,
        config_dict: dict[str, Any],
    ) -> RunResult:
        """Execute the sampler strategy and return a :class:`RunResult`.

        Parameters
        ----------
        spec:
            The problem specification.
        seed:
            Random seed (stored for reproducibility; used by subclasses as
            needed).
        config_dict:
            Serialised experiment config for embedding in the result.
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    async def _generate_candidates(
        self,
        spec: ProblemSpec,
        current_program: str,
        n: int,
    ) -> list[tuple[list[dict[str, str]], str, str | None, str | None]]:
        """Ask the LLM to generate *n* improved candidates.

        Parameters
        ----------
        spec:
            Problem specification.
        current_program:
            The program to improve (used as the context in the prompt).
        n:
            Number of candidates to generate concurrently.

        Returns
        -------
        list of (messages, raw_response, reasoning_content, extracted_code)
            Each element corresponds to one LLM call.  ``reasoning_content``
            is ``None`` when the model does not emit reasoning tokens.
            ``extracted_code`` is ``None`` when no code block could be extracted.
        """
        messages = self.prompt_builder.build(spec, current_program)
        messages_list = [messages] * n

        raw_responses = await self.client.batch_complete(
            model=self.model,
            messages_list=messages_list,
            **self.llm_params,
        )

        results = []
        for raw_result in raw_responses:
            if isinstance(raw_result, Exception):
                logger.warning("LLM call failed: %s", raw_result)
                results.append((messages, "", None, None))
                continue
            raw, reasoning_content = raw_result
            code = extract_code(raw, mode=self.prompt_builder.mode)
            if code is not None and self.prompt_builder.mode == "edit":
                code = apply_edit(current_program, code)
            results.append((messages, raw, reasoning_content, code))
        return results

    def _evaluate_candidates(
        self,
        spec: ProblemSpec,
        codes: list[str],
    ) -> list[EvalResult]:
        """Evaluate a list of candidate programs against *spec*.

        Parameters
        ----------
        spec:
            Problem specification.
        codes:
            List of program source strings.

        Returns
        -------
        list[EvalResult]
            One result per input code string.
        """
        return execute_and_evaluate_batch(spec, codes, max_workers=self.eval_max_workers)

    def _best(self, candidates: list[tuple[float | None, str]], maximize: bool) -> tuple[float | None, str]:
        """Return the (score, code) pair with the best score.

        Parameters
        ----------
        candidates:
            List of (score, code) pairs.  ``None`` scores are treated as worst.
        maximize:
            Whether higher scores are better.

        Returns
        -------
        tuple[float | None, str]
            The best (score, code) pair.
        """
        valid = [(s, c) for s, c in candidates if s is not None]
        if not valid:
            return candidates[0] if candidates else (None, "")
        return max(valid, key=lambda x: x[0]) if maximize else min(valid, key=lambda x: x[0])  # type: ignore[return-value]
