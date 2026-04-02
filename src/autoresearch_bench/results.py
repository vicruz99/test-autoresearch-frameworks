"""Result data structures and JSON serialization utilities.

Each experiment run produces a :class:`RunResult` which bundles together all
metadata needed to reproduce or analyse the run.

Example usage::

    result = RunResult(...)
    result.save(output_dir="experiments/results")
    summary = result.summary()
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class StepResult:
    """Result of a single LLM generation + evaluation step.

    Attributes:
        step: Step index (0-based).
        prompt_messages: The messages sent to the LLM.
        raw_response: Raw text returned by the LLM.
        generated_code: Extracted / applied code.
        score: Evaluation score (``None`` if evaluation failed).
        valid: Whether the evaluation was valid.
        error: Error message, if any.
        execution_time: Time taken to execute the candidate code (seconds).
        metrics: Additional metrics from the evaluator.
        reasoning_content: Chain-of-thought reasoning produced by the model.
        prompt_tokens: Number of input tokens consumed.
        reasoning_tokens: Number of tokens spent on reasoning.
        completion_tokens: Total output tokens (including reasoning).
        total_tokens: Sum of prompt + completion tokens.
    """

    step: int
    prompt_messages: list[dict[str, str]]
    raw_response: str
    generated_code: str
    score: float | None
    valid: bool
    error: str
    execution_time: float
    metrics: dict[str, Any] = dataclasses.field(default_factory=dict)
    reasoning_content: str = ""
    prompt_tokens: int | None = None
    reasoning_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict."""
        return dataclasses.asdict(self)


@dataclasses.dataclass
class RunResult:
    """Full result for one (sampler, model, problem, seed) combination.

    Attributes:
        sampler_type: ``"random"`` or ``"iterative"``.
        sampler_mode: ``"full_rewrite"`` or ``"edit"``.
        model: Model identifier string.
        problem: Problem ID (e.g. ``"combinatorics/cap_set"``).
        seed: Random seed used for this run.
        steps: Ordered list of :class:`StepResult` objects.
        best_score: Best score achieved across all steps.
        best_code: Code that produced the best score.
        initial_score: Score of the initial program.
        config_dict: Serialised experiment configuration for reproducibility.
        initial_program: The initial program source code.
        timestamp: ISO-format timestamp of when the run completed.
    """

    sampler_type: str
    sampler_mode: str
    model: str
    problem: str
    seed: int
    steps: list[StepResult]
    best_score: float | None
    best_code: str
    initial_score: float | None
    config_dict: dict[str, Any]
    initial_program: str = ""
    timestamp: str = dataclasses.field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Return a compact summary dict (no generated code or prompts)."""
        return {
            "sampler_type": self.sampler_type,
            "sampler_mode": self.sampler_mode,
            "model": self.model,
            "problem": self.problem,
            "seed": self.seed,
            "best_score": self.best_score,
            "initial_score": self.initial_score,
            "num_steps": len(self.steps),
            "timestamp": self.timestamp,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialise the full run result to a plain dict."""
        return {
            "sampler_type": self.sampler_type,
            "sampler_mode": self.sampler_mode,
            "model": self.model,
            "problem": self.problem,
            "seed": self.seed,
            "best_score": self.best_score,
            "best_code": self.best_code,
            "initial_score": self.initial_score,
            "initial_program": self.initial_program,
            "config": self.config_dict,
            "timestamp": self.timestamp,
            "steps": [s.to_dict() for s in self.steps],
        }

    def save(self, output_dir: str | Path) -> Path:
        """Persist this run result to a per-run folder.

        Creates a folder named after the run and writes all artifacts inside it::

            <output_dir>/
            └── <run_name>/
                ├── result.json
                ├── initial_program.py
                └── programs/
                    ├── step_000.py
                    ├── step_001.py
                    └── ...

        Parameters
        ----------
        output_dir:
            Parent directory where the run folder will be created.

        Returns
        -------
        Path
            The path of the written JSON file.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        # Build a safe run-folder name from the key dimensions
        safe_model = self.model.replace("/", "_").replace(" ", "_")
        safe_problem = self.problem.replace("/", "_")
        stem = f"{self.sampler_type}_{self.sampler_mode}_{safe_model}_{safe_problem}_seed{self.seed}_{self.timestamp.replace(':', '-')}"

        run_dir = out / stem
        run_dir.mkdir(parents=True, exist_ok=True)

        # Write JSON result inside the run folder
        fpath = run_dir / "result.json"
        with open(fpath, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        logger.info("Saved result → %s", fpath)

        if self.initial_program:
            (run_dir / "initial_program.py").write_text(self.initial_program)

        programs_dir = run_dir / "programs"
        programs_dir.mkdir(parents=True, exist_ok=True)
        for step in self.steps:
            if step.generated_code:
                step_file = programs_dir / f"step_{step.step:03d}.py"
                step_file.write_text(step.generated_code)

        logger.info("Saved programs → %s", run_dir)
        return fpath


def aggregate_results(results: list[RunResult]) -> dict[str, Any]:
    """Compute aggregate statistics across a list of run results.

    Groups results by ``(sampler_type, sampler_mode, model, problem)`` and
    computes mean / std / max best_score over seeds.

    Parameters
    ----------
    results:
        List of completed run results.

    Returns
    -------
    dict
        Nested dict: ``{key: {"scores": [...], "mean": ..., "max": ...}}``.
    """
    import statistics

    groups: dict[str, list[float]] = {}
    for r in results:
        key = f"{r.sampler_type}/{r.sampler_mode}/{r.model}/{r.problem}"
        if r.best_score is not None:
            groups.setdefault(key, []).append(r.best_score)

    agg: dict[str, Any] = {}
    for key, scores in groups.items():
        entry: dict[str, Any] = {"scores": scores, "mean": statistics.mean(scores), "max": max(scores)}
        if len(scores) > 1:
            entry["stdev"] = statistics.stdev(scores)
        agg[key] = entry
    return agg
