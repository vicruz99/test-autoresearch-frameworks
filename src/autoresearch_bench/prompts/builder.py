"""Prompt construction for ``full_rewrite`` and ``edit`` modes.

The :class:`PromptBuilder` produces the system and user message lists that
are sent to the LLM for each code-generation step.

``full_rewrite`` mode
    The LLM is asked to rewrite the entire program.

``edit`` mode
    The LLM is asked to produce a unified diff / patch that can be applied
    to the current program via :func:`autoresearch_bench.code_utils.apply_edit`.
"""

from __future__ import annotations

from autoresearch_problems import ProblemSpec


_SYSTEM_FULL_REWRITE = """\
You are an expert programmer tasked with improving an existing solution to an optimization problem.
Your goal is to produce a better-performing implementation.

Rules:
- Return ONLY the complete, runnable Python code inside a single markdown code block (```python ... ```).
- Do NOT include any explanation, prose, or commentary outside the code block.
- The function signature must match the original exactly.
- The code must be self-contained and import any libraries it needs.
"""

_SYSTEM_EDIT = """\
You are an expert programmer tasked with improving an existing solution to an optimization problem.
Your goal is to produce a unified diff patch that improves the program's performance.

Rules:
- Return ONLY a unified diff patch inside a single markdown code block (```diff ... ```).
- The patch must apply cleanly to the current program.
- Do NOT include any explanation or prose outside the code block.
- Preserve the original function name and signature.
"""

_USER_FULL_REWRITE = """\
## Problem

{description}

## Current Program

```python
{current_program}
```

## Task

Rewrite the entire program to achieve a better score.
Focus on algorithmic improvements, smarter search strategies, or more efficient data structures.
Return the complete improved program in a single ```python ... ``` code block.
"""

_USER_EDIT = """\
## Problem

{description}

## Current Program

```python
{current_program}
```

## Task

Produce a unified diff patch that improves the current program's performance.
Focus on targeted algorithmic improvements.
Return only a ```diff ... ``` code block containing the patch.
"""


class PromptBuilder:
    """Builds LLM message lists for code-generation steps.

    Parameters
    ----------
    mode:
        ``"full_rewrite"`` or ``"edit"``.
    """

    def __init__(self, mode: str = "full_rewrite") -> None:
        if mode not in ("full_rewrite", "edit"):
            raise ValueError(f"Unknown mode {mode!r}; expected 'full_rewrite' or 'edit'.")
        self.mode = mode

    def build(self, spec: ProblemSpec, current_program: str) -> list[dict[str, str]]:
        """Build the message list for a single LLM call.

        Parameters
        ----------
        spec:
            The problem specification.
        current_program:
            The program to improve (initial or best-so-far).

        Returns
        -------
        list[dict[str, str]]
            OpenAI-format ``[{"role": ..., "content": ...}, ...]`` messages.
        """
        if self.mode == "full_rewrite":
            system = _SYSTEM_FULL_REWRITE
            user_template = _USER_FULL_REWRITE
        else:
            system = _SYSTEM_EDIT
            user_template = _USER_EDIT

        description = _build_description(spec)
        user = user_template.format(
            description=description,
            current_program=current_program,
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]


def _build_description(spec: ProblemSpec) -> str:
    """Combine the spec's description and initial prompt into a rich context string."""
    parts: list[str] = []
    if spec.description:
        parts.append(spec.description.strip())
    if spec.initial_prompt and spec.initial_prompt.strip() != spec.description.strip():
        parts.append(spec.initial_prompt.strip())
    return "\n\n".join(parts) if parts else spec.name
