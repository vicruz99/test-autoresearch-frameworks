"""Model registry — maps human-friendly names to vLLM model identifiers.

Adding a new model is as simple as adding an entry to :data:`KNOWN_MODELS`.
The model name string is used verbatim as the ``model`` parameter in all
OpenAI-compatible API calls.

Example usage::

    from autoresearch_bench.llm.models import resolve_model
    model_id = resolve_model("gpt-oss-120b")
"""

from __future__ import annotations

# Maps alias → vLLM model identifier.
# The identifier must match exactly what vLLM serves at /v1/models.
KNOWN_MODELS: dict[str, str] = {
    "gpt-oss-120b": "gpt-oss-120b",
    "qwen3.5-32b": "Qwen/Qwen3.5-32B",
    "Qwen/Qwen3.5-32B": "Qwen/Qwen3.5-32B",
}


def resolve_model(name: str) -> str:
    """Return the vLLM model identifier for *name*.

    If *name* is already a known alias it is resolved; otherwise it is
    returned as-is (allows arbitrary model IDs without pre-registration).

    Parameters
    ----------
    name:
        Model alias or raw model identifier string.

    Returns
    -------
    str
        The vLLM model identifier.
    """
    return KNOWN_MODELS.get(name, name)
