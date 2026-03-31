"""Utilities for extracting and applying LLM-generated code changes.

Functions
---------
extract_code(text, mode)
    Pull the first code block out of an LLM response.
apply_edit(original, patch)
    Apply a unified-diff patch string to the original source.
"""

from __future__ import annotations

import difflib
import logging
import re

logger = logging.getLogger(__name__)

# Regex patterns for markdown code blocks
_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
_DIFF_BLOCK_RE = re.compile(r"```(?:diff)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str, mode: str = "full_rewrite") -> str | None:
    """Extract the first code block from an LLM response.

    Parameters
    ----------
    text:
        Raw LLM response text.
    mode:
        ``"full_rewrite"`` extracts a Python code block;
        ``"edit"`` extracts a diff/patch block.

    Returns
    -------
    str | None
        The extracted code/patch, or ``None`` if no block was found.
    """
    if mode == "edit":
        match = _DIFF_BLOCK_RE.search(text)
        if match:
            return match.group(1).strip()
        # Fall back to any code block
        match = _CODE_BLOCK_RE.search(text)
        if match:
            return match.group(1).strip()
    else:
        match = _CODE_BLOCK_RE.search(text)
        if match:
            return match.group(1).strip()
        # Try a generic fenced block
        generic = re.compile(r"```\w*\s*\n(.*?)```", re.DOTALL)
        match = generic.search(text)
        if match:
            return match.group(1).strip()

    logger.warning("No code block found in LLM response (mode=%s). Raw text preview: %.200s", mode, text)
    return None


def apply_edit(original: str, patch: str) -> str:
    """Apply a unified-diff *patch* to *original* and return the patched source.

    If the patch cannot be applied cleanly, the original is returned unchanged
    and a warning is logged.

    Parameters
    ----------
    original:
        The current program source code.
    patch:
        A unified diff string (as produced by ``diff -u`` or similar).

    Returns
    -------
    str
        The patched source, or *original* if application fails.
    """
    try:
        patched = _apply_unified_diff(original, patch)
        return patched
    except Exception as exc:
        logger.warning("Failed to apply patch: %s. Returning original.", exc)
        return original


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_unified_diff(original: str, patch: str) -> str:
    """Apply a unified diff patch to the original source string.

    This is a minimal implementation sufficient for the patches produced by
    typical LLMs (context lines + ``+``/``-`` hunks).
    """
    original_lines = original.splitlines(keepends=True)
    patch_lines = patch.splitlines(keepends=True)

    # Use difflib's restore to check if patch is a proper unified diff
    # First, try to parse hunks manually
    result_lines = _apply_hunks(original_lines, patch_lines)
    return "".join(result_lines)


def _apply_hunks(original_lines: list[str], patch_lines: list[str]) -> list[str]:
    """Parse and apply unified diff hunks."""
    result = list(original_lines)
    offset = 0  # accumulated line offset due to previous hunks

    hunk_header = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

    i = 0
    while i < len(patch_lines):
        line = patch_lines[i]
        m = hunk_header.match(line)
        if not m:
            i += 1
            continue

        orig_start = int(m.group(1)) - 1  # convert to 0-based
        orig_count = int(m.group(2)) if m.group(2) is not None else 1
        i += 1

        # Collect hunk body
        removes: list[tuple[int, str]] = []
        adds: list[str] = []
        context_pos = orig_start

        hunk_body: list[str] = []
        while i < len(patch_lines) and not hunk_header.match(patch_lines[i]):
            hunk_body.append(patch_lines[i])
            i += 1

        # Apply hunk: reconstruct the affected slice
        old_slice: list[str] = []
        new_slice: list[str] = []
        for hline in hunk_body:
            if hline.startswith("-"):
                old_slice.append(hline[1:])
            elif hline.startswith("+"):
                new_slice.append(hline[1:])
            elif hline.startswith(" ") or hline.startswith("\\ No newline"):
                if hline.startswith(" "):
                    ctx = hline[1:]
                    old_slice.append(ctx)
                    new_slice.append(ctx)

        # Find this slice in the result using fuzzy matching
        adjusted_start = orig_start + offset
        end = adjusted_start + len(old_slice)
        # Replace the old slice with the new slice
        result[adjusted_start:end] = new_slice
        offset += len(new_slice) - len(old_slice)

    return result
