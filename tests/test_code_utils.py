"""Tests for autoresearch_bench.code_utils: extract_code and apply_edit."""

from __future__ import annotations

import pytest

from autoresearch_bench.code_utils import apply_edit, extract_code


# ---------------------------------------------------------------------------
# extract_code — full_rewrite mode
# ---------------------------------------------------------------------------

class TestExtractCodeFullRewrite:
    """Tests for extract_code in full_rewrite (python) mode."""

    def test_python_fenced_block(self):
        """Extracts content from a ```python ... ``` block."""
        text = "Here is the solution:\n```python\ndef solve(n):\n    return []\n```\nDone."
        result = extract_code(text, mode="full_rewrite")
        assert result == "def solve(n):\n    return []"

    def test_generic_fenced_block_fallback(self):
        """Falls back to a generic ``` block when no ```python block present."""
        text = "```\ndef solve():\n    pass\n```"
        result = extract_code(text, mode="full_rewrite")
        assert result == "def solve():\n    pass"

    def test_named_fenced_block_other_language(self):
        """Generic fallback catches a ```py or other language tag."""
        text = "```py\nreturn 42\n```"
        result = extract_code(text, mode="full_rewrite")
        assert result == "return 42"

    def test_no_code_block_returns_none(self):
        """Returns None when the response contains no fenced block."""
        result = extract_code("Just some text with no code block.", mode="full_rewrite")
        assert result is None

    def test_empty_string_returns_none(self):
        """Returns None for an empty input string."""
        assert extract_code("", mode="full_rewrite") is None

    def test_strips_whitespace(self):
        """Extracted code is stripped of leading/trailing whitespace."""
        text = "```python\n\n   def f(): pass\n\n```"
        result = extract_code(text, mode="full_rewrite")
        assert result == "def f(): pass"

    def test_multiline_code_block(self):
        """Handles multi-line python code correctly."""
        code = "import sys\n\ndef solve(n):\n    return list(range(n))"
        text = f"```python\n{code}\n```"
        result = extract_code(text, mode="full_rewrite")
        assert result == code


# ---------------------------------------------------------------------------
# extract_code — edit mode
# ---------------------------------------------------------------------------

class TestExtractCodeEditMode:
    """Tests for extract_code in edit (diff) mode."""

    def test_diff_block_preferred(self):
        """Prefers a ```diff block over a ```python block in edit mode."""
        text = (
            "```diff\n--- a/solve.py\n+++ b/solve.py\n@@ -1 +1 @@\n-old\n+new\n```\n"
            "```python\ndef solve(): pass\n```"
        )
        result = extract_code(text, mode="edit")
        assert "--- a/solve.py" in result
        assert "+new" in result

    def test_diff_block_extracted(self):
        """Correctly extracts a standalone diff block."""
        text = "```diff\n@@ -1,2 +1,2 @@\n-x = 1\n+x = 2\n```"
        result = extract_code(text, mode="edit")
        assert "@@ -1,2 +1,2 @@" in result
        assert "+x = 2" in result

    def test_fallback_to_python_block(self):
        """Falls back to python block when no diff block is present in edit mode."""
        text = "```python\ndef solve(): return 1\n```"
        result = extract_code(text, mode="edit")
        assert result == "def solve(): return 1"

    def test_no_block_returns_none(self):
        """Returns None when no block is found in edit mode."""
        result = extract_code("no code here", mode="edit")
        assert result is None

    def test_empty_diff_block(self):
        """An empty diff block is extracted as an empty string."""
        text = "```diff\n\n```"
        result = extract_code(text, mode="edit")
        assert result == ""


# ---------------------------------------------------------------------------
# apply_edit
# ---------------------------------------------------------------------------

class TestApplyEdit:
    """Tests for apply_edit with unified diff patches."""

    def test_simple_line_replacement(self):
        """Applies a single-hunk patch that replaces one line."""
        original = "def solve(n):\n    return []\n"
        patch = (
            "@@ -1,2 +1,2 @@\n"
            " def solve(n):\n"
            "-    return []\n"
            "+    return [[0] * n]\n"
        )
        result = apply_edit(original, patch)
        assert "return [[0] * n]" in result
        assert "return []" not in result

    def test_add_lines(self):
        """Applies a patch that adds new lines."""
        original = "x = 1\n"
        patch = (
            "@@ -1,1 +1,2 @@\n"
            " x = 1\n"
            "+y = 2\n"
        )
        result = apply_edit(original, patch)
        assert "x = 1" in result
        assert "y = 2" in result

    def test_remove_lines(self):
        """Applies a patch that removes a line."""
        original = "x = 1\ny = 2\nz = 3\n"
        patch = (
            "@@ -1,3 +1,2 @@\n"
            " x = 1\n"
            "-y = 2\n"
            " z = 3\n"
        )
        result = apply_edit(original, patch)
        assert "y = 2" not in result
        assert "x = 1" in result
        assert "z = 3" in result

    def test_multi_hunk_patch(self):
        """Applies a patch with two separate hunks."""
        original = "a\nb\nc\nd\ne\n"
        patch = (
            "@@ -1,2 +1,2 @@\n"
            "-a\n"
            "+A\n"
            " b\n"
            "@@ -4,2 +4,2 @@\n"
            " d\n"
            "-e\n"
            "+E\n"
        )
        result = apply_edit(original, patch)
        assert "A" in result
        assert "a" not in result

    def test_malformed_patch_returns_original(self):
        """Returns the original string when the patch is malformed."""
        original = "def solve(): pass\n"
        # A completely invalid patch with no hunk headers
        patch = "this is not a valid diff"
        result = apply_edit(original, patch)
        assert result == original

    def test_empty_original(self):
        """Applies a patch to an empty original."""
        patch = "@@ -0,0 +1,1 @@\n+new line\n"
        result = apply_edit("", patch)
        # Should at least not crash; result may vary
        assert isinstance(result, str)

    def test_empty_patch(self):
        """An empty patch leaves the original unchanged."""
        original = "def solve(): pass\n"
        result = apply_edit(original, "")
        assert result == original
