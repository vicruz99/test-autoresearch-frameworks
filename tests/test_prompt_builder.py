"""Tests for autoresearch_bench.prompts.builder: PromptBuilder."""

from __future__ import annotations

import pytest

from autoresearch_bench.prompts.builder import PromptBuilder, _build_description


class TestPromptBuilderInit:
    """Tests for PromptBuilder construction and mode validation."""

    def test_full_rewrite_mode_accepted(self):
        """PromptBuilder accepts 'full_rewrite' without raising."""
        builder = PromptBuilder(mode="full_rewrite")
        assert builder.mode == "full_rewrite"

    def test_edit_mode_accepted(self):
        """PromptBuilder accepts 'edit' without raising."""
        builder = PromptBuilder(mode="edit")
        assert builder.mode == "edit"

    def test_invalid_mode_raises_value_error(self):
        """PromptBuilder raises ValueError for an unrecognised mode."""
        with pytest.raises(ValueError, match="Unknown mode"):
            PromptBuilder(mode="invalid_mode")

    def test_default_mode_is_full_rewrite(self):
        """Default mode is 'full_rewrite'."""
        builder = PromptBuilder()
        assert builder.mode == "full_rewrite"


class TestPromptBuilderBuildFullRewrite:
    """Tests for PromptBuilder.build in full_rewrite mode."""

    def test_returns_two_messages(self, sample_spec, full_rewrite_builder):
        """build() returns exactly two messages: system and user."""
        messages = full_rewrite_builder.build(sample_spec, "def solve(n): return []")
        assert len(messages) == 2

    def test_first_message_is_system(self, sample_spec, full_rewrite_builder):
        """The first message has role 'system'."""
        messages = full_rewrite_builder.build(sample_spec, "def solve(n): return []")
        assert messages[0]["role"] == "system"

    def test_second_message_is_user(self, sample_spec, full_rewrite_builder):
        """The second message has role 'user'."""
        messages = full_rewrite_builder.build(sample_spec, "def solve(n): return []")
        assert messages[1]["role"] == "user"

    def test_system_message_mentions_full_rewrite(self, sample_spec, full_rewrite_builder):
        """The system message instructs for a full rewrite (```python block)."""
        messages = full_rewrite_builder.build(sample_spec, "def solve(n): return []")
        assert "```python" in messages[0]["content"]

    def test_user_message_contains_current_program(self, sample_spec, full_rewrite_builder):
        """The user message embeds the current program."""
        current = "def solve(n):\n    return list(range(n))"
        messages = full_rewrite_builder.build(sample_spec, current)
        assert current in messages[1]["content"]

    def test_user_message_contains_description(self, sample_spec, full_rewrite_builder):
        """The user message contains the problem description."""
        messages = full_rewrite_builder.build(sample_spec, "def solve(n): return []")
        assert sample_spec.description in messages[1]["content"]

    def test_messages_are_dicts_with_role_and_content(self, sample_spec, full_rewrite_builder):
        """Each message is a dict with exactly 'role' and 'content' keys."""
        messages = full_rewrite_builder.build(sample_spec, "pass")
        for msg in messages:
            assert set(msg.keys()) == {"role", "content"}
            assert isinstance(msg["content"], str)


class TestPromptBuilderBuildEdit:
    """Tests for PromptBuilder.build in edit mode."""

    def test_returns_two_messages(self, sample_spec, edit_builder):
        """build() returns exactly two messages in edit mode."""
        messages = edit_builder.build(sample_spec, "def solve(n): return []")
        assert len(messages) == 2

    def test_system_message_mentions_diff(self, sample_spec, edit_builder):
        """The system message instructs for a diff/patch output."""
        messages = edit_builder.build(sample_spec, "def solve(n): return []")
        assert "diff" in messages[0]["content"].lower() or "patch" in messages[0]["content"].lower()

    def test_user_message_mentions_diff_block(self, sample_spec, edit_builder):
        """The user message requests a ```diff code block."""
        messages = edit_builder.build(sample_spec, "def solve(n): return []")
        assert "```diff" in messages[1]["content"]

    def test_user_message_contains_current_program(self, sample_spec, edit_builder):
        """The user message embeds the current program in edit mode."""
        current = "def solve(n):\n    return []"
        messages = edit_builder.build(sample_spec, current)
        assert current in messages[1]["content"]


class TestBuildDescription:
    """Tests for the _build_description helper."""

    def test_description_only(self, sample_spec):
        """Uses the description when initial_prompt is absent or equal."""
        from autoresearch_problems import ProblemSpec
        spec = ProblemSpec(
            name="test",
            category="cat",
            description="My description.",
            output_type="list",
            evaluator_code="",
            evaluator_entrypoint="evaluate",
            evaluator_dependencies=[],
            parameters={},
            initial_prompt=None,
        )
        result = _build_description(spec)
        assert result == "My description."

    def test_description_and_distinct_prompt(self, sample_spec):
        """Combines description and initial_prompt when they differ."""
        from autoresearch_problems import ProblemSpec
        spec = ProblemSpec(
            name="test",
            category="cat",
            description="My description.",
            output_type="list",
            evaluator_code="",
            evaluator_entrypoint="evaluate",
            evaluator_dependencies=[],
            parameters={},
            initial_prompt="Extra context.",
        )
        result = _build_description(spec)
        assert "My description." in result
        assert "Extra context." in result

    def test_no_description_no_prompt_returns_name(self):
        """Falls back to spec.name when both description and prompt are empty."""
        from autoresearch_problems import ProblemSpec
        spec = ProblemSpec(
            name="fallback_name",
            category="cat",
            description="",
            output_type="list",
            evaluator_code="",
            evaluator_entrypoint="evaluate",
            evaluator_dependencies=[],
            parameters={},
            initial_prompt=None,
        )
        result = _build_description(spec)
        assert result == "fallback_name"

    def test_duplicate_prompt_not_repeated(self):
        """When initial_prompt equals description, it is not repeated."""
        from autoresearch_problems import ProblemSpec
        text = "Both the same."
        spec = ProblemSpec(
            name="dup",
            category="cat",
            description=text,
            output_type="list",
            evaluator_code="",
            evaluator_entrypoint="evaluate",
            evaluator_dependencies=[],
            parameters={},
            initial_prompt=text,
        )
        result = _build_description(spec)
        # Should not contain the text twice
        assert result.count(text) == 1
