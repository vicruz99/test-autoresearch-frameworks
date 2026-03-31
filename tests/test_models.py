"""Tests for autoresearch_bench.llm.models: resolve_model."""

from __future__ import annotations

import pytest

from autoresearch_bench.llm.models import KNOWN_MODELS, resolve_model


class TestResolveModel:
    """Tests for the resolve_model function."""

    def test_known_alias_gpt_oss(self):
        """resolve_model returns the correct id for the gpt-oss-120b alias."""
        assert resolve_model("gpt-oss-120b") == "gpt-oss-120b"

    def test_known_alias_qwen_short(self):
        """resolve_model resolves the short qwen alias to the full HF model id."""
        assert resolve_model("qwen3.5-32b") == "Qwen/Qwen3.5-32B"

    def test_known_alias_qwen_full(self):
        """resolve_model is a no-op when the full model id is passed."""
        assert resolve_model("Qwen/Qwen3.5-32B") == "Qwen/Qwen3.5-32B"

    def test_unknown_model_passthrough(self):
        """An unknown model name is returned unchanged."""
        assert resolve_model("my-custom-model-v1") == "my-custom-model-v1"

    def test_empty_string_passthrough(self):
        """An empty string is returned as-is."""
        assert resolve_model("") == ""

    def test_all_known_models_resolve_to_string(self):
        """Every entry in KNOWN_MODELS resolves to a non-empty string."""
        for alias in KNOWN_MODELS:
            result = resolve_model(alias)
            assert isinstance(result, str)
            assert len(result) > 0

    def test_known_models_dict_not_empty(self):
        """KNOWN_MODELS contains at least one entry."""
        assert len(KNOWN_MODELS) > 0
