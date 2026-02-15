"""
Unit tests for the LangGraph workflow state and individual nodes.
All external services (LightRAG, Prisma, LLM) are mocked.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.graph.state import GenerationState
from src.services.graph.nodes import (
    input_analysis_node,
    validation_node,
)


def _make_state(**overrides) -> GenerationState:
    """Build a minimal valid state dict."""
    base = {
        "platform": "linkedin",
        "post_format": "general",
        "tone": "professional",
        "specific_resource_context": None,
        "custom_instructions": None,
        "retrieved_knowledge": None,
        "retrieved_rules": [],
        "source_ids": [],
        "draft_content": None,
        "final_content": None,
        "validation_passed": False,
        "validation_issues": [],
        "generation_id": None,
        "rule_ids_applied": [],
        "model_config": {},
        "critique_result": None,
        "retry_count": 0,
        "max_retries": 2,
        "previous_issues": [],
        "best_attempt": None,
    }
    base.update(overrides)
    return base


class TestGenerationStateShape:
    """The state TypedDict should accept all required keys."""

    def test_minimal_state(self):
        state = _make_state()
        assert state["platform"] == "linkedin"
        assert state["retrieved_knowledge"] is None

    def test_with_overrides(self):
        state = _make_state(platform="twitter", tone="casual")
        assert state["platform"] == "twitter"
        assert state["tone"] == "casual"


class TestInputAnalysisNode:
    @pytest.mark.asyncio
    async def test_generates_id(self):
        state = _make_state()
        result = await input_analysis_node(state)
        assert result["generation_id"] is not None
        assert len(result["generation_id"]) > 0

    @pytest.mark.asyncio
    async def test_sets_model_config(self):
        state = _make_state()
        result = await input_analysis_node(state)
        assert "model" in result["model_config"]
        assert result["model_config"]["model"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_initializes_lists(self):
        state = _make_state()
        result = await input_analysis_node(state)
        assert result["source_ids"] == []
        assert result["rule_ids_applied"] == []


class TestValidationNode:
    @pytest.mark.asyncio
    async def test_clean_draft_passes(self):
        state = _make_state(
            draft_content="We help app creators improve their products.",
            retrieved_knowledge="General knowledge context.",
        )
        result = await validation_node(state)
        assert result["validation_passed"] is True
        assert result["final_content"] == state["draft_content"]

    @pytest.mark.asyncio
    async def test_draft_with_unverified_claim(self):
        state = _make_state(
            draft_content="We are the best agency with 10,000 clients worldwide.",
            retrieved_knowledge="We help founders.",
            max_retries=0,
        )
        result = await validation_node(state)
        # "best" is a superlative and "10,000" is unverified
        assert len(result["validation_issues"]) > 0
        # retries exhausted, so best attempt is used as final content
        assert result["final_content"] is not None

    @pytest.mark.asyncio
    async def test_draft_with_verified_numbers(self):
        knowledge = "Team size: 25+ developers and specialists. Founded in 2019."
        state = _make_state(
            draft_content="Our team of 25 developers was founded in 2019.",
            retrieved_knowledge=knowledge,
        )
        result = await validation_node(state)
        assert result["validation_passed"] is True
