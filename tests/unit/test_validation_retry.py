"""
Tests for the validation retry loop.
Covers routing logic, correction prompts, best-attempt tracking,
and the full draft->validate->retry cycle.
"""
import pytest
from unittest.mock import AsyncMock, patch

from src.services.graph.state import GenerationState
from src.services.graph.nodes import (
    validation_node,
    draft_generation_node,
    _format_correction_prompt,
)
from src.services.graph.workflow import should_retry


def _make_state(**overrides) -> GenerationState:
    """Build a minimal valid state dict with retry fields."""
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


# -- should_retry routing ---------------------------------------------------

class TestShouldRetry:
    """The routing function decides whether to loop back or move on."""

    def test_valid_content_goes_to_critique(self):
        state = _make_state(validation_passed=True, retry_count=0, max_retries=2)
        assert should_retry(state) == "critique"

    def test_failed_with_retries_left_goes_to_draft(self):
        state = _make_state(validation_passed=False, retry_count=0, max_retries=2)
        assert should_retry(state) == "draft_generation"

    def test_failed_second_attempt_still_retries(self):
        state = _make_state(validation_passed=False, retry_count=1, max_retries=2)
        assert should_retry(state) == "draft_generation"

    def test_failed_retries_exhausted_goes_to_critique(self):
        state = _make_state(validation_passed=False, retry_count=2, max_retries=2)
        assert should_retry(state) == "critique"

    def test_valid_after_retries_goes_to_critique(self):
        state = _make_state(validation_passed=True, retry_count=2, max_retries=2)
        assert should_retry(state) == "critique"


# -- correction prompt -------------------------------------------------------

class TestFormatCorrectionPrompt:
    """The correction prompt should clearly describe each issue for the LLM."""

    def test_includes_header(self):
        issues = [{"claim": "500 users", "issue_type": "unverified_number",
                    "severity": "high", "suggestion": "Remove number"}]
        result = _format_correction_prompt(issues)
        assert "PREVIOUS DRAFT HAD VALIDATION ISSUES" in result

    def test_includes_claim_text(self):
        issues = [{"claim": "We are the best", "issue_type": "unverified_superlative",
                    "severity": "high", "suggestion": "Remove superlative"}]
        result = _format_correction_prompt(issues)
        assert "We are the best" in result

    def test_labels_issue_types(self):
        issues = [
            {"claim": "100", "issue_type": "unverified_number",
             "severity": "high", "suggestion": "Fix"},
            {"claim": "best", "issue_type": "unverified_superlative",
             "severity": "high", "suggestion": "Fix"},
            {"claim": "Acme Corp", "issue_type": "unverified_entity",
             "severity": "medium", "suggestion": "Fix"},
        ]
        result = _format_correction_prompt(issues)
        assert "Unverified number" in result
        assert "Unverified superlative" in result
        assert "Unverified company/entity name" in result

    def test_includes_severity(self):
        issues = [{"claim": "x", "issue_type": "unverified_number",
                    "severity": "high", "suggestion": "Fix"}]
        result = _format_correction_prompt(issues)
        assert "[HIGH]" in result

    def test_includes_grounding_instruction(self):
        issues = [{"claim": "x", "issue_type": "unverified_number",
                    "severity": "high", "suggestion": "Fix"}]
        result = _format_correction_prompt(issues)
        assert "Only use facts, numbers, and names" in result

    def test_numbers_each_issue(self):
        issues = [
            {"claim": "a", "issue_type": "unverified_number",
             "severity": "high", "suggestion": "Fix"},
            {"claim": "b", "issue_type": "unverified_superlative",
             "severity": "high", "suggestion": "Fix"},
        ]
        result = _format_correction_prompt(issues)
        assert "1." in result
        assert "2." in result


# -- validation node retry tracking -----------------------------------------

class TestValidationNodeRetry:
    """The validation node should track retries and pick the best attempt."""

    @pytest.mark.asyncio
    async def test_clean_pass_sets_final_content(self):
        state = _make_state(
            draft_content="We help app creators improve their products.",
            retrieved_knowledge="General knowledge context.",
        )
        result = await validation_node(state)
        assert result["validation_passed"] is True
        assert result["final_content"] == "We help app creators improve their products."
        assert result["retry_count"] == 0

    @pytest.mark.asyncio
    async def test_failed_validation_bumps_retry_count(self):
        state = _make_state(
            draft_content="We are the best agency with 10,000 clients.",
            retrieved_knowledge="We help founders.",
            retry_count=0,
            max_retries=2,
        )
        result = await validation_node(state)
        assert result["validation_passed"] is False
        assert result["retry_count"] == 1
        # final_content should NOT be set yet (will retry)
        assert result["final_content"] is None

    @pytest.mark.asyncio
    async def test_failed_validation_stashes_issues(self):
        state = _make_state(
            draft_content="We are the best agency with 10,000 clients.",
            retrieved_knowledge="We help founders.",
            retry_count=0,
            max_retries=2,
        )
        result = await validation_node(state)
        assert len(result["previous_issues"]) > 0
        assert result["previous_issues"] == result["validation_issues"]

    @pytest.mark.asyncio
    async def test_retries_exhausted_uses_best_attempt(self):
        state = _make_state(
            draft_content="We are the best agency.",
            retrieved_knowledge="We help founders.",
            retry_count=2,
            max_retries=2,
            best_attempt={"content": "A better draft.", "issue_count": 1},
        )
        result = await validation_node(state)
        # retries exhausted, should use best attempt
        assert result["final_content"] == "A better draft."

    @pytest.mark.asyncio
    async def test_best_attempt_updated_when_fewer_issues(self):
        state = _make_state(
            draft_content="We help app creators improve their products.",
            retrieved_knowledge="General knowledge context.",
            best_attempt={"content": "Old draft with issues.", "issue_count": 5},
        )
        result = await validation_node(state)
        # clean draft (0 issues) should replace the old best (5 issues)
        assert result["best_attempt"]["issue_count"] == 0
        assert result["best_attempt"]["content"] == "We help app creators improve their products."

    @pytest.mark.asyncio
    async def test_best_attempt_not_replaced_when_more_issues(self):
        best = {"content": "Good draft.", "issue_count": 0}
        state = _make_state(
            draft_content="We are the best agency with 10,000 clients.",
            retrieved_knowledge="We help founders.",
            best_attempt=best,
            max_retries=0,
        )
        result = await validation_node(state)
        # current draft has issues, best attempt (0 issues) should stay
        assert result["best_attempt"]["content"] == "Good draft."
        assert result["best_attempt"]["issue_count"] == 0


# -- draft generation node with corrections ----------------------------------

class TestDraftGenerationRetry:
    """Draft generation should include correction prompt on retries."""

    @pytest.mark.asyncio
    async def test_first_pass_no_correction(self):
        mock_generate = AsyncMock(return_value="Generated content")
        state = _make_state(
            retrieved_knowledge="Some context.",
            retrieved_rules=[],
            retry_count=0,
            previous_issues=[],
        )
        with patch("src.services.graph.nodes.llm_client") as mock_llm:
            mock_llm.generate_content = mock_generate
            await draft_generation_node(state)

        call_args = mock_generate.call_args
        user_prompt = call_args.kwargs.get("user_prompt", call_args[1] if len(call_args[1]) > 1 else call_args[0][1])
        assert "PREVIOUS DRAFT HAD VALIDATION ISSUES" not in user_prompt

    @pytest.mark.asyncio
    async def test_retry_includes_correction(self):
        mock_generate = AsyncMock(return_value="Fixed content")
        issues = [{"claim": "500 users", "issue_type": "unverified_number",
                    "severity": "high", "suggestion": "Remove number"}]
        state = _make_state(
            retrieved_knowledge="Some context.",
            retrieved_rules=[],
            retry_count=1,
            previous_issues=issues,
        )
        with patch("src.services.graph.nodes.llm_client") as mock_llm:
            mock_llm.generate_content = mock_generate
            await draft_generation_node(state)

        call_args = mock_generate.call_args
        user_prompt = call_args.kwargs.get("user_prompt", call_args[1] if len(call_args[1]) > 1 else call_args[0][1])
        assert "PREVIOUS DRAFT HAD VALIDATION ISSUES" in user_prompt
        assert "500 users" in user_prompt

    @pytest.mark.asyncio
    async def test_retry_updates_draft_content(self):
        mock_generate = AsyncMock(return_value="Corrected draft")
        state = _make_state(
            retrieved_knowledge="Some context.",
            retrieved_rules=[],
            retry_count=1,
            previous_issues=[{"claim": "x", "issue_type": "unverified_number",
                              "severity": "high", "suggestion": "Fix"}],
        )
        with patch("src.services.graph.nodes.llm_client") as mock_llm:
            mock_llm.generate_content = mock_generate
            result = await draft_generation_node(state)

        assert result["draft_content"] == "Corrected draft"


# -- state shape check ------------------------------------------------------

class TestRetryStateFields:
    """Retry fields should be part of the state."""

    def test_state_has_retry_fields(self):
        state = _make_state()
        assert "retry_count" in state
        assert "max_retries" in state
        assert "previous_issues" in state
        assert "best_attempt" in state

    def test_defaults(self):
        state = _make_state()
        assert state["retry_count"] == 0
        assert state["max_retries"] == 2
        assert state["previous_issues"] == []
        assert state["best_attempt"] is None
