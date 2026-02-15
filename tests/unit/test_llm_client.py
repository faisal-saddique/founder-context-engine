"""
Unit tests for the LLM client wrapper.
All LLM calls are mocked to avoid real API calls.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.llm.client import LLMClient


class TestGenerateContent:
    @pytest.mark.asyncio
    async def test_returns_content(self):
        client = LLMClient.__new__(LLMClient)
        mock_response = MagicMock()
        mock_response.content = "Generated LinkedIn post about ASO."
        client.generation_model = MagicMock()
        client.generation_model.ainvoke = AsyncMock(return_value=mock_response)

        result = await client.generate_content(
            system_prompt="You are a content writer.",
            user_prompt="Write a post about ASO.",
        )
        assert result == "Generated LinkedIn post about ASO."

    @pytest.mark.asyncio
    async def test_includes_context(self):
        client = LLMClient.__new__(LLMClient)
        mock_response = MagicMock()
        mock_response.content = "Post with context."
        client.generation_model = MagicMock()
        client.generation_model.ainvoke = AsyncMock(return_value=mock_response)

        result = await client.generate_content(
            system_prompt="System",
            user_prompt="Prompt",
            context="Some brand context",
        )
        assert result == "Post with context."
        # the invoke should have been called with 3 messages (system + context + user)
        call_args = client.generation_model.ainvoke.call_args[0][0]
        assert len(call_args) == 3

    @pytest.mark.asyncio
    async def test_no_model_returns_fallback(self):
        client = LLMClient.__new__(LLMClient)
        client.generation_model = None

        result = await client.generate_content("system", "user")
        assert "not configured" in result

    @pytest.mark.asyncio
    async def test_api_error_returns_error_message(self):
        client = LLMClient.__new__(LLMClient)
        client.generation_model = MagicMock()
        client.generation_model.ainvoke = AsyncMock(
            side_effect=Exception("Rate limit")
        )

        result = await client.generate_content("system", "user")
        assert "Error:" in result
        assert "Rate limit" in result


class TestCritiqueContent:
    @pytest.mark.asyncio
    async def test_returns_critique(self):
        client = LLMClient.__new__(LLMClient)
        mock_response = MagicMock()
        mock_response.content = "Good quality content. Score: 8/10."
        client.critique_model = MagicMock()
        client.critique_model.ainvoke = AsyncMock(return_value=mock_response)

        result = await client.critique_content(
            content="Some post content",
            criteria=["Professional tone", "No unsupported claims"],
        )
        assert result["score"] == 8
        assert "Good quality" in result["feedback"]

    @pytest.mark.asyncio
    async def test_no_model_fallback(self):
        client = LLMClient.__new__(LLMClient)
        client.critique_model = None

        result = await client.critique_content("Content", ["Criteria"])
        assert result["score"] == 0
        assert "not configured" in result["feedback"]

    @pytest.mark.asyncio
    async def test_critique_error(self):
        client = LLMClient.__new__(LLMClient)
        client.critique_model = MagicMock()
        client.critique_model.ainvoke = AsyncMock(
            side_effect=Exception("API error")
        )

        result = await client.critique_content("Content", ["Criteria"])
        assert result["score"] == 0
        assert "Error:" in result["feedback"]
