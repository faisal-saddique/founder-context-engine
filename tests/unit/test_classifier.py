"""
Unit tests for the Librarian content classifier.
Mocks the LLM chain to test classification logic in isolation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.models.schemas import ContentSchema
from src.services.ingestion.classifier import ContentClassifier, ClassificationResult


def _mock_chain(schema_type: str, confidence: float):
    """Create a mock chain that returns a ClassificationResult."""
    mock = AsyncMock()
    mock.ainvoke = AsyncMock(return_value=ClassificationResult(
        schema_type=schema_type,
        confidence=confidence,
    ))
    return mock


class TestClassifierThreshold:
    """Confidence threshold logic."""

    @pytest.mark.asyncio
    async def test_high_confidence_returns_detected_schema(self):
        classifier = ContentClassifier.__new__(ContentClassifier)
        classifier.chain = _mock_chain("case_study", 85)
        schema, confidence = await classifier.classify("Some app metrics content")
        assert schema == ContentSchema.CASE_STUDY
        assert confidence == 85.0

    @pytest.mark.asyncio
    async def test_low_confidence_falls_back_to_general(self):
        classifier = ContentClassifier.__new__(ContentClassifier)
        classifier.chain = _mock_chain("profile", 45)
        schema, confidence = await classifier.classify("Mixed ambiguous content")
        assert schema == ContentSchema.GENERAL
        assert confidence == 45.0

    @pytest.mark.asyncio
    async def test_exact_60_uses_detected_schema(self):
        classifier = ContentClassifier.__new__(ContentClassifier)
        classifier.chain = _mock_chain("guide", 60)
        schema, confidence = await classifier.classify("Some guide content")
        assert schema == ContentSchema.GUIDE
        assert confidence == 60.0


class TestClassifierErrorHandling:
    """Error cases should fall back to GENERAL gracefully."""

    @pytest.mark.asyncio
    async def test_chain_exception_falls_back(self):
        classifier = ContentClassifier.__new__(ContentClassifier)
        classifier.chain = AsyncMock()
        classifier.chain.ainvoke = AsyncMock(side_effect=Exception("LLM down"))
        schema, confidence = await classifier.classify("Some content")
        assert schema == ContentSchema.GENERAL
        assert confidence == 0.0

    @pytest.mark.asyncio
    async def test_no_chain_configured(self):
        classifier = ContentClassifier.__new__(ContentClassifier)
        classifier.chain = None
        schema, confidence = await classifier.classify("Anything")
        assert schema == ContentSchema.GENERAL
        assert confidence == 0.0


class TestClassifierSchemaTypes:
    """Each schema type should be correctly returned when confidence is high."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("schema_name", ["case_study", "profile", "guide", "market_intel", "general"])
    async def test_each_schema_type(self, schema_name):
        classifier = ContentClassifier.__new__(ContentClassifier)
        classifier.chain = _mock_chain(schema_name, 90)
        schema, confidence = await classifier.classify("Content")
        assert schema == ContentSchema(schema_name)
        assert confidence == 90.0


class TestClassificationResultModel:
    """The Pydantic model enforces valid schema types."""

    def test_valid_schema(self):
        result = ClassificationResult(schema_type="case_study", confidence=85)
        assert result.schema_type == "case_study"
        assert result.confidence == 85.0

    def test_invalid_schema_rejected(self):
        with pytest.raises(Exception):
            ClassificationResult(schema_type="nonexistent", confidence=90)

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            ClassificationResult(schema_type="general", confidence=150)
        with pytest.raises(Exception):
            ClassificationResult(schema_type="general", confidence=-10)
