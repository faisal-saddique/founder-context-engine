"""
Unit tests for the 5 master content schemas and updated request/response models.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from src.models.schemas import (
    ContentSchema,
    SourceType,
    ParsePreviewResponse,
    CommitIngestRequest,
    UsagePermission,
    TrustScore,
)
from src.models.knowledge import UnifiedKnowledge, KnowledgeMetadata


class TestContentSchemaEnum:
    """Verify the 5 master schema enum values."""

    def test_all_five_schemas_exist(self):
        assert ContentSchema.CASE_STUDY == "case_study"
        assert ContentSchema.PROFILE == "profile"
        assert ContentSchema.GUIDE == "guide"
        assert ContentSchema.MARKET_INTEL == "market_intel"
        assert ContentSchema.GENERAL == "general"

    def test_schema_count(self):
        assert len(ContentSchema) == 5

    def test_markdown_source_type(self):
        assert SourceType.MARKDOWN == "markdown"


class TestParsePreviewResponse:
    """ParsePreviewResponse should include schema + confidence."""

    def test_includes_schema_and_confidence(self):
        resp = ParsePreviewResponse(
            markdown_content="# Hello",
            preview_metadata={"source_type": "markdown"},
            source_type=SourceType.MARKDOWN,
            content_schema=ContentSchema.GUIDE,
            confidence_score=87.5,
        )
        assert resp.content_schema == ContentSchema.GUIDE
        assert resp.confidence_score == 87.5

    def test_defaults_to_general(self):
        resp = ParsePreviewResponse(
            markdown_content="# Hello",
            preview_metadata={},
            source_type=SourceType.MARKDOWN,
        )
        assert resp.content_schema == ContentSchema.GENERAL
        assert resp.confidence_score == 0.0


class TestCommitIngestRequest:
    """CommitIngestRequest should accept a schema override."""

    def test_schema_override(self):
        req = CommitIngestRequest(
            markdown_content="# Test",
            source_type=SourceType.MARKDOWN,
            content_schema=ContentSchema.CASE_STUDY,
            metadata={"author": "Test"},
        )
        assert req.content_schema == ContentSchema.CASE_STUDY

    def test_defaults_to_general(self):
        req = CommitIngestRequest(
            markdown_content="# Test",
            source_type=SourceType.MARKDOWN,
            metadata={},
        )
        assert req.content_schema == ContentSchema.GENERAL


class TestUnifiedKnowledgeLightragFormat:
    """to_lightrag_format() should include the schema in the header."""

    def test_schema_in_header(self):
        k = UnifiedKnowledge(
            source_type="markdown",
            content_body="Some content here",
            metadata=KnowledgeMetadata(
                content_schema="case_study",
                author="Bilal",
            ),
        )
        output = k.to_lightrag_format()
        assert "**Schema:** case_study" in output
        assert "**Author:** Bilal" in output
        assert "Some content here" in output

    def test_default_schema_is_general(self):
        k = UnifiedKnowledge(
            source_type="markdown",
            content_body="Fallback content",
            metadata=KnowledgeMetadata(),
        )
        output = k.to_lightrag_format()
        assert "**Schema:** general" in output
