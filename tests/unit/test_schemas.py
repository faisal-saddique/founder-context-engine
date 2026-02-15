"""
Thorough unit tests for all Pydantic request/response models and enums.
Validates every enum value, default behaviour, required fields, and edge cases.
"""
import pytest
from datetime import datetime

from src.models.schemas import (
    SourceType,
    ContentSchema,
    UsagePermission,
    TrustScore,
    PostFormat,
    Platform,
    ParsePreviewRequest,
    ParsePreviewResponse,
    CommitIngestRequest,
    CommitIngestResponse,
    GenerateRequest,
    GenerateResponse,
    ValidateRequest,
    ValidateResponse,
    ValidationIssue,
    HealthResponse,
)


# ── Enum completeness ─────────────────────────────────────────────

class TestSourceTypeEnum:
    def test_all_source_types_present(self):
        expected = {"pdf_deck", "youtube_summary", "app_store_link",
                    "web_url", "research", "markdown"}
        assert {s.value for s in SourceType} == expected

    def test_count(self):
        assert len(SourceType) == 6

    def test_string_lookup(self):
        assert SourceType("markdown") == SourceType.MARKDOWN


class TestContentSchemaEnum:
    def test_five_schemas(self):
        assert len(ContentSchema) == 5

    def test_values(self):
        expected = {"case_study", "profile", "guide", "market_intel", "general"}
        assert {s.value for s in ContentSchema} == expected


class TestUsagePermissionEnum:
    def test_values(self):
        assert UsagePermission.PUBLIC_SAFE.value == "public_safe"
        assert UsagePermission.INTERNAL_ONLY.value == "internal_only"
        assert len(UsagePermission) == 2


class TestTrustScoreEnum:
    def test_values(self):
        assert TrustScore.HIGH.value == "High"
        assert TrustScore.MEDIUM.value == "Medium"
        assert TrustScore.LOW.value == "Low"
        assert len(TrustScore) == 3


class TestPostFormatEnum:
    def test_formats(self):
        expected = {"comparison", "deep_dive", "story", "general"}
        assert {f.value for f in PostFormat} == expected


class TestPlatformEnum:
    def test_platforms(self):
        expected = {"linkedin", "twitter", "upwork"}
        assert {p.value for p in Platform} == expected


# ── ParsePreview models ───────────────────────────────────────────

class TestParsePreviewRequest:
    def test_valid_minimal(self):
        req = ParsePreviewRequest(source_type=SourceType.MARKDOWN, content="# Hello")
        assert req.source_type == SourceType.MARKDOWN
        assert req.content == "# Hello"
        assert req.metadata is None

    def test_with_metadata(self):
        req = ParsePreviewRequest(
            source_type=SourceType.PDF_DECK,
            content="/path/to/file.pdf",
            metadata={"author": "Alice"},
        )
        assert req.metadata["author"] == "Alice"

    def test_missing_content_raises(self):
        with pytest.raises(Exception):
            ParsePreviewRequest(source_type=SourceType.MARKDOWN)


class TestParsePreviewResponse:
    def test_defaults(self):
        resp = ParsePreviewResponse(
            markdown_content="# Hi",
            preview_metadata={},
            source_type=SourceType.MARKDOWN,
        )
        assert resp.content_schema == ContentSchema.GENERAL
        assert resp.confidence_score == 0.0

    def test_explicit_schema(self):
        resp = ParsePreviewResponse(
            markdown_content="# Case",
            preview_metadata={"detected_schema": "case_study"},
            source_type=SourceType.PDF_DECK,
            content_schema=ContentSchema.CASE_STUDY,
            confidence_score=87.5,
        )
        assert resp.content_schema == ContentSchema.CASE_STUDY
        assert resp.confidence_score == 87.5


# ── CommitIngest models ───────────────────────────────────────────

class TestCommitIngestRequest:
    def test_defaults(self):
        req = CommitIngestRequest(
            markdown_content="# Test",
            source_type=SourceType.MARKDOWN,
            metadata={},
        )
        assert req.content_schema == ContentSchema.GENERAL
        assert req.usage_permission == UsagePermission.PUBLIC_SAFE
        assert req.trust_score == TrustScore.MEDIUM

    def test_overrides(self):
        req = CommitIngestRequest(
            markdown_content="# Study",
            source_type=SourceType.PDF_DECK,
            content_schema=ContentSchema.CASE_STUDY,
            metadata={"author": "Jane"},
            usage_permission=UsagePermission.INTERNAL_ONLY,
            trust_score=TrustScore.HIGH,
        )
        assert req.content_schema == ContentSchema.CASE_STUDY
        assert req.usage_permission == UsagePermission.INTERNAL_ONLY
        assert req.trust_score == TrustScore.HIGH

    def test_missing_metadata_raises(self):
        with pytest.raises(Exception):
            CommitIngestRequest(
                markdown_content="# Test",
                source_type=SourceType.MARKDOWN,
            )


class TestCommitIngestResponse:
    def test_creation(self):
        resp = CommitIngestResponse(
            success=True,
            knowledge_id="abc-123",
            message="OK",
        )
        assert resp.success is True
        assert resp.knowledge_id == "abc-123"


# ── Generate models ───────────────────────────────────────────────

class TestGenerateRequest:
    def test_defaults(self):
        req = GenerateRequest(platform=Platform.LINKEDIN)
        assert req.post_format == PostFormat.GENERAL
        assert req.tone == "professional"
        assert req.specific_resource_context is None
        assert req.custom_instructions is None

    def test_full(self):
        req = GenerateRequest(
            platform=Platform.TWITTER,
            post_format=PostFormat.DEEP_DIVE,
            tone="casual",
            specific_resource_context="ASO trends",
            custom_instructions="Keep under 280 chars",
        )
        assert req.platform == Platform.TWITTER
        assert req.post_format == PostFormat.DEEP_DIVE


class TestGenerateResponse:
    def test_creation(self):
        resp = GenerateResponse(
            content="Generated content",
            generation_id="gen-123",
            sources_used=["s1"],
            rules_applied=["r1"],
            metadata={"validation_passed": True},
        )
        assert resp.content == "Generated content"
        assert len(resp.sources_used) == 1


# ── Validate models ──────────────────────────────────────────────

class TestValidationIssue:
    def test_fields(self):
        issue = ValidationIssue(
            claim="We have 1000 users",
            issue_type="unverified_number",
            severity="high",
            suggestion="Remove or cite source",
        )
        assert issue.severity == "high"

    def test_optional_suggestion(self):
        issue = ValidationIssue(
            claim="The best",
            issue_type="unverified_superlative",
            severity="high",
        )
        assert issue.suggestion is None


class TestValidateRequest:
    def test_creation(self):
        req = ValidateRequest(
            content="We have 500 users",
            sources=[{"content": "500 users", "trust_score": "High"}],
        )
        assert len(req.sources) == 1


class TestValidateResponse:
    def test_defaults(self):
        resp = ValidateResponse(is_valid=True, issues=[])
        assert resp.is_valid is True
        assert isinstance(resp.validated_at, datetime)


# ── Health model ──────────────────────────────────────────────────

class TestHealthResponse:
    def test_defaults(self):
        resp = HealthResponse(
            status="healthy",
            environment="test",
            services={"neo4j": True},
        )
        assert resp.version == "1.0.0"
        assert resp.services["neo4j"] is True
