"""
Unit tests for the UnifiedKnowledge and KnowledgeMetadata models.
Tests creation, defaults, LightRAG format, and edge cases with
real dataset content.
"""
import pytest
import uuid

from src.models.knowledge import UnifiedKnowledge, KnowledgeMetadata


# ── KnowledgeMetadata ─────────────────────────────────────────────

class TestKnowledgeMetadata:
    def test_defaults(self):
        meta = KnowledgeMetadata()
        assert meta.author is None
        assert meta.usage_permission == "public_safe"
        assert meta.trust_score == "Medium"
        assert meta.content_type == "general"
        assert meta.content_schema == "general"
        assert meta.file_name is None
        assert meta.source_url is None
        assert meta.tags == []

    def test_full_metadata(self):
        meta = KnowledgeMetadata(
            author="Bilal",
            usage_permission="internal_only",
            trust_score="High",
            content_type="case_study",
            content_schema="case_study",
            file_name="cleaner.md",
            source_url="https://example.com",
            tags=["aso", "mobile"],
        )
        assert meta.author == "Bilal"
        assert meta.tags == ["aso", "mobile"]


# ── UnifiedKnowledge creation ─────────────────────────────────────

class TestUnifiedKnowledgeCreation:
    def test_auto_id(self):
        k = UnifiedKnowledge(
            source_type="markdown",
            content_body="# Hello",
            metadata=KnowledgeMetadata(),
        )
        assert k.id is not None
        # should be a valid UUID
        uuid.UUID(k.id)

    def test_auto_timestamp(self):
        k = UnifiedKnowledge(
            source_type="pdf_deck",
            content_body="Content",
            metadata=KnowledgeMetadata(),
        )
        assert k.created_at is not None
        assert "T" in k.created_at  # ISO format

    def test_explicit_fields(self):
        k = UnifiedKnowledge(
            source_type="web_url",
            content_body="# Web Content",
            metadata=KnowledgeMetadata(author="Test"),
        )
        assert k.source_type == "web_url"
        assert k.content_body == "# Web Content"


# ── LightRAG format conversion ────────────────────────────────────

class TestLightRAGFormat:
    def test_basic_conversion(self):
        k = UnifiedKnowledge(
            source_type="pdf_deck",
            content_body="Test content here",
            metadata=KnowledgeMetadata(
                author="John Doe",
                usage_permission="public_safe",
                trust_score="High",
                source_url="https://example.com/doc.pdf",
            ),
        )
        output = k.to_lightrag_format()
        assert "# pdf_deck" in output
        assert f"**ID:** {k.id}" in output
        assert "**Author:** John Doe" in output
        assert "**Permission:** public_safe" in output
        assert "**Trust:** High" in output
        assert "**Source:** https://example.com/doc.pdf" in output
        assert "---" in output
        assert "Test content here" in output

    def test_no_source_url_omitted(self):
        k = UnifiedKnowledge(
            source_type="markdown",
            content_body="Content",
            metadata=KnowledgeMetadata(),
        )
        output = k.to_lightrag_format()
        assert "**Source:**" not in output

    def test_unknown_author_shows_unknown(self):
        k = UnifiedKnowledge(
            source_type="markdown",
            content_body="Content",
            metadata=KnowledgeMetadata(),
        )
        output = k.to_lightrag_format()
        assert "**Author:** Unknown" in output

    def test_schema_in_header(self):
        k = UnifiedKnowledge(
            source_type="markdown",
            content_body="Some content",
            metadata=KnowledgeMetadata(content_schema="case_study"),
        )
        output = k.to_lightrag_format()
        assert "**Schema:** case_study" in output

    def test_default_schema_general(self):
        k = UnifiedKnowledge(
            source_type="markdown",
            content_body="Fallback content",
            metadata=KnowledgeMetadata(),
        )
        output = k.to_lightrag_format()
        assert "**Schema:** general" in output


# ── with real dataset content ─────────────────────────────────────

class TestKnowledgeWithDataset:
    def test_case_study_format(self, cleaner_case_study):
        k = UnifiedKnowledge(
            source_type="markdown",
            content_body=cleaner_case_study,
            metadata=KnowledgeMetadata(
                content_schema="case_study",
                usage_permission="internal_only",
                trust_score="High",
                tags=["utility", "cleaner"],
            ),
        )
        output = k.to_lightrag_format()
        assert "**Schema:** case_study" in output
        assert "**Permission:** internal_only" in output
        assert "Utilities" in output or "Cleaner" in output

    def test_profile_format(self, icp_profiles):
        k = UnifiedKnowledge(
            source_type="markdown",
            content_body=icp_profiles,
            metadata=KnowledgeMetadata(
                content_schema="profile",
                trust_score="High",
            ),
        )
        output = k.to_lightrag_format()
        assert "**Schema:** profile" in output
        assert "Mobile App Startups" in output

    def test_guide_format(self, aso_playbook):
        k = UnifiedKnowledge(
            source_type="markdown",
            content_body=aso_playbook,
            metadata=KnowledgeMetadata(
                content_schema="guide",
                usage_permission="internal_only",
            ),
        )
        output = k.to_lightrag_format()
        assert "**Schema:** guide" in output
        assert "ASO" in output

    def test_market_intel_format(self, aso_trends):
        k = UnifiedKnowledge(
            source_type="markdown",
            content_body=aso_trends,
            metadata=KnowledgeMetadata(
                content_schema="market_intel",
                trust_score="Medium",
            ),
        )
        output = k.to_lightrag_format()
        assert "**Schema:** market_intel" in output
        assert "Trends" in output or "ASO" in output
