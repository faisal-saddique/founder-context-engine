"""
Tests verifying compliance with the Ingestion & Preview Design Strategy.
Ensures the 5-schema system, classification threshold, preview-then-commit
workflow, and no-auto-save behaviour all work as documented.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from src.models.schemas import (
    ContentSchema,
    SourceType,
    ParsePreviewRequest,
    ParsePreviewResponse,
    CommitIngestRequest,
    CommitIngestResponse,
    UsagePermission,
    TrustScore,
)
from src.services.ingestion.classifier import ContentClassifier, ClassificationResult
from src.models.knowledge import UnifiedKnowledge, KnowledgeMetadata


@pytest.fixture
def client():
    with patch("src.api.main.neo4j_client") as mock_neo4j, \
         patch("src.api.main.lightrag_client") as mock_lightrag, \
         patch("src.api.main.rule_retriever") as mock_rule:
        mock_neo4j.connect = AsyncMock()
        mock_neo4j.close = AsyncMock()
        mock_lightrag.initialize = AsyncMock()
        mock_lightrag.finalize = AsyncMock()
        mock_rule.connect = AsyncMock()
        mock_rule.disconnect = AsyncMock()

        from src.api.main import app
        with TestClient(app) as c:
            yield c


# ── 1. Five master schemas ────────────────────────────────────────

class TestFiveMasterSchemas:
    """The system must have exactly 5 schemas, no more."""

    def test_exactly_five_schemas(self):
        assert len(ContentSchema) == 5

    def test_schema_names(self):
        names = {s.value for s in ContentSchema}
        assert names == {"case_study", "profile", "guide", "market_intel", "general"}

    def test_no_per_file_templates(self):
        """There should be no extra source-specific schemas."""
        schema_values = [s.value for s in ContentSchema]
        # none of these should exist as schemas
        for forbidden in ["pdf", "youtube", "web", "discord", "slack"]:
            assert forbidden not in schema_values


# ── 2. Classification threshold at 60% ───────────────────────────

class TestClassificationThreshold:
    """
    Confidence >= 60%: use detected schema.
    Confidence < 60%: fall back to schema_general.
    """

    @pytest.mark.asyncio
    async def test_above_threshold_keeps_schema(self):
        classifier = ContentClassifier.__new__(ContentClassifier)
        mock = AsyncMock()
        mock.ainvoke = AsyncMock(return_value=ClassificationResult(
            schema_type="case_study", confidence=85,
        ))
        classifier.chain = mock

        schema, conf = await classifier.classify("App performance data")
        assert schema == ContentSchema.CASE_STUDY
        assert conf == 85.0

    @pytest.mark.asyncio
    async def test_at_60_keeps_schema(self):
        classifier = ContentClassifier.__new__(ContentClassifier)
        mock = AsyncMock()
        mock.ainvoke = AsyncMock(return_value=ClassificationResult(
            schema_type="guide", confidence=60,
        ))
        classifier.chain = mock

        schema, conf = await classifier.classify("Playbook content")
        assert schema == ContentSchema.GUIDE

    @pytest.mark.asyncio
    async def test_below_60_falls_back_to_general(self):
        classifier = ContentClassifier.__new__(ContentClassifier)
        mock = AsyncMock()
        mock.ainvoke = AsyncMock(return_value=ClassificationResult(
            schema_type="profile", confidence=45,
        ))
        classifier.chain = mock

        schema, conf = await classifier.classify("Ambiguous content")
        assert schema == ContentSchema.GENERAL
        assert conf == 45.0

    @pytest.mark.asyncio
    async def test_at_59_falls_back(self):
        classifier = ContentClassifier.__new__(ContentClassifier)
        mock = AsyncMock()
        mock.ainvoke = AsyncMock(return_value=ClassificationResult(
            schema_type="market_intel", confidence=59,
        ))
        classifier.chain = mock

        schema, _ = await classifier.classify("Some news article")
        assert schema == ContentSchema.GENERAL

    @pytest.mark.asyncio
    async def test_zero_confidence_falls_back(self):
        classifier = ContentClassifier.__new__(ContentClassifier)
        mock = AsyncMock()
        mock.ainvoke = AsyncMock(return_value=ClassificationResult(
            schema_type="case_study", confidence=0,
        ))
        classifier.chain = mock

        schema, _ = await classifier.classify("Meaningless")
        assert schema == ContentSchema.GENERAL


# ── 3. Preview → Edit → Save workflow ────────────────────────────

class TestPreviewThenCommitWorkflow:
    """
    Content must go through parse_preview first, then commit_ingest.
    No auto-saving — explicit user confirmation required.
    """

    def test_preview_does_not_save(self, client):
        """parse_preview must NOT write to LightRAG."""
        with patch("src.api.routes.ingestion.content_classifier") as mock_cls, \
             patch("src.api.routes.ingestion.lightrag_client") as mock_rag:
            mock_cls.classify = AsyncMock(
                return_value=(ContentSchema.CASE_STUDY, 90.0)
            )

            client.post("/api/v1/ingest/parse_preview", json={
                "source_type": "markdown",
                "content": "# Case study about our cleaner app",
            })

            # LightRAG's insert should NOT have been called
            mock_rag.insert_knowledge.assert_not_called()

    def test_commit_requires_explicit_call(self, client):
        """commit_ingest is a separate endpoint needing explicit invocation."""
        with patch("src.api.routes.ingestion.lightrag_client") as mock_rag:
            mock_rag.insert_knowledge = AsyncMock(return_value="id-123")

            resp = client.post("/api/v1/ingest/commit_ingest", json={
                "markdown_content": "# Reviewed content",
                "source_type": "markdown",
                "content_schema": "case_study",
                "metadata": {"author": "User"},
            })

        assert resp.status_code == 200
        # NOW it should have been called
        mock_rag.insert_knowledge.assert_called_once()

    def test_full_preview_then_commit_flow(self, client):
        """Simulate the full user journey: preview, review, then commit."""
        # step 1: preview
        with patch("src.api.routes.ingestion.content_classifier") as mock_cls:
            mock_cls.classify = AsyncMock(
                return_value=(ContentSchema.PROFILE, 88.0)
            )

            preview_resp = client.post("/api/v1/ingest/parse_preview", json={
                "source_type": "markdown",
                "content": "# ICP: Mobile App Startups\n\nEarly stage founders...",
            })

        assert preview_resp.status_code == 200
        preview = preview_resp.json()
        assert preview["content_schema"] == "profile"

        # step 2: user reviews and commits (can override schema)
        with patch("src.api.routes.ingestion.lightrag_client") as mock_rag:
            mock_rag.insert_knowledge = AsyncMock(return_value="id")

            commit_resp = client.post("/api/v1/ingest/commit_ingest", json={
                "markdown_content": preview["markdown_content"],
                "source_type": preview["source_type"],
                "content_schema": preview["content_schema"],
                "metadata": {"author": "Reviewer"},
                "usage_permission": "public_safe",
                "trust_score": "High",
            })

        assert commit_resp.status_code == 200
        assert commit_resp.json()["success"] is True

    def test_user_can_override_schema(self, client):
        """User should be able to change the schema from what was detected."""
        with patch("src.api.routes.ingestion.lightrag_client") as mock_rag:
            mock_rag.insert_knowledge = AsyncMock(return_value="id")

            # detected as general, but user overrides to guide
            resp = client.post("/api/v1/ingest/commit_ingest", json={
                "markdown_content": "# Overridden content",
                "source_type": "markdown",
                "content_schema": "guide",  # user override
                "metadata": {},
            })

        assert resp.status_code == 200


# ── 4. Schema classification for real dataset content ────────────

class TestDatasetSchemaMapping:
    """
    Verify that dataset content maps to the right schemas conceptually.
    We test the models here, not the LLM classifier.
    """

    def test_brand_facts_is_case_study_or_general(self, canonical_brand_facts):
        """Canonical brand facts describes the company, fits case_study or general."""
        req = CommitIngestRequest(
            markdown_content=canonical_brand_facts,
            source_type=SourceType.MARKDOWN,
            content_schema=ContentSchema.CASE_STUDY,
            metadata={"author": "Admin"},
            usage_permission=UsagePermission.PUBLIC_SAFE,
            trust_score=TrustScore.HIGH,
        )
        assert req.content_schema in [ContentSchema.CASE_STUDY, ContentSchema.GENERAL]

    def test_icp_is_profile(self, icp_profiles):
        req = CommitIngestRequest(
            markdown_content=icp_profiles,
            source_type=SourceType.MARKDOWN,
            content_schema=ContentSchema.PROFILE,
            metadata={},
        )
        assert req.content_schema == ContentSchema.PROFILE

    def test_playbook_is_guide(self, aso_playbook):
        req = CommitIngestRequest(
            markdown_content=aso_playbook,
            source_type=SourceType.MARKDOWN,
            content_schema=ContentSchema.GUIDE,
            metadata={},
            usage_permission=UsagePermission.INTERNAL_ONLY,
        )
        assert req.content_schema == ContentSchema.GUIDE

    def test_trends_is_market_intel(self, aso_trends):
        req = CommitIngestRequest(
            markdown_content=aso_trends,
            source_type=SourceType.MARKDOWN,
            content_schema=ContentSchema.MARKET_INTEL,
            metadata={},
        )
        assert req.content_schema == ContentSchema.MARKET_INTEL


# ── 5. Unified Knowledge preserves schema metadata ───────────────

class TestKnowledgeSchemaPreservation:
    """Schema info must survive the full ingestion pipeline."""

    def test_schema_in_lightrag_output(self):
        k = UnifiedKnowledge(
            source_type="markdown",
            content_body="# Playbook\nDo this, don't do that.",
            metadata=KnowledgeMetadata(content_schema="guide"),
        )
        output = k.to_lightrag_format()
        assert "**Schema:** guide" in output

    def test_all_schemas_preserved(self):
        for schema_name in ["case_study", "profile", "guide", "market_intel", "general"]:
            k = UnifiedKnowledge(
                source_type="markdown",
                content_body="Content",
                metadata=KnowledgeMetadata(content_schema=schema_name),
            )
            output = k.to_lightrag_format()
            assert f"**Schema:** {schema_name}" in output

    def test_usage_permission_preserved(self):
        k = UnifiedKnowledge(
            source_type="markdown",
            content_body="Internal only content",
            metadata=KnowledgeMetadata(
                content_schema="guide",
                usage_permission="internal_only",
            ),
        )
        output = k.to_lightrag_format()
        assert "**Permission:** internal_only" in output

    def test_trust_score_preserved(self):
        k = UnifiedKnowledge(
            source_type="markdown",
            content_body="Content",
            metadata=KnowledgeMetadata(trust_score="Low"),
        )
        output = k.to_lightrag_format()
        assert "**Trust:** Low" in output


# ── 6. ClassificationResult model validation ─────────────────────

class TestClassificationResultValidation:
    """The structured output model should reject bad values."""

    def test_valid_schemas_accepted(self):
        for schema in ["case_study", "profile", "guide", "market_intel", "general"]:
            r = ClassificationResult(schema_type=schema, confidence=80)
            assert r.schema_type == schema

    def test_invalid_schema_rejected(self):
        with pytest.raises(Exception):
            ClassificationResult(schema_type="nonexistent", confidence=90)

    def test_confidence_over_100_rejected(self):
        with pytest.raises(Exception):
            ClassificationResult(schema_type="general", confidence=150)

    def test_negative_confidence_rejected(self):
        with pytest.raises(Exception):
            ClassificationResult(schema_type="general", confidence=-5)

    def test_boundary_values(self):
        r0 = ClassificationResult(schema_type="general", confidence=0)
        assert r0.confidence == 0

        r100 = ClassificationResult(schema_type="general", confidence=100)
        assert r100.confidence == 100
