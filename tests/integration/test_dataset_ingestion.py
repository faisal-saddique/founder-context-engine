"""
Tests that exercise the ingestion pipeline with every piece of
real content from the Dataset directory. Mocks external services
but validates the full parse → classify → commit chain.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from src.models.schemas import ContentSchema


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


def _preview_and_commit(client, content, expected_schema, metadata=None):
    """
    Run the full preview-then-commit cycle for a dataset file.
    Returns the commit response.
    """
    # step 1: preview
    with patch("src.api.routes.ingestion.content_classifier") as mock_cls:
        mock_cls.classify = AsyncMock(
            return_value=(ContentSchema(expected_schema), 90.0)
        )

        preview_resp = client.post("/api/v1/ingest/parse_preview", json={
            "source_type": "markdown",
            "content": content,
            "metadata": metadata,
        })

    assert preview_resp.status_code == 200
    preview = preview_resp.json()
    assert preview["content_schema"] == expected_schema
    assert len(preview["markdown_content"]) > 0

    # step 2: commit
    with patch("src.api.routes.ingestion.lightrag_client") as mock_rag:
        mock_rag.insert_knowledge = AsyncMock(return_value="test-id")

        commit_resp = client.post("/api/v1/ingest/commit_ingest", json={
            "markdown_content": preview["markdown_content"],
            "source_type": "markdown",
            "content_schema": expected_schema,
            "metadata": metadata or {},
            "usage_permission": "public_safe",
            "trust_score": "High",
        })

    assert commit_resp.status_code == 200
    assert commit_resp.json()["success"] is True
    return commit_resp.json()


class TestCanonicalBrandFacts:
    def test_ingest(self, client, canonical_brand_facts):
        result = _preview_and_commit(
            client, canonical_brand_facts, "case_study",
            metadata={"author": "Admin", "type": "canonical_brand_facts"},
        )
        assert result["knowledge_id"] is not None


class TestICPProfiles:
    def test_ingest(self, client, icp_profiles):
        result = _preview_and_commit(
            client, icp_profiles, "profile",
            metadata={"type": "icp"},
        )
        assert result["success"] is True


class TestASOPlaybook:
    def test_ingest(self, client, aso_playbook):
        result = _preview_and_commit(
            client, aso_playbook, "guide",
            metadata={"type": "playbook", "category": "aso"},
        )
        assert result["success"] is True


class TestASOTrends:
    def test_ingest(self, client, aso_trends):
        result = _preview_and_commit(
            client, aso_trends, "market_intel",
            metadata={"type": "industry_news"},
        )
        assert result["success"] is True


class TestCleanerCaseStudy:
    def test_ingest(self, client, cleaner_case_study):
        result = _preview_and_commit(
            client, cleaner_case_study, "case_study",
            metadata={"type": "app_insight", "app": "cleaner"},
        )
        assert result["success"] is True


class TestFormsCaseStudy:
    def test_ingest(self, client, forms_case_study):
        result = _preview_and_commit(
            client, forms_case_study, "case_study",
            metadata={"type": "app_insight", "app": "forms"},
        )
        assert result["success"] is True


class TestLinkedinRules:
    def test_ingest(self, client, linkedin_rules):
        result = _preview_and_commit(
            client, linkedin_rules, "guide",
            metadata={"type": "platform_rules", "platform": "linkedin"},
        )
        assert result["success"] is True


class TestProposalStyleGuidelines:
    def test_ingest(self, client, proposal_style_guidelines):
        if not proposal_style_guidelines:
            pytest.skip("Proposal style guidelines file not found")
        result = _preview_and_commit(
            client, proposal_style_guidelines, "guide",
            metadata={"type": "style_guide"},
        )
        assert result["success"] is True


class TestPreviewMetadataQuality:
    """Verify the preview response provides useful info to the frontend."""

    def test_parsed_length_is_accurate(self, client, canonical_brand_facts):
        with patch("src.api.routes.ingestion.content_classifier") as mock_cls:
            mock_cls.classify = AsyncMock(
                return_value=(ContentSchema.CASE_STUDY, 92.0)
            )

            resp = client.post("/api/v1/ingest/parse_preview", json={
                "source_type": "markdown",
                "content": canonical_brand_facts,
            })

        data = resp.json()
        meta = data["preview_metadata"]
        assert meta["parsed_length"] == len(data["markdown_content"])

    def test_original_content_truncated(self, client, canonical_brand_facts):
        """If content is long, original_content in metadata should be truncated."""
        with patch("src.api.routes.ingestion.content_classifier") as mock_cls:
            mock_cls.classify = AsyncMock(
                return_value=(ContentSchema.GENERAL, 50.0)
            )

            resp = client.post("/api/v1/ingest/parse_preview", json={
                "source_type": "markdown",
                "content": canonical_brand_facts,
            })

        meta = resp.json()["preview_metadata"]
        # should be truncated with "..."
        if len(canonical_brand_facts) > 200:
            assert meta["original_content"].endswith("...")
            assert len(meta["original_content"]) <= 203  # 200 + "..."

    def test_short_content_not_truncated(self, client):
        with patch("src.api.routes.ingestion.content_classifier") as mock_cls:
            mock_cls.classify = AsyncMock(
                return_value=(ContentSchema.GENERAL, 50.0)
            )

            resp = client.post("/api/v1/ingest/parse_preview", json={
                "source_type": "markdown",
                "content": "# Short",
            })

        meta = resp.json()["preview_metadata"]
        assert meta["original_content"] == "# Short"
        assert "..." not in meta["original_content"]
