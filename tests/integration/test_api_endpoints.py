"""
Integration tests for all API endpoints.
Uses FastAPI TestClient with mocked backend services so we can
exercise the full HTTP request/response cycle without real databases.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from src.models.schemas import ContentSchema, SourceType


# We need to patch the lifespan so it doesn't try to connect real DBs
@pytest.fixture
def client():
    """TestClient with mocked lifespan dependencies."""
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


# ── Root endpoint ────────────────────────────────────────────────

class TestRootEndpoint:
    def test_returns_api_info(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Founder Context Engine API"
        assert data["version"] == "1.0.0"
        assert "docs" in data
        assert "health" in data


# ── Health endpoint ──────────────────────────────────────────────

class TestHealthEndpoint:
    def test_returns_healthy(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "services" in data
        assert "neo4j" in data["services"]
        assert "postgresql" in data["services"]

    def test_version_field(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert data["version"] == "1.0.0"


# ── Parse Preview endpoint ──────────────────────────────────────

class TestParsePreviewEndpoint:
    def test_markdown_parse_preview(self, client):
        """Markdown source type should pass through and get classified."""
        with patch("src.api.routes.ingestion.content_classifier") as mock_cls:
            mock_cls.classify = AsyncMock(
                return_value=(ContentSchema.CASE_STUDY, 85.0)
            )

            resp = client.post("/api/v1/ingest/parse_preview", json={
                "source_type": "markdown",
                "content": "# Cleaner App Case Study\n\nWe scaled a utility app.",
            })

        assert resp.status_code == 200
        data = resp.json()
        assert "markdown_content" in data
        assert data["content_schema"] == "case_study"
        assert data["confidence_score"] == 85.0
        assert data["source_type"] == "markdown"

    def test_preview_returns_metadata(self, client):
        with patch("src.api.routes.ingestion.content_classifier") as mock_cls:
            mock_cls.classify = AsyncMock(
                return_value=(ContentSchema.GUIDE, 75.0)
            )

            resp = client.post("/api/v1/ingest/parse_preview", json={
                "source_type": "markdown",
                "content": "# ASO Playbook\n\nStep 1: Optimize keywords.",
            })

        data = resp.json()
        assert "preview_metadata" in data
        meta = data["preview_metadata"]
        assert meta["source_type"] == "markdown"
        assert meta["detected_schema"] == "guide"
        assert meta["confidence_score"] == 75.0
        assert "parsed_length" in meta

    def test_preview_with_web_url_source(self, client):
        """Web URL should use WebParser (mocked) and classify."""
        with patch("src.api.routes.ingestion.ParserFactory") as mock_factory, \
             patch("src.api.routes.ingestion.content_classifier") as mock_cls:
            mock_parser = MagicMock()
            mock_parser.parse = AsyncMock(return_value="# Scraped Content\n\nImportant info.")
            mock_factory.create_parser = MagicMock(return_value=mock_parser)
            mock_cls.classify = AsyncMock(
                return_value=(ContentSchema.MARKET_INTEL, 72.0)
            )

            resp = client.post("/api/v1/ingest/parse_preview", json={
                "source_type": "web_url",
                "content": "https://example.com/article",
            })

        assert resp.status_code == 200
        data = resp.json()
        assert data["content_schema"] == "market_intel"
        assert "Scraped Content" in data["markdown_content"]

    def test_preview_with_youtube_source(self, client):
        with patch("src.api.routes.ingestion.ParserFactory") as mock_factory, \
             patch("src.api.routes.ingestion.content_classifier") as mock_cls:
            mock_parser = MagicMock()
            mock_parser.parse = AsyncMock(return_value="# YouTube Transcript\n\nContent here.")
            mock_factory.create_parser = MagicMock(return_value=mock_parser)
            mock_cls.classify = AsyncMock(
                return_value=(ContentSchema.MARKET_INTEL, 80.0)
            )

            resp = client.post("/api/v1/ingest/parse_preview", json={
                "source_type": "youtube_summary",
                "content": "https://youtube.com/watch?v=abc123",
            })

        assert resp.status_code == 200
        assert resp.json()["source_type"] == "youtube_summary"

    def test_preview_invalid_source_type(self, client):
        resp = client.post("/api/v1/ingest/parse_preview", json={
            "source_type": "invalid_type",
            "content": "something",
        })
        assert resp.status_code == 422  # validation error

    def test_preview_missing_content(self, client):
        resp = client.post("/api/v1/ingest/parse_preview", json={
            "source_type": "markdown",
        })
        assert resp.status_code == 422

    def test_preview_parser_error_returns_400(self, client):
        """When the parser raises ParsingError, endpoint should return 400."""
        from src.core.exceptions import ParsingError

        with patch("src.api.routes.ingestion.ParserFactory") as mock_factory:
            mock_parser = MagicMock()
            mock_parser.parse = AsyncMock(side_effect=ParsingError("Empty content"))
            mock_factory.create_parser = MagicMock(return_value=mock_parser)

            resp = client.post("/api/v1/ingest/parse_preview", json={
                "source_type": "markdown",
                "content": "",
            })

        assert resp.status_code == 400
        assert "Empty content" in resp.json()["detail"]

    def test_preview_with_metadata(self, client):
        with patch("src.api.routes.ingestion.content_classifier") as mock_cls:
            mock_cls.classify = AsyncMock(
                return_value=(ContentSchema.GENERAL, 50.0)
            )

            resp = client.post("/api/v1/ingest/parse_preview", json={
                "source_type": "markdown",
                "content": "# Test",
                "metadata": {"author": "TestUser", "tags": ["test"]},
            })

        assert resp.status_code == 200
        meta = resp.json()["preview_metadata"]
        assert meta["custom_metadata"]["author"] == "TestUser"


# ── Commit Ingest endpoint ───────────────────────────────────────

class TestCommitIngestEndpoint:
    def test_successful_commit(self, client):
        with patch("src.api.routes.ingestion.lightrag_client") as mock_rag:
            mock_rag.insert_knowledge = AsyncMock(return_value="test-id")

            resp = client.post("/api/v1/ingest/commit_ingest", json={
                "markdown_content": "# Canonical Brand Facts\n\nWe are a mobile agency.",
                "source_type": "markdown",
                "content_schema": "case_study",
                "metadata": {"author": "Admin", "tags": ["brand"]},
                "usage_permission": "public_safe",
                "trust_score": "High",
            })

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "knowledge_id" in data
        assert len(data["knowledge_id"]) > 0

    def test_commit_with_defaults(self, client):
        with patch("src.api.routes.ingestion.lightrag_client") as mock_rag:
            mock_rag.insert_knowledge = AsyncMock(return_value="test-id")

            resp = client.post("/api/v1/ingest/commit_ingest", json={
                "markdown_content": "# Test",
                "source_type": "markdown",
                "metadata": {},
            })

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True

    def test_commit_missing_markdown_content(self, client):
        resp = client.post("/api/v1/ingest/commit_ingest", json={
            "source_type": "markdown",
            "metadata": {},
        })
        assert resp.status_code == 422

    def test_commit_invalid_schema(self, client):
        resp = client.post("/api/v1/ingest/commit_ingest", json={
            "markdown_content": "# Test",
            "source_type": "markdown",
            "content_schema": "nonexistent",
            "metadata": {},
        })
        assert resp.status_code == 422

    def test_commit_lightrag_error_returns_500(self, client):
        from src.core.exceptions import IngestionError

        with patch("src.api.routes.ingestion.lightrag_client") as mock_rag:
            mock_rag.insert_knowledge = AsyncMock(
                side_effect=IngestionError("DB connection failed")
            )

            resp = client.post("/api/v1/ingest/commit_ingest", json={
                "markdown_content": "# Test",
                "source_type": "markdown",
                "metadata": {},
            })

        assert resp.status_code == 500

    def test_commit_all_schemas(self, client):
        """Every schema type should be accepted."""
        for schema in ["case_study", "profile", "guide", "market_intel", "general"]:
            with patch("src.api.routes.ingestion.lightrag_client") as mock_rag:
                mock_rag.insert_knowledge = AsyncMock(return_value="id")

                resp = client.post("/api/v1/ingest/commit_ingest", json={
                    "markdown_content": f"# {schema} content",
                    "source_type": "markdown",
                    "content_schema": schema,
                    "metadata": {},
                })

            assert resp.status_code == 200, f"Failed for schema: {schema}"


# ── Validate endpoint ────────────────────────────────────────────

class TestValidateEndpoint:
    def test_clean_content(self, client):
        resp = client.post("/api/v1/generate/validate", json={
            "content": "We help app creators grow their products.",
            "sources": [],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_valid"] is True
        assert data["issues"] == []

    def test_content_with_unverified_claims(self, client):
        resp = client.post("/api/v1/generate/validate", json={
            "content": "We are the best agency with 10,000 clients.",
            "sources": [{"content": "We are an agency.", "trust_score": "High"}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["issues"]) > 0

    def test_content_verified_by_sources(self, client):
        resp = client.post("/api/v1/generate/validate", json={
            "content": "Team size: 25 developers.",
            "sources": [
                {"content": "Team size: 25+ developers", "trust_score": "High", "type": "canonical_brand_facts"},
            ],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_valid"] is True

    def test_validate_has_timestamp(self, client):
        resp = client.post("/api/v1/generate/validate", json={
            "content": "Simple test.",
            "sources": [],
        })
        data = resp.json()
        assert "validated_at" in data

    def test_validate_empty_content(self, client):
        resp = client.post("/api/v1/generate/validate", json={
            "content": "",
            "sources": [],
        })
        assert resp.status_code == 200
        assert resp.json()["is_valid"] is True


# ── Generate endpoint ────────────────────────────────────────────

class TestGenerateEndpoint:
    def test_successful_generation(self, client):
        with patch("src.api.routes.generation.content_workflow") as mock_wf:
            mock_wf.generate = AsyncMock(return_value={
                "final_content": "Here's a great LinkedIn post about ASO.",
                "generation_id": "gen-123",
                "source_ids": ["src-1"],
                "rule_ids_applied": ["rule-1"],
                "validation_passed": True,
                "validation_issues": [],
                "critique_result": {"score": 8, "feedback": "Good"},
                "model_config": {"model": "gpt-4"},
            })

            # also mock the Prisma logging
            with patch("src.api.routes.generation.Prisma") as mock_prisma:
                mock_db = MagicMock()
                mock_db.connect = AsyncMock()
                mock_db.disconnect = AsyncMock()
                mock_db.generation = MagicMock()
                mock_db.generation.create = AsyncMock(return_value=MagicMock(id="log-1"))
                mock_prisma.return_value = mock_db

                resp = client.post("/api/v1/generate/", json={
                    "platform": "linkedin",
                    "post_format": "deep_dive",
                    "tone": "professional",
                })

        assert resp.status_code == 200
        data = resp.json()
        assert "LinkedIn post" in data["content"] or len(data["content"]) > 0
        assert data["generation_id"] == "gen-123"
        assert "src-1" in data["sources_used"]
        assert data["metadata"]["validation_passed"] is True

    def test_generate_invalid_platform(self, client):
        resp = client.post("/api/v1/generate/", json={
            "platform": "tiktok",
        })
        assert resp.status_code == 422

    def test_generate_all_platforms(self, client):
        for platform in ["linkedin", "twitter", "upwork"]:
            with patch("src.api.routes.generation.content_workflow") as mock_wf, \
                 patch("src.api.routes.generation.Prisma") as mock_prisma:
                mock_wf.generate = AsyncMock(return_value={
                    "final_content": f"Post for {platform}",
                    "generation_id": "gen",
                    "source_ids": [],
                    "rule_ids_applied": [],
                    "validation_passed": True,
                    "validation_issues": [],
                    "critique_result": {"score": 7},
                    "model_config": {},
                })
                mock_db = MagicMock()
                mock_db.connect = AsyncMock()
                mock_db.disconnect = AsyncMock()
                mock_db.generation = MagicMock()
                mock_db.generation.create = AsyncMock(return_value=MagicMock())
                mock_prisma.return_value = mock_db

                resp = client.post("/api/v1/generate/", json={
                    "platform": platform,
                })

            assert resp.status_code == 200, f"Failed for platform: {platform}"
