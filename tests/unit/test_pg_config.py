"""
Unit tests for Postgres config parsing from DIRECT_URL.
Verifies that Supabase connection URLs are correctly decomposed
into individual POSTGRES_* params for LightRAG.
"""
import os
import pytest
from unittest.mock import patch


# minimal env vars needed to instantiate Settings
REQUIRED_ENV = {
    "TAVILY_API_KEY": "test",
    "LLAMA_CLOUD_API_KEY": "test",
    "FIRECRAWL_API_KEY": "test",
    "DATABASE_URL": "postgresql://user:pass@host:6543/db",
    "NEO4J_URI": "neo4j+s://test.databases.neo4j.io",
    "NEO4J_USERNAME": "neo4j",
    "NEO4J_PASSWORD": "testpass",
}


def _make_settings(direct_url: str):
    """Helper to create a Settings instance with a given DIRECT_URL."""
    env = {**REQUIRED_ENV, "DIRECT_URL": direct_url}
    with patch.dict(os.environ, env, clear=False):
        # reimport to get a fresh instance with patched env
        from src.core.config import Settings
        return Settings()


class TestLightragPgConfig:
    """Test URL parsing for LightRAG PG configuration."""

    def test_supabase_url_with_encoded_password(self):
        """Parse a Supabase URL with percent-encoded special chars in password."""
        url = "postgresql://postgres.projref:8%3F%24%254%3Fd%23z%2FV4cGF@aws-1-ap-southeast-2.pooler.supabase.com:5432/postgres"
        s = _make_settings(url)
        config = s.lightrag_pg_config

        assert config["POSTGRES_HOST"] == "aws-1-ap-southeast-2.pooler.supabase.com"
        assert config["POSTGRES_PORT"] == "5432"
        assert config["POSTGRES_USER"] == "postgres.projref"
        assert config["POSTGRES_PASSWORD"] == "8?$%4?d#z/V4cGF"
        assert config["POSTGRES_DATABASE"] == "postgres"

    def test_simple_localhost_url(self):
        """Parse a basic localhost PostgreSQL URL."""
        url = "postgresql://admin:secret@localhost:5432/mydb"
        s = _make_settings(url)
        config = s.lightrag_pg_config

        assert config["POSTGRES_HOST"] == "localhost"
        assert config["POSTGRES_PORT"] == "5432"
        assert config["POSTGRES_USER"] == "admin"
        assert config["POSTGRES_PASSWORD"] == "secret"
        assert config["POSTGRES_DATABASE"] == "mydb"

    def test_custom_port(self):
        """Parse a URL with non-default port."""
        url = "postgresql://user:pass@db.example.com:15432/production"
        s = _make_settings(url)
        config = s.lightrag_pg_config

        assert config["POSTGRES_PORT"] == "15432"
        assert config["POSTGRES_DATABASE"] == "production"

    def test_password_with_at_sign(self):
        """Handle passwords containing @ (encoded as %40)."""
        url = "postgresql://user:p%40ssword@host:5432/db"
        s = _make_settings(url)
        config = s.lightrag_pg_config

        assert config["POSTGRES_PASSWORD"] == "p@ssword"

    def test_username_with_dot(self):
        """Supabase usernames contain dots (postgres.project_ref)."""
        url = "postgresql://postgres.xyzproject:pass@host:5432/postgres"
        s = _make_settings(url)
        config = s.lightrag_pg_config

        assert config["POSTGRES_USER"] == "postgres.xyzproject"

    def test_returns_all_required_keys(self):
        """Config dict should contain exactly the 5 expected keys."""
        url = "postgresql://user:pass@host:5432/db"
        s = _make_settings(url)
        config = s.lightrag_pg_config

        expected_keys = {
            "POSTGRES_HOST",
            "POSTGRES_PORT",
            "POSTGRES_USER",
            "POSTGRES_PASSWORD",
            "POSTGRES_DATABASE",
        }
        assert set(config.keys()) == expected_keys
