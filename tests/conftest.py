"""
Pytest configuration and shared fixtures.
Provides reusable test data drawn from the actual Dataset directory
so tests reflect real-world content.
"""
import os
import sys
import pytest
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

DATASET_DIR = Path(__file__).parent.parent / "Dataset"


# ── event loop ──────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ── simple sample fixtures ──────────────────────────────────────────
@pytest.fixture
def sample_markdown():
    """Sample markdown content for testing."""
    return """# Test Document

This is a test document about Company Inc.

## Key Points

- We have 100 customers
- We are the best in the industry
- Founded in 2020

## Details

Our product is used by Fortune 500 companies.
We have 95% customer satisfaction."""


@pytest.fixture
def sample_sources():
    """Sample sources for validation testing."""
    return [
        {
            "content": "Company Inc has 100 customers and was founded in 2020",
            "trust_score": "High",
            "type": "canonical_brand_facts",
        },
        {
            "content": "The product has 95% customer satisfaction rating",
            "trust_score": "Medium",
            "type": "survey_data",
        },
    ]


# ── real dataset fixtures ──────────────────────────────────────────
def _read_dataset_file(relative_path: str) -> str:
    """Read a file from the Dataset directory."""
    filepath = DATASET_DIR / relative_path
    if filepath.exists():
        return filepath.read_text(encoding="utf-8")
    return ""


@pytest.fixture
def canonical_brand_facts():
    """Canonical brand facts from the dataset."""
    return _read_dataset_file("Canonical Brand Facts/canonical_brand_facts.md")


@pytest.fixture
def icp_profiles():
    """ICP profiles from the dataset."""
    return _read_dataset_file("ICP/icp.md")


@pytest.fixture
def aso_playbook():
    """ASO playbook from the dataset."""
    return _read_dataset_file("playbooks/asoplaybook.md")


@pytest.fixture
def aso_trends():
    """ASO trends / market intel from the dataset."""
    return _read_dataset_file("industry_news/aso_trends_and_predictions.md")


@pytest.fixture
def aso_platform_changes():
    """ASO platform changes / market intel."""
    return _read_dataset_file("industry_news/aso_platform_changes.md")


@pytest.fixture
def linkedin_rules():
    """LinkedIn platform rules from the dataset."""
    return _read_dataset_file("PlatformRules/linkedin.md")


@pytest.fixture
def global_rules():
    """Global platform rules from the dataset."""
    return _read_dataset_file("PlatformRules/global_rules.md")


@pytest.fixture
def cleaner_case_study():
    """Cleaner app case study from the dataset."""
    return _read_dataset_file("Apps/Cleaner/app_case_insight_cleaner.md")


@pytest.fixture
def forms_case_study():
    """Forms app case study from the dataset."""
    return _read_dataset_file("Apps/Forms/app_case_insight_forms_app.md")


@pytest.fixture
def proposal_style_guidelines():
    """Proposal style guidelines from the dataset."""
    return _read_dataset_file("ProposalStyleGuidelines/proposal style guidelines.md")


# ── mock environment for settings ──────────────────────────────────
MOCK_ENV = {
    "TAVILY_API_KEY": "test-tavily-key",
    "LLAMA_CLOUD_API_KEY": "test-llama-key",
    "FIRECRAWL_API_KEY": "test-firecrawl-key",
    "DATABASE_URL": "postgresql://user:pass@localhost:6543/testdb?pgbouncer=true",
    "DIRECT_URL": "postgresql://user:pass@localhost:5432/testdb",
    "NEO4J_URI": "neo4j+s://test.databases.neo4j.io",
    "NEO4J_USERNAME": "neo4j",
    "NEO4J_PASSWORD": "testpass",
    "OPENAI_API_KEY": "test-openai-key",
    "ENVIRONMENT": "test",
}


@pytest.fixture
def mock_env():
    """Patch environment with test values."""
    with patch.dict(os.environ, MOCK_ENV, clear=False):
        yield MOCK_ENV


# ── mock LightRAG client ──────────────────────────────────────────
@pytest.fixture
def mock_lightrag():
    """A mocked LightRAG client that doesn't need real databases."""
    client = MagicMock()
    client.initialize = AsyncMock()
    client.insert_knowledge = AsyncMock(return_value="test-doc-id")
    client.query_knowledge = AsyncMock(return_value="Mocked knowledge retrieval")
    client.finalize = AsyncMock()
    client.rag = True  # truthy so "not initialized" checks pass
    return client


# ── mock classifier ───────────────────────────────────────────────
@pytest.fixture
def mock_classifier():
    """A mocked content classifier returning case_study @ 90%."""
    from src.models.schemas import ContentSchema

    classifier = MagicMock()
    classifier.classify = AsyncMock(
        return_value=(ContentSchema.CASE_STUDY, 90.0)
    )
    return classifier
