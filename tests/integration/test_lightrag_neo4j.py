"""
Integration tests for LightRAG with Neo4j storage.
Tests the full pipeline: ingestion, chunking, and retrieval.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os
import pytest
import httpx
import asyncio

MOCK_DATA_URL = "https://raw.githubusercontent.com/gusye1234/nano-graphrag/main/tests/mock_data.txt"
API_BASE_URL = os.getenv("TEST_API_URL", "http://localhost:8000")


@pytest.fixture(scope="module")
def mock_data():
    """Fetch mock data (A Christmas Carol) for testing."""
    import requests
    response = requests.get(MOCK_DATA_URL)
    response.raise_for_status()
    return response.text


class TestLightRAGDirect:
    """Test LightRAG client directly without API."""

    @pytest.mark.asyncio
    async def test_lightrag_initialization(self):
        """Test LightRAG initializes with Neo4j."""
        from src.services.retrieval.lightrag_client import LightRAGClient

        client = LightRAGClient(working_dir="./test_rag_storage")
        await client.initialize()

        assert client.rag is not None

        await client.finalize()

    @pytest.mark.asyncio
    async def test_lightrag_insert_and_query(self, mock_data):
        """Test inserting and querying knowledge."""
        from src.services.retrieval.lightrag_client import LightRAGClient

        client = LightRAGClient(working_dir="./test_rag_storage")
        await client.initialize()

        # Insert a small portion of mock data
        test_content = mock_data[:5000]
        doc_id = await client.insert_knowledge(
            content=test_content,
            document_id="test-christmas-carol"
        )

        assert doc_id is not None

        # Query about Scrooge
        result = await client.query_knowledge(
            query="Who is Scrooge and what is his personality?",
            mode="hybrid"
        )

        assert result is not None
        assert len(result) > 0
        print(f"\n=== Query Result ===\n{result[:500]}...")

        await client.finalize()


class TestLightRAGAPI:
    """Test LightRAG through the FastAPI endpoints."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for API tests."""
        self.client = httpx.AsyncClient(base_url=API_BASE_URL, timeout=120.0)
        yield
        asyncio.get_event_loop().run_until_complete(self.client.aclose())

    @pytest.mark.asyncio
    async def test_api_health(self):
        """Test API is running and healthy."""
        async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=30.0) as client:
            response = await client.get("/health")
            assert response.status_code == 200
            data = response.json()
            print(f"\n=== Health Check ===\n{data}")
            assert data["status"] in ["healthy", "degraded"]

    @pytest.mark.asyncio
    async def test_ingest_and_query_via_api(self, mock_data):
        """Test the full ingestion and query flow through the API."""
        async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=120.0) as client:
            # Step 1: Commit the content to LightRAG
            test_content = mock_data[:8000]

            commit_payload = {
                "markdown_content": test_content,
                "source_type": "research",
                "metadata": {
                    "author": "Charles Dickens",
                    "type": "classic_literature",
                    "file_name": "christmas_carol.txt",
                    "tags": ["literature", "christmas", "victorian"]
                },
                "usage_permission": "public_safe",
                "trust_score": "High"
            }

            print("\n=== Ingesting Content ===")
            response = await client.post("/api/v1/ingest/commit_ingest", json=commit_payload)

            if response.status_code != 200:
                print(f"Ingestion failed: {response.text}")

            assert response.status_code == 200
            ingest_data = response.json()
            print(f"Ingest response: {ingest_data}")
            assert ingest_data["success"] is True

            # Give some time for indexing
            await asyncio.sleep(2)

            # Step 2: Query the knowledge
            print("\n=== Querying Knowledge ===")

            # Test different query modes
            queries = [
                ("Who is Ebenezer Scrooge?", "hybrid"),
                ("What ghosts visit Scrooge?", "local"),
                ("What is the main theme of the story?", "global"),
            ]

            for query, mode in queries:
                query_payload = {
                    "platform": "linkedin",
                    "post_format": "deep_dive",
                    "specific_resource_context": query,
                    "custom_instructions": f"Answer based on the ingested content. Use {mode} mode."
                }

                print(f"\nQuery: {query} (mode: {mode})")
                response = await client.post("/api/v1/generate", json=query_payload)

                if response.status_code == 200:
                    gen_data = response.json()
                    print(f"Response preview: {gen_data['content'][:300]}...")
                else:
                    print(f"Query failed: {response.status_code} - {response.text}")


def run_quick_test():
    """Quick standalone test for manual execution."""
    import asyncio
    import requests

    print("=" * 60)
    print("LightRAG Neo4j Integration Test")
    print("=" * 60)

    # Fetch mock data
    print("\n1. Fetching mock data...")
    response = requests.get(MOCK_DATA_URL)
    mock_data = response.text
    print(f"   Fetched {len(mock_data)} characters")

    async def run_test():
        from src.services.retrieval.lightrag_client import LightRAGClient

        print("\n2. Initializing LightRAG with Neo4j...")
        client = LightRAGClient(working_dir="./test_rag_storage")
        await client.initialize()
        print("   Initialized successfully")

        print("\n3. Inserting test content...")
        test_content = mock_data[:10000]
        doc_id = await client.insert_knowledge(
            content=test_content,
            document_id="test-christmas-carol-quick"
        )
        print(f"   Inserted document: {doc_id}")

        print("\n4. Testing queries...")
        test_queries = [
            "Who is Scrooge?",
            "What happens on Christmas Eve?",
            "Who is Bob Cratchit?",
        ]

        for query in test_queries:
            print(f"\n   Query: {query}")
            result = await client.query_knowledge(query=query, mode="hybrid")
            print(f"   Result preview: {result[:200]}...")

        print("\n5. Finalizing...")
        await client.finalize()
        print("   Done!")

        return True

    success = asyncio.run(run_test())
    return success


if __name__ == "__main__":
    run_quick_test()
