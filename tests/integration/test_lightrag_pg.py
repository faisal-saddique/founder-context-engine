"""
Integration tests for LightRAG with PostgreSQL storage.
Tests init, insert, and query with PG KV/Vector/DocStatus + Neo4j graph.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import asyncio

MOCK_DATA_URL = "https://raw.githubusercontent.com/gusye1234/nano-graphrag/main/tests/mock_data.txt"


@pytest.fixture(scope="module")
def mock_data():
    """Fetch mock data for testing."""
    import requests
    response = requests.get(MOCK_DATA_URL)
    response.raise_for_status()
    return response.text


class TestLightRAGWithPGStorage:
    """Test LightRAG client with Supabase PG storage backends."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """LightRAG should init with PG KV/Vector/DocStatus + Neo4j graph."""
        from src.services.retrieval.lightrag_client import LightRAGClient

        client = LightRAGClient(working_dir="./test_rag_storage")
        await client.initialize()

        assert client.rag is not None

        # verify graph is Neo4j, other storages are PG-backed
        rag = client.rag
        assert type(rag.chunk_entity_relation_graph).__name__ == "Neo4JStorage"
        # kv_storage_cls is a partial wrapping PGKVStorage
        assert "PGKVStorage" in str(rag.key_string_value_json_storage_cls)

        await client.finalize()

    @pytest.mark.asyncio
    async def test_insert_and_query(self, mock_data):
        """Full insert + query cycle through PG storage."""
        from src.services.retrieval.lightrag_client import LightRAGClient

        client = LightRAGClient(working_dir="./test_rag_storage")
        await client.initialize()

        test_content = mock_data[:5000]
        doc_id = await client.insert_knowledge(
            content=test_content,
            document_id="test-pg-christmas-carol"
        )
        assert doc_id is not None

        result = await client.query_knowledge(
            query="Who is Scrooge and what is his personality?",
            mode="hybrid"
        )
        assert result is not None
        assert len(result) > 0
        print(f"\n=== PG Storage Query Result ===\n{result[:500]}...")

        await client.finalize()


def run_quick_test():
    """Standalone test runner for manual execution."""
    import requests

    print("=" * 60)
    print("LightRAG PG Storage Integration Test")
    print("=" * 60)

    print("\n1. Fetching mock data...")
    response = requests.get(MOCK_DATA_URL)
    mock_data = response.text
    print(f"   Fetched {len(mock_data)} characters")

    async def run():
        from src.services.retrieval.lightrag_client import LightRAGClient

        print("\n2. Initializing LightRAG with PG storage...")
        client = LightRAGClient(working_dir="./test_rag_storage")
        await client.initialize()
        print("   Initialized successfully")

        print("\n3. Inserting test content...")
        test_content = mock_data[:10000]
        doc_id = await client.insert_knowledge(
            content=test_content,
            document_id="test-pg-quick"
        )
        print(f"   Inserted document: {doc_id}")

        print("\n4. Testing queries...")
        for query in ["Who is Scrooge?", "What happens on Christmas Eve?"]:
            print(f"\n   Query: {query}")
            result = await client.query_knowledge(query=query, mode="hybrid")
            print(f"   Result preview: {result[:200]}...")

        print("\n5. Finalizing...")
        await client.finalize()
        print("   Done!")

    asyncio.run(run())


if __name__ == "__main__":
    run_quick_test()
