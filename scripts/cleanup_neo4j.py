"""
Cleanup script for Neo4j database.
Removes all nodes and relationships.
"""
import asyncio
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.neo4j_client import neo4j_client
from src.core.logging import logger


async def cleanup_neo4j():
    """Clear all data from Neo4j database."""
    try:
        print("Connecting to Neo4j...")
        await neo4j_client.connect()
        
        print("Clearing all nodes and relationships...")
        await neo4j_client.clear_database()
        
        print("✓ Neo4j database cleared successfully")
        
        await neo4j_client.close()
        
    except Exception as e:
        print(f"✗ Failed to cleanup Neo4j: {str(e)}")
        logger.error(f"Neo4j cleanup failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    confirm = input("Are you sure you want to clear the Neo4j database? (yes/no): ")
    if confirm.lower() == "yes":
        asyncio.run(cleanup_neo4j())
    else:
        print("Cleanup cancelled")
