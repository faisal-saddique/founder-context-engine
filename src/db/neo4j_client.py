"""
Neo4j client for graph database operations.
"""
from neo4j import GraphDatabase, AsyncGraphDatabase
from typing import Optional

from ..core.config import settings
from ..core.logging import logger


class Neo4jClient:
    """Async Neo4j client wrapper."""
    
    def __init__(self):
        """Initialize Neo4j connection."""
        self.driver = None
        self.uri = settings.neo4j_uri
        self.username = settings.neo4j_username
        self.password = settings.neo4j_password
    
    async def connect(self):
        """Establish connection to Neo4j."""
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            logger.info("Connected to Neo4j successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise
    
    async def close(self):
        """Close Neo4j connection."""
        if self.driver:
            await self.driver.close()
            logger.info("Closed Neo4j connection")
    
    async def verify_connectivity(self) -> bool:
        """Verify Neo4j connection is working."""
        try:
            async with self.driver.session() as session:
                result = await session.run("RETURN 1 as num")
                record = await result.single()
                return record["num"] == 1
        except Exception as e:
            logger.error(f"Neo4j connectivity check failed: {str(e)}")
            return False
    
    async def clear_database(self):
        """Clear all nodes and relationships from the database."""
        try:
            async with self.driver.session() as session:
                await session.run("MATCH (n) DETACH DELETE n")
            logger.info("Cleared Neo4j database")
        except Exception as e:
            logger.error(f"Failed to clear Neo4j database: {str(e)}")
            raise


# Global Neo4j client instance
neo4j_client = Neo4jClient()
