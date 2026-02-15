"""
Database setup script.
Initializes databases and creates initial data.
"""
import asyncio
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from prisma import Prisma
from src.db.neo4j_client import neo4j_client
from src.core.logging import logger


async def setup_databases():
    """Initialize and verify database connections."""
    try:
        # Test Neo4j connection
        print("Testing Neo4j connection...")
        await neo4j_client.connect()
        if await neo4j_client.verify_connectivity():
            print("✓ Neo4j connection successful")
        else:
            print("✗ Neo4j connection failed")
            return False
        await neo4j_client.close()
        
        # Test PostgreSQL connection
        print("\nTesting PostgreSQL connection...")
        db = Prisma()
        await db.connect()
        print("✓ PostgreSQL connection successful")
        
        # Create initial platform configs
        print("\nCreating initial platform configurations...")
        
        platforms = ["linkedin", "twitter", "upwork"]
        for platform in platforms:
            existing = await db.platformconfig.find_unique(
                where={"platformName": platform}
            )
            
            if not existing:
                await db.platformconfig.create(
                    data={
                        "platformName": platform,
                        "customInstructions": f"Default instructions for {platform}"
                    }
                )
                print(f"✓ Created config for {platform}")
            else:
                print(f"- Config for {platform} already exists")
        
        # Create some default global rules
        print("\nCreating default global rules...")
        
        default_rules = [
            {
                "platform": "global",
                "scope": "GLOBAL",
                "constraintText": "Never fabricate metrics, client names, or rankings",
                "ruleType": "negative",
                "priority": 100,
                "status": "ACTIVE"
            },
            {
                "platform": "global",
                "scope": "GLOBAL",
                "constraintText": "Always maintain professional and respectful tone",
                "ruleType": "positive",
                "priority": 90,
                "status": "ACTIVE"
            },
            {
                "platform": "global",
                "scope": "GLOBAL",
                "constraintText": "Ensure all claims are backed by provided sources",
                "ruleType": "positive",
                "priority": 95,
                "status": "ACTIVE"
            }
        ]
        
        for rule_data in default_rules:
            await db.agentrule.create(data=rule_data)
            print(f"✓ Created rule: {rule_data['constraintText'][:50]}...")
        
        print("\n✓ Database setup complete!")
        
        await db.disconnect()
        return True
        
    except Exception as e:
        print(f"\n✗ Database setup failed: {str(e)}")
        logger.error(f"Database setup failed: {str(e)}")
        return False


if __name__ == "__main__":
    success = asyncio.run(setup_databases())
    sys.exit(0 if success else 1)
