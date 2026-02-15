"""
Cleanup script for PostgreSQL database.
Removes all data from Prisma ORM tables and LightRAG PG tables.
"""
import asyncio
import ssl
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from prisma import Prisma
from src.core.config import settings
from src.core.logging import logger


# known LightRAG fixed-name tables (lowercase in Supabase)
LIGHTRAG_TABLES = [
    "lightrag_doc_full",
    "lightrag_doc_chunks",
    "lightrag_doc_status",
    "lightrag_llm_cache",
    "lightrag_full_entities",
    "lightrag_full_relations",
    "lightrag_entity_chunks",
    "lightrag_relation_chunks",
]

# vector tables use dynamic names based on embedding model
LIGHTRAG_VECTOR_PATTERNS = [
    "lightrag_vdb_entity_%",
    "lightrag_vdb_relation_%",
    "lightrag_vdb_chunks_%",
]


async def cleanup_prisma_tables():
    """Clear Prisma ORM tables."""
    print("Connecting to PostgreSQL via Prisma...")
    db = Prisma()
    await db.connect()

    print("Clearing feedback_logs...")
    await db.feedbacklog.delete_many()

    print("Clearing generations...")
    await db.generation.delete_many()

    print("Clearing agent_rules...")
    await db.agentrule.delete_many()

    print("Clearing platform_configs...")
    await db.platformconfig.delete_many()

    await db.disconnect()
    print("  Prisma tables cleared")


async def cleanup_lightrag_tables():
    """Clear all LightRAG PG tables via asyncpg."""
    import asyncpg

    pg = settings.lightrag_pg_config
    # Supabase needs SSL but the intermediate cert is broken, so skip verification
    ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    print("Connecting to PostgreSQL via asyncpg...")
    conn = await asyncpg.connect(
        host=pg["POSTGRES_HOST"],
        port=int(pg["POSTGRES_PORT"]),
        user=pg["POSTGRES_USER"],
        password=pg["POSTGRES_PASSWORD"],
        database=pg["POSTGRES_DATABASE"],
        ssl=ssl_ctx,
        statement_cache_size=0,
    )

    try:
        # truncate fixed-name tables
        for table in LIGHTRAG_TABLES:
            exists = await conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = $1)",
                table,
            )
            if exists:
                await conn.execute(f'TRUNCATE TABLE "{table}" CASCADE')
                print(f"  Truncated {table}")
            else:
                print(f"  Skipped {table} (not found)")

        # find and truncate vector tables by pattern
        for pattern in LIGHTRAG_VECTOR_PATTERNS:
            rows = await conn.fetch(
                "SELECT table_name FROM information_schema.tables WHERE table_name LIKE $1",
                pattern,
            )
            for row in rows:
                name = row["table_name"]
                await conn.execute(f'TRUNCATE TABLE "{name}" CASCADE')
                print(f"  Truncated {name}")

        print("  LightRAG tables cleared")
    finally:
        await conn.close()


async def cleanup_postgres():
    """Clear all data from PostgreSQL tables."""
    try:
        await cleanup_prisma_tables()
        await cleanup_lightrag_tables()
        print("\nPostgreSQL database cleared successfully")
    except Exception as e:
        print(f"Failed to cleanup PostgreSQL: {str(e)}")
        logger.error(f"PostgreSQL cleanup failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    confirm = input("Are you sure you want to clear the PostgreSQL database? (yes/no): ")
    if confirm.lower() == "yes":
        asyncio.run(cleanup_postgres())
    else:
        print("Cleanup cancelled")
