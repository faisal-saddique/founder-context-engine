"""
LightRAG client for semantic memory operations.
Wraps LightRAG with our custom configuration.
"""
import os
import ssl
from typing import Optional

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.postgres_impl import PostgreSQLDB

from ...core.config import settings
from ...core.logging import logger
from ...core.exceptions import IngestionError, RetrievalError


def _supabase_ssl_context(_self):
    """Supabase uses a private CA with a broken intermediate cert (missing keyUsage).
    Standard verification fails, so we use encrypted-only TLS here."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


# Patch before any PG connection is made
PostgreSQLDB._create_ssl_context = _supabase_ssl_context


class LightRAGClient:
    """Client for LightRAG semantic memory."""

    def __init__(self, working_dir: str = "./rag_storage"):
        """
        Initialize LightRAG client.

        Args:
            working_dir: Directory for LightRAG storage
        """
        self.working_dir = working_dir
        self.rag = None

        # Ensure working directory exists
        os.makedirs(working_dir, exist_ok=True)

    async def initialize(self):
        """Initialize LightRAG with Neo4j graph + Supabase PG storage."""
        try:
            # Neo4j env vars for graph storage
            os.environ["NEO4J_URI"] = settings.neo4j_uri
            os.environ["NEO4J_USERNAME"] = settings.neo4j_username
            os.environ["NEO4J_PASSWORD"] = settings.neo4j_password

            # Postgres env vars for KV/Vector/DocStatus (parsed from DIRECT_URL)
            for key, value in settings.lightrag_pg_config.items():
                os.environ[key] = value
            os.environ["POSTGRES_SSL_MODE"] = "require"
            os.environ["POSTGRES_STATEMENT_CACHE_SIZE"] = "0"
            os.environ["POSTGRES_MAX_CONNECTIONS"] = "10"

            self.rag = LightRAG(
                working_dir=self.working_dir,
                llm_model_func=gpt_4o_mini_complete,
                llm_model_name="gpt-4o-mini",
                embedding_func=openai_embed,
                graph_storage="Neo4JStorage",
                kv_storage="PGKVStorage",
                vector_storage="PGVectorStorage",
                doc_status_storage="PGDocStatusStorage",
                # concurrency tuning for OpenAI rate limits
                llm_model_max_async=8,
                max_parallel_insert=4,
                embedding_func_max_async=8,
                # domain-specific entity types for founder content
                addon_params={
                    "language": "English",
                    "entity_types": [
                        "Person",
                        "Organization",
                        "Product",
                        "Location",
                        "Event",
                        "Concept",
                        "Data",
                        "Content",
                    ],
                },
            )

            await self.rag.initialize_storages()
            logger.info("LightRAG initialized with PG storage + Neo4j graph")

        except Exception as e:
            logger.error(f"Failed to initialize LightRAG: {str(e)}")
            raise IngestionError(f"LightRAG initialization failed: {str(e)}")

    async def insert_knowledge(
        self,
        content: str,
        document_id: Optional[str] = None
    ) -> str:
        """
        Insert knowledge into LightRAG.

        Args:
            content: Markdown-formatted content to insert
            document_id: Optional document ID

        Returns:
            Document ID
        """
        if not self.rag:
            raise IngestionError("LightRAG not initialized. Call initialize() first.")

        try:
            logger.info("Inserting knowledge into LightRAG")

            # Insert the content
            await self.rag.ainsert(
                content,
                ids=[document_id] if document_id else None
            )

            logger.info(f"Successfully inserted knowledge: {document_id or 'auto-generated ID'}")
            return document_id or "auto-generated"

        except Exception as e:
            logger.error(f"Failed to insert knowledge: {str(e)}")
            raise IngestionError(f"Knowledge insertion failed: {str(e)}")

    async def query_knowledge(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 60,
        chunk_top_k: int = 20
    ) -> str:
        """
        Query knowledge from LightRAG.

        Args:
            query: Query string
            mode: Query mode (local, global, hybrid, mix, naive)
            top_k: Number of KG entities/relations to retrieve
            chunk_top_k: Number of text chunks to retrieve

        Returns:
            Query result
        """
        if not self.rag:
            raise RetrievalError("LightRAG not initialized. Call initialize() first.")

        try:
            logger.info(f"Querying LightRAG with mode: {mode}")

            result = await self.rag.aquery(
                query,
                param=QueryParam(
                    mode=mode,
                    top_k=top_k,
                    chunk_top_k=chunk_top_k
                )
            )

            logger.info("Successfully retrieved knowledge from LightRAG")
            return result

        except Exception as e:
            logger.error(f"Failed to query knowledge: {str(e)}")
            raise RetrievalError(f"Knowledge retrieval failed: {str(e)}")

    async def finalize(self):
        """Clean up LightRAG resources."""
        if self.rag:
            await self.rag.finalize_storages()
            logger.info("LightRAG finalized")


# Global LightRAG client instance
lightrag_client = LightRAGClient()
