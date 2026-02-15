"""
Health check endpoints.
"""
import asyncio
from fastapi import APIRouter
from ...models.schemas import HealthResponse
from ...core.config import settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check server health and dependency availability.

    Returns `true`/`false` for each external service based on whether
    the required API key or connection string is configured.
    Services checked: Neo4j, PostgreSQL, Tavily, Firecrawl, LlamaParse.
    """

    # Quick health check - don't make actual connections (those happen at startup)
    # Just verify config is present
    return HealthResponse(
        status="healthy",
        environment=settings.environment,
        services={
            "neo4j": bool(settings.neo4j_uri),
            "postgresql": bool(settings.database_url),
            "tavily": bool(settings.tavily_api_key),
            "firecrawl": bool(settings.firecrawl_api_key),
            "llamaparse": bool(settings.llama_cloud_api_key)
        }
    )
