"""
FastAPI application entry point.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .routes import health, ingestion, generation
from ..core.config import settings
from ..core.logging import logger
from ..db.neo4j_client import neo4j_client
from ..services.retrieval.lightrag_client import lightrag_client
from ..services.retrieval.rule_retriever import rule_retriever


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown events for the application.
    """
    # Startup
    logger.info("Starting Founder Context Engine API...")
    
    try:
        # Initialize Neo4j
        await neo4j_client.connect()
        
        # Initialize LightRAG
        await lightrag_client.initialize()
        
        # Connect rule retriever
        await rule_retriever.connect()
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Founder Context Engine API...")
    
    try:
        await neo4j_client.close()
        await lightrag_client.finalize()
        await rule_retriever.disconnect()
        
        logger.info("All services shut down successfully")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")


# Tag descriptions shown in the /docs sidebar
tags_metadata = [
    {
        "name": "health",
        "description": "Server health and dependency status checks.",
    },
    {
        "name": "ingestion",
        "description": (
            "Two-step content ingestion pipeline. "
            "**Step 1** — `parse_preview` parses raw content (PDF, URL, YouTube, etc.) "
            "into Markdown and classifies it into one of 5 schemas. Nothing is saved. "
            "**Step 2** — `commit_ingest` saves the reviewed Markdown into the knowledge base "
            "(Neo4j graph + Supabase PG vectors via LightRAG)."
        ),
    },
    {
        "name": "generation",
        "description": (
            "Content generation and claim validation. "
            "The `/generate` endpoint runs a 5-node LangGraph workflow: "
            "Input Analysis -> Dual Retrieval -> Draft Generation -> Claim Validation -> Critique. "
            "If validation catches hallucinated claims, the workflow retries up to 2 times "
            "with issue-aware correction prompts. "
            "The `/validate` endpoint provides standalone claim checking."
        ),
    },
]

# Create FastAPI app
app = FastAPI(
    title="Founder Context Engine API",
    description=(
        "A brand intelligence engine that ingests curated business knowledge and generates "
        "high-authority, brand-safe marketing content across LinkedIn, Twitter, and Upwork.\n\n"
        "**Key features:**\n"
        "- Preview-then-commit ingestion with auto-classification into 5 content schemas\n"
        "- LangGraph generation workflow with claim validation and retry loop\n"
        "- Dual-memory retrieval (Neo4j knowledge graph + Supabase PG vectors)\n"
        "- Platform-specific rules from PostgreSQL\n\n"
        "See the [GitHub repo](https://github.com/faisal-saddique/founder-context-engine) for full docs."
    ),
    version="1.0.0",
    lifespan=lifespan,
    openapi_tags=tags_metadata,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(ingestion.router)
app.include_router(generation.router)


@app.get("/", tags=["health"])
async def root():
    """API info and quick links to docs and health check."""
    return {
        "name": "Founder Context Engine API",
        "version": "1.0.0",
        "environment": settings.environment,
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.environment == "development"
    )
