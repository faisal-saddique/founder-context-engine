"""
Ingestion API endpoints for parsing and committing content.
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from ...models.schemas import (
    ParsePreviewRequest,
    ParsePreviewResponse,
    CommitIngestRequest,
    CommitIngestResponse,
    ContentSchema,
)
from ...models.knowledge import UnifiedKnowledge, KnowledgeMetadata
from ...services.ingestion.parser_factory import ParserFactory
from ...services.ingestion.classifier import content_classifier
from ...services.retrieval.lightrag_client import lightrag_client
from ...core.logging import logger
from ...core.exceptions import ParsingError, IngestionError

router = APIRouter(prefix="/api/v1/ingest", tags=["ingestion"])


@router.post("/parse_preview", response_model=ParsePreviewResponse)
async def parse_preview(request: ParsePreviewRequest):
    """
    **Step 1 of ingestion** — Parse raw content and classify it into one of 5 schemas.

    Returns Markdown preview for user review. **Nothing is saved to the knowledge base.**

    The classifier (GPT-4o-mini) auto-detects the content schema with a confidence score.
    If confidence drops below 60%, it falls back to `general`. You can override the schema
    when committing.

    Supported source types: `markdown`, `pdf_deck`, `web_url`, `app_store_link`,
    `youtube_summary`, `research`.
    """
    try:
        logger.info(f"Preview parse request for source type: {request.source_type}")
        
        # Create appropriate parser
        parser = ParserFactory.create_parser(request.source_type)
        
        # Parse the content
        markdown_content = await parser.parse(
            request.content,
            metadata=request.metadata
        )

        # Classify into one of the 5 master schemas
        detected_schema, confidence = await content_classifier.classify(markdown_content)

        # Prepare preview metadata
        preview_metadata = {
            "source_type": request.source_type.value,
            "original_content": request.content[:200] + "..." if len(request.content) > 200 else request.content,
            "parsed_length": len(markdown_content),
            "custom_metadata": request.metadata or {},
            "detected_schema": detected_schema.value,
            "confidence_score": confidence,
        }

        logger.info(f"Preview parse successful — classified as {detected_schema.value} ({confidence}%)")

        return ParsePreviewResponse(
            markdown_content=markdown_content,
            preview_metadata=preview_metadata,
            source_type=request.source_type,
            content_schema=detected_schema,
            confidence_score=confidence,
        )
        
    except ParsingError as e:
        logger.error(f"Parsing failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in parse_preview: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/commit_ingest", response_model=CommitIngestResponse)
async def commit_ingest(request: CommitIngestRequest):
    """
    **Step 2 of ingestion** — Save reviewed Markdown into the knowledge base.

    Call this after reviewing (and optionally editing) the preview from `parse_preview`.
    The content is wrapped in a Unified Knowledge JSON and inserted into LightRAG,
    which stores it in both Neo4j (entity graph) and Supabase PostgreSQL (vector embeddings).

    You can override the auto-detected `content_schema` before committing.
    Set `usage_permission` and `trust_score` based on how you want this content used in generation.
    """
    try:
        logger.info(f"Commit ingest request for source type: {request.source_type}")
        
        # Create Unified Knowledge JSON
        knowledge = UnifiedKnowledge(
            source_type=request.source_type.value,
            content_body=request.markdown_content,
            metadata=KnowledgeMetadata(
                author=request.metadata.get("author"),
                usage_permission=request.usage_permission.value,
                trust_score=request.trust_score.value,
                content_type=request.metadata.get("type", "general"),
                content_schema=request.content_schema.value,
                file_name=request.metadata.get("file_name"),
                source_url=request.metadata.get("source_url"),
                tags=request.metadata.get("tags", [])
            )
        )
        
        # Convert to LightRAG format
        lightrag_content = knowledge.to_lightrag_format()
        
        # Insert into LightRAG
        document_id = await lightrag_client.insert_knowledge(
            content=lightrag_content,
            document_id=knowledge.id
        )
        
        logger.info(f"Successfully committed knowledge: {knowledge.id}")
        
        return CommitIngestResponse(
            success=True,
            knowledge_id=knowledge.id,
            message=f"Knowledge successfully ingested with ID: {knowledge.id}"
        )
        
    except IngestionError as e:
        logger.error(f"Ingestion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in commit_ingest: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
