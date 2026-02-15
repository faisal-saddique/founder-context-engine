"""
Content generation API endpoints.
"""
from fastapi import APIRouter, HTTPException
from prisma import Prisma

from ...models.schemas import (
    GenerateRequest,
    GenerateResponse,
    ValidateRequest,
    ValidateResponse
)
from ...services.graph.workflow import content_workflow
from ...services.validation.claim_validator import claim_validator
from ...core.logging import logger

router = APIRouter(prefix="/api/v1/generate", tags=["generation"])


@router.post("/", response_model=GenerateResponse)
async def generate_content(request: GenerateRequest):
    """
    Generate brand-safe content using the 5-node LangGraph workflow.

    **Workflow:** Input Analysis -> Dual Retrieval (LightRAG + Prisma rules) ->
    Draft Generation (GPT-4) -> Claim Validation -> Critique & Score.

    If validation catches hallucinated claims (unverified numbers, superlatives, or entities),
    the workflow retries draft generation with issue-aware correction prompts — up to 2 retries.
    If all retries fail, the attempt with the fewest issues is used.

    The response `metadata` includes `validation_passed`, `validation_issues`,
    `retry_count`, and `critique` score.
    """
    try:
        logger.info(f"Generation request for platform: {request.platform.value}")
        
        # Run the workflow
        final_state = await content_workflow.generate(
            platform=request.platform.value,
            post_format=request.post_format.value,
            tone=request.tone,
            specific_resource_context=request.specific_resource_context,
            custom_instructions=request.custom_instructions
        )
        
        # Log generation to database
        try:
            db = Prisma()
            await db.connect()
            
            generation_record = await db.generation.create(
                data={
                    "generationReason": f"{request.post_format.value} post for {request.platform.value}",
                    "promptVersion": "1.0",
                    "finalOutput": final_state["final_content"],
                    "sourceIdsUsed": final_state["source_ids"],
                    "ruleIdsApplied": final_state["rule_ids_applied"],
                    "modelConfig": final_state["model_config"]
                }
            )
            
            await db.disconnect()
            
        except Exception as e:
            logger.warning(f"Failed to log generation to database: {str(e)}")
        
        # Return response
        return GenerateResponse(
            content=final_state["final_content"],
            generation_id=final_state["generation_id"],
            sources_used=final_state["source_ids"],
            rules_applied=final_state["rule_ids_applied"],
            metadata={
                "validation_passed": final_state["validation_passed"],
                "validation_issues": final_state["validation_issues"],
                "critique": final_state["critique_result"],
                "retry_count": final_state.get("retry_count", 0),
            }
        )
        
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate", response_model=ValidateResponse)
async def validate_content(request: ValidateRequest):
    """
    Standalone claim validation — check any content against source material.

    Detects three types of claims: **numbers/percentages** (e.g. "500 clients"),
    **superlatives** (e.g. "the best", "fastest"), and **named entities** (e.g. "Acme Corp").

    Each claim is checked against the provided sources via substring matching.
    Returns `is_valid: false` with specific issues and suggested fixes if problems are found.
    """
    try:
        logger.info("Content validation request")
        
        # Validate the content
        is_valid, issues = claim_validator.validate_content(
            content=request.content,
            sources=request.sources
        )
        
        return ValidateResponse(
            is_valid=is_valid,
            issues=issues
        )
        
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
