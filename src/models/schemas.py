"""
Pydantic models for API request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime, UTC
from enum import Enum


class SourceType(str, Enum):
    """Types of content sources."""
    PDF_DECK = "pdf_deck"
    YOUTUBE_SUMMARY = "youtube_summary"
    APP_STORE_LINK = "app_store_link"
    WEB_URL = "web_url"
    RESEARCH = "research"
    MARKDOWN = "markdown"


class ContentSchema(str, Enum):
    """The 5 master content schemas for knowledge classification."""
    CASE_STUDY = "case_study"
    PROFILE = "profile"
    GUIDE = "guide"
    MARKET_INTEL = "market_intel"
    GENERAL = "general"


class UsagePermission(str, Enum):
    """Content usage permissions."""
    PUBLIC_SAFE = "public_safe"
    INTERNAL_ONLY = "internal_only"


class TrustScore(str, Enum):
    """Source trust levels."""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class ParsePreviewRequest(BaseModel):
    """Parse raw content into Markdown and classify it. Nothing is saved yet."""
    source_type: SourceType = Field(
        ..., description="How the content should be parsed"
    )
    content: str = Field(
        ...,
        description="File path (pdf_deck), URL (web_url, youtube_summary, app_store_link), search query (research), or raw text (markdown)",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Optional tags, author, or other context to attach"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "source_type": "markdown",
                    "content": "# Our App Growth Story\nWe grew from 0 to 150K downloads in 6 months using ASO.",
                    "metadata": {"author": "Team", "tags": ["case-study"]},
                },
                {
                    "source_type": "web_url",
                    "content": "https://example.com/blog/aso-guide",
                    "metadata": {"tags": ["guide", "aso"]},
                },
                {
                    "source_type": "youtube_summary",
                    "content": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    "metadata": {"author": "Speaker Name"},
                },
            ]
        }
    }


class ParsePreviewResponse(BaseModel):
    """Preview result with classified schema. Review and optionally edit before committing."""
    markdown_content: str = Field(
        ..., description="Parsed content in Markdown format"
    )
    preview_metadata: Dict[str, Any] = Field(
        ..., description="Parse details including detected schema and confidence"
    )
    source_type: SourceType
    content_schema: ContentSchema = Field(
        ContentSchema.GENERAL,
        description="Auto-detected schema (case_study, profile, guide, market_intel, general)",
    )
    confidence_score: float = Field(
        0.0, description="Classifier confidence (0-100). Falls back to 'general' below 60%"
    )


class CommitIngestRequest(BaseModel):
    """Save reviewed Markdown into the knowledge base (Neo4j + Supabase PG via LightRAG)."""
    markdown_content: str = Field(
        ..., description="Reviewed/edited Markdown from the preview step"
    )
    source_type: SourceType
    content_schema: ContentSchema = Field(
        ContentSchema.GENERAL,
        description="Content schema — use the auto-detected value or override manually",
    )
    metadata: Dict[str, Any] = Field(
        ..., description="Must include 'author'; can include 'file_name', 'source_url', 'tags', 'type'"
    )
    usage_permission: UsagePermission = Field(
        UsagePermission.PUBLIC_SAFE,
        description="public_safe = can be used in external posts, internal_only = sales prep / internal docs only",
    )
    trust_score: TrustScore = Field(
        TrustScore.MEDIUM,
        description="High = founder-verified, Medium = curated external, Low = experimental/unverified",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "markdown_content": "# Our App Growth Story\nWe grew from 0 to 150K downloads in 6 months.",
                    "source_type": "markdown",
                    "content_schema": "case_study",
                    "metadata": {"author": "Team", "file_name": "growth.md", "tags": ["case-study"]},
                    "usage_permission": "public_safe",
                    "trust_score": "High",
                }
            ]
        }
    }


class CommitIngestResponse(BaseModel):
    """Result of saving content to the knowledge base."""
    success: bool
    knowledge_id: str = Field(..., description="UUID of the stored knowledge entry")
    message: str


class PostFormat(str, Enum):
    """Content post formats."""
    COMPARISON = "comparison"
    DEEP_DIVE = "deep_dive"
    STORY = "story"
    GENERAL = "general"


class Platform(str, Enum):
    """Target platforms."""
    LINKEDIN = "linkedin"
    TWITTER = "twitter"
    UPWORK = "upwork"


class GenerateRequest(BaseModel):
    """Generate brand-safe content using the 5-node LangGraph workflow with claim validation and retry loop."""
    platform: Platform = Field(
        ..., description="Target platform — affects rules, tone, and formatting"
    )
    post_format: PostFormat = Field(
        PostFormat.GENERAL,
        description="Content structure: comparison, deep_dive, story, or general",
    )
    specific_resource_context: Optional[str] = Field(
        None,
        description="Topic or focus area for retrieval (e.g. 'ASO strategies for indie founders')",
    )
    tone: Optional[str] = Field(
        "professional", description="Writing tone (professional, casual, authoritative, etc.)"
    )
    custom_instructions: Optional[str] = Field(
        None, description="Extra instructions for the LLM (e.g. 'Keep it under 300 words')"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "platform": "linkedin",
                    "post_format": "deep_dive",
                    "specific_resource_context": "Share insights about ASO strategies for indie founders",
                    "tone": "professional",
                    "custom_instructions": "Keep it under 300 words",
                },
                {
                    "platform": "twitter",
                    "post_format": "story",
                    "specific_resource_context": "App growth case study",
                    "tone": "casual",
                },
            ]
        }
    }


class GenerateResponse(BaseModel):
    """Generated content with validation results, retry count, and critique score."""
    content: str = Field(..., description="The final generated content")
    generation_id: str = Field(..., description="Unique ID for this generation run")
    sources_used: List[str] = Field(
        ..., description="IDs of knowledge sources used during retrieval"
    )
    rules_applied: List[str] = Field(
        ..., description="IDs of platform rules applied during generation"
    )
    metadata: Dict[str, Any] = Field(
        ...,
        description="Includes validation_passed, validation_issues, critique, and retry_count",
    )


class ValidateRequest(BaseModel):
    """Check content for hallucinated claims against provided source material."""
    content: str = Field(
        ..., description="The text to validate for claims"
    )
    sources: List[Dict[str, Any]] = Field(
        ...,
        description="Source documents to validate against. Each should have 'content' and optionally 'trust_score'",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "content": "We have 500 clients and are the best agency in the market.",
                    "sources": [
                        {"content": "We have helped 50 founders launch their apps.", "trust_score": "High"}
                    ],
                }
            ]
        }
    }


class ValidationIssue(BaseModel):
    """A single detected claim issue."""
    claim: str = Field(..., description="The problematic claim text")
    issue_type: str = Field(
        ...,
        description="One of: unverified_number, unverified_superlative, unverified_entity, low_trust_source",
    )
    severity: str = Field(..., description="high, medium, or low")
    suggestion: Optional[str] = Field(
        None, description="Recommended fix (e.g. 'Remove specific number or use a range')"
    )


class ValidateResponse(BaseModel):
    """Validation result with list of detected issues."""
    is_valid: bool = Field(
        ..., description="True if no claim issues were found"
    )
    issues: List[ValidationIssue]
    validated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class HealthResponse(BaseModel):
    """Server health status with dependency availability."""
    status: str = Field(..., description="'healthy' when the server is running")
    environment: str = Field(..., description="'development' or 'production'")
    version: str = "1.0.0"
    services: Dict[str, bool] = Field(
        ..., description="True/false for each dependency (neo4j, postgresql, tavily, firecrawl, llamaparse)"
    )
