"""
Unified Knowledge JSON schema for LightRAG.
All ingested data must conform to this structure.
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime, UTC
import uuid


class KnowledgeMetadata(BaseModel):
    """Metadata for knowledge entries."""
    author: Optional[str] = None
    usage_permission: str = "public_safe"
    trust_score: str = "Medium"
    content_type: str = "general"
    content_schema: str = "general"
    file_name: Optional[str] = None
    source_url: Optional[str] = None
    created_by: Optional[str] = None
    tags: Optional[list[str]] = []


class UnifiedKnowledge(BaseModel):
    """
    Unified knowledge structure for LightRAG insertion.
    This is the contract between our ingestion pipeline and LightRAG.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_type: str
    content_body: str = Field(..., description="Markdown-formatted content")
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: KnowledgeMetadata
    
    def to_lightrag_format(self) -> str:
        """
        Convert to format expected by LightRAG.
        LightRAG expects plain text, so we include metadata as headers.
        """
        header = f"# {self.source_type}\n"
        header += f"**ID:** {self.id}\n"
        header += f"**Schema:** {self.metadata.content_schema}\n"
        header += f"**Created:** {self.created_at}\n"
        header += f"**Author:** {self.metadata.author or 'Unknown'}\n"
        header += f"**Permission:** {self.metadata.usage_permission}\n"
        header += f"**Trust:** {self.metadata.trust_score}\n"
        
        if self.metadata.source_url:
            header += f"**Source:** {self.metadata.source_url}\n"
        
        header += "\n---\n\n"
        
        return header + self.content_body
