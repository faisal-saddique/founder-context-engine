"""
Document parser using LlamaParse for PDF/PPTX files.
Handles multimodal content including charts and slides.
"""
import os
from typing import Dict, Any
from llama_parse import LlamaParse

from .base import BaseParser
from ...core.config import settings
from ...core.logging import logger
from ...core.exceptions import ParsingError


class DocumentParser(BaseParser):
    """Parser for PDF and PPTX files using LlamaParse."""
    
    def __init__(self):
        """Initialize LlamaParse client."""
        self.client = LlamaParse(
            api_key=settings.llama_cloud_api_key,
            result_type="markdown",
            verbose=True
        )
    
    def validate_input(self, content: str) -> bool:
        """Check if file exists and has valid extension."""
        if not os.path.exists(content):
            return False
        
        valid_extensions = ['.pdf', '.pptx', '.docx', '.doc']
        return any(content.lower().endswith(ext) for ext in valid_extensions)
    
    async def parse(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """
        Parse document file to Markdown.
        
        Args:
            content: Path to the document file
            metadata: Optional metadata
            
        Returns:
            Markdown-formatted content
        """
        if not self.validate_input(content):
            raise ParsingError(f"Invalid document file: {content}")
        
        try:
            logger.info(f"Parsing document with LlamaParse: {content}")
            
            # Parse the document
            documents = self.client.load_data(content)
            
            # Combine all pages into single markdown
            markdown_content = "\n\n".join([doc.text for doc in documents])
            
            logger.info(f"Successfully parsed document: {content}")
            return markdown_content
            
        except Exception as e:
            logger.error(f"Failed to parse document {content}: {str(e)}")
            raise ParsingError(f"Document parsing failed: {str(e)}")
