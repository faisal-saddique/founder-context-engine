"""
Web parser using Firecrawl for clean Markdown extraction.
"""
from typing import Dict, Any
from firecrawl import FirecrawlApp

from .base import BaseParser
from ...core.config import settings
from ...core.logging import logger
from ...core.exceptions import ParsingError


class WebParser(BaseParser):
    """Parser for web URLs using Firecrawl."""
    
    def __init__(self):
        """Initialize Firecrawl client."""
        self.client = FirecrawlApp(api_key=settings.firecrawl_api_key)
    
    def validate_input(self, content: str) -> bool:
        """Check if input is a valid URL."""
        return content.startswith(('http://', 'https://'))
    
    async def parse(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """
        Scrape URL and return clean Markdown.
        
        Args:
            content: URL to scrape
            metadata: Optional metadata
            
        Returns:
            Markdown-formatted content
        """
        if not self.validate_input(content):
            raise ParsingError(f"Invalid URL: {content}")
        
        try:
            logger.info(f"Scraping URL with Firecrawl: {content}")
            
            # Scrape the URL
            result = self.client.scrape(
                url=content,
                formats=['markdown'],
            )

            # Extract markdown content
            markdown_content = result.markdown or ''
            
            if not markdown_content:
                raise ParsingError(f"No content extracted from URL: {content}")
            
            logger.info(f"Successfully scraped URL: {content}")
            return markdown_content
            
        except Exception as e:
            logger.error(f"Failed to scrape URL {content}: {str(e)}")
            raise ParsingError(f"Web scraping failed: {str(e)}")
