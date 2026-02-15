"""
Research parser using Tavily for deep web research.
"""
from typing import Dict, Any
from tavily import TavilyClient

from .base import BaseParser
from ...core.config import settings
from ...core.logging import logger
from ...core.exceptions import ParsingError


class ResearchParser(BaseParser):
    """Parser for research queries using Tavily."""
    
    def __init__(self):
        """Initialize Tavily client."""
        self.client = TavilyClient(api_key=settings.tavily_api_key)
    
    def validate_input(self, content: str) -> bool:
        """Research queries are always valid if non-empty."""
        return bool(content.strip())
    
    async def parse(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """
        Perform research and return formatted results.
        
        Args:
            content: Research query string
            metadata: Optional metadata (can include search_depth)
            
        Returns:
            Markdown-formatted research results
        """
        if not self.validate_input(content):
            raise ParsingError("Research query cannot be empty")
        
        try:
            logger.info(f"Performing research with Tavily: {content}")
            
            # Determine search depth
            search_depth = "advanced" if metadata and metadata.get("deep_research") else "basic"
            
            # Perform search
            response = self.client.search(
                query=content,
                search_depth=search_depth,
                max_results=5
            )
            
            # Format as markdown
            markdown_content = f"# Research Query: {content}\n\n"
            markdown_content += f"**Search Depth:** {search_depth}\n\n"
            markdown_content += "---\n\n"
            
            # Add results
            if 'results' in response:
                for idx, result in enumerate(response['results'], 1):
                    markdown_content += f"## Result {idx}: {result.get('title', 'Untitled')}\n\n"
                    markdown_content += f"**Source:** {result.get('url', 'Unknown')}\n\n"
                    markdown_content += f"{result.get('content', '')}\n\n"
                    markdown_content += "---\n\n"
            
            # Add summary if available
            if 'answer' in response:
                markdown_content += f"## Summary\n\n{response['answer']}\n\n"
            
            logger.info(f"Successfully completed research for: {content}")
            return markdown_content
            
        except Exception as e:
            logger.error(f"Failed to perform research for {content}: {str(e)}")
            raise ParsingError(f"Research failed: {str(e)}")
