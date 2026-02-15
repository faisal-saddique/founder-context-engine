"""
Factory for creating the appropriate parser based on source type.
"""
from typing import Dict, Any

from .base import BaseParser
from .document_parser import DocumentParser
from .web_parser import WebParser
from .video_parser import VideoParser
from .research_parser import ResearchParser
from .markdown_parser import MarkdownParser
from ...models.schemas import SourceType
from ...core.exceptions import ParsingError


class ParserFactory:
    """Factory class to create appropriate parser instances."""
    
    @staticmethod
    def create_parser(source_type: SourceType) -> BaseParser:
        """
        Create and return the appropriate parser for the given source type.
        
        Args:
            source_type: The type of content source
            
        Returns:
            Parser instance
            
        Raises:
            ParsingError: If source type is not supported
        """
        parsers = {
            SourceType.PDF_DECK: DocumentParser,
            SourceType.WEB_URL: WebParser,
            SourceType.APP_STORE_LINK: WebParser,
            SourceType.YOUTUBE_SUMMARY: VideoParser,
            SourceType.RESEARCH: ResearchParser,
            SourceType.MARKDOWN: MarkdownParser,
        }
        
        parser_class = parsers.get(source_type)
        if not parser_class:
            raise ParsingError(f"Unsupported source type: {source_type}")
        
        return parser_class()
