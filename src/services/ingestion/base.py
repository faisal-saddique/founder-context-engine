"""
Base parser interface that all parsers must implement.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseParser(ABC):
    """Abstract base class for all content parsers."""
    
    @abstractmethod
    async def parse(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """
        Parse content and return Markdown format.
        
        Args:
            content: The content to parse (file path, URL, etc.)
            metadata: Optional metadata to include
            
        Returns:
            Markdown-formatted string
        """
        pass
    
    @abstractmethod
    def validate_input(self, content: str) -> bool:
        """
        Validate that the input is acceptable for this parser.
        
        Args:
            content: The content to validate
            
        Returns:
            True if valid, False otherwise
        """
        pass
