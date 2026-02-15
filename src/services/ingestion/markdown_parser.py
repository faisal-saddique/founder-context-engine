"""
Passthrough parser for content that is already in Markdown format.
Used for direct markdown files and pre-formatted text.
"""
from typing import Dict, Any

from .base import BaseParser
from ...core.exceptions import ParsingError


class MarkdownParser(BaseParser):
    """Parser for raw Markdown content — returns it as-is after validation."""

    def validate_input(self, content: str) -> bool:
        """Markdown is valid as long as it's non-empty."""
        return bool(content.strip())

    async def parse(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """Return the content directly (it's already Markdown)."""
        if not self.validate_input(content):
            raise ParsingError("Markdown content cannot be empty")
        return content.strip()
