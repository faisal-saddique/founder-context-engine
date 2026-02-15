"""
Video parser for YouTube using transcript API.
Extracts transcript and optionally summarizes using LLM.
"""
from typing import Dict, Any
import re
from youtube_transcript_api import YouTubeTranscriptApi

from .base import BaseParser
from ...core.logging import logger
from ...core.exceptions import ParsingError


class VideoParser(BaseParser):
    """Parser for YouTube videos."""
    
    def validate_input(self, content: str) -> bool:
        """Check if input is a valid YouTube URL."""
        youtube_patterns = [
            r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/',
            r'youtu\.be/',
            r'youtube\.com/watch\?v='
        ]
        return any(re.search(pattern, content) for pattern in youtube_patterns)
    
    def extract_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL."""
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'youtu\.be\/([0-9A-Za-z_-]{11})',
            r'embed\/([0-9A-Za-z_-]{11})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        raise ParsingError(f"Could not extract video ID from URL: {url}")
    
    async def parse(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """
        Extract YouTube transcript and format as Markdown.
        
        Args:
            content: YouTube URL
            metadata: Optional metadata
            
        Returns:
            Markdown-formatted transcript
        """
        if not self.validate_input(content):
            raise ParsingError(f"Invalid YouTube URL: {content}")
        
        try:
            video_id = self.extract_video_id(content)
            logger.info(f"Fetching transcript for video: {video_id}")
            
            # Get transcript
            api = YouTubeTranscriptApi()
            transcript = api.fetch(video_id)

            # Format as markdown
            markdown_content = f"# YouTube Video Transcript\n\n"
            markdown_content += f"**Video ID:** {video_id}\n"
            markdown_content += f"**URL:** {content}\n\n"
            markdown_content += "---\n\n"

            # Combine transcript entries
            full_transcript = " ".join([entry.text for entry in transcript])
            markdown_content += full_transcript
            
            logger.info(f"Successfully extracted transcript for video: {video_id}")
            return markdown_content
            
        except Exception as e:
            logger.error(f"Failed to extract transcript from {content}: {str(e)}")
            raise ParsingError(f"Video parsing failed: {str(e)}")
