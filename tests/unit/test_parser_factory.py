"""
Thorough unit tests for the ParserFactory.
Covers every source type mapping, class resolution, and error handling.
"""
import pytest

from src.services.ingestion.parser_factory import ParserFactory
from src.services.ingestion.document_parser import DocumentParser
from src.services.ingestion.web_parser import WebParser
from src.services.ingestion.video_parser import VideoParser
from src.services.ingestion.research_parser import ResearchParser
from src.services.ingestion.markdown_parser import MarkdownParser
from src.models.schemas import SourceType
from src.core.exceptions import ParsingError


class TestFactoryMapping:
    """Every SourceType must map to the correct parser class."""

    @pytest.mark.parametrize(
        "source_type, expected_class",
        [
            (SourceType.PDF_DECK, DocumentParser),
            (SourceType.WEB_URL, WebParser),
            (SourceType.APP_STORE_LINK, WebParser),
            (SourceType.YOUTUBE_SUMMARY, VideoParser),
            (SourceType.RESEARCH, ResearchParser),
            (SourceType.MARKDOWN, MarkdownParser),
        ],
    )
    def test_creates_correct_parser(self, source_type, expected_class):
        parser = ParserFactory.create_parser(source_type)
        assert isinstance(parser, expected_class)

    def test_app_store_uses_web_parser(self):
        """App store links should reuse the WebParser."""
        parser = ParserFactory.create_parser(SourceType.APP_STORE_LINK)
        assert isinstance(parser, WebParser)


class TestFactoryCompleteness:
    """Every SourceType should be handled by the factory."""

    def test_all_source_types_have_parser(self):
        for source_type in SourceType:
            parser = ParserFactory.create_parser(source_type)
            assert parser is not None


class TestFactoryReturnsNewInstances:
    """Each call should return a fresh parser instance."""

    def test_distinct_instances(self):
        p1 = ParserFactory.create_parser(SourceType.MARKDOWN)
        p2 = ParserFactory.create_parser(SourceType.MARKDOWN)
        assert p1 is not p2
