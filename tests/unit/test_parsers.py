"""
Comprehensive unit tests for each content parser.
External services (Firecrawl, LlamaParse, YouTube, Tavily) are mocked
so tests run offline and fast.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.ingestion.markdown_parser import MarkdownParser
from src.services.ingestion.web_parser import WebParser
from src.services.ingestion.video_parser import VideoParser
from src.services.ingestion.document_parser import DocumentParser
from src.services.ingestion.research_parser import ResearchParser
from src.services.ingestion.parser_factory import ParserFactory
from src.models.schemas import SourceType
from src.core.exceptions import ParsingError


# ── MarkdownParser (no external deps) ─────────────────────────────

class TestMarkdownParser:
    """Markdown is a pass-through — fast and straightforward."""

    @pytest.mark.asyncio
    async def test_returns_trimmed_content(self):
        parser = MarkdownParser()
        result = await parser.parse("  # Hello World  ")
        assert result == "# Hello World"

    @pytest.mark.asyncio
    async def test_preserves_formatting(self, canonical_brand_facts):
        parser = MarkdownParser()
        result = await parser.parse(canonical_brand_facts)
        assert "Company Overview" in result
        assert "Core Capabilities" in result

    @pytest.mark.asyncio
    async def test_raises_on_empty(self):
        parser = MarkdownParser()
        with pytest.raises(ParsingError, match="cannot be empty"):
            await parser.parse("")

    @pytest.mark.asyncio
    async def test_raises_on_whitespace_only(self):
        parser = MarkdownParser()
        with pytest.raises(ParsingError, match="cannot be empty"):
            await parser.parse("   \n\t  ")

    def test_validate_input_true(self):
        parser = MarkdownParser()
        assert parser.validate_input("# Title") is True

    def test_validate_input_false(self):
        parser = MarkdownParser()
        assert parser.validate_input("") is False
        assert parser.validate_input("   ") is False

    @pytest.mark.asyncio
    async def test_real_dataset_icp(self, icp_profiles):
        """Parse ICP document from the dataset."""
        parser = MarkdownParser()
        result = await parser.parse(icp_profiles)
        assert "Mobile App Startups" in result
        assert "Subscription-Based App Businesses" in result

    @pytest.mark.asyncio
    async def test_real_dataset_playbook(self, aso_playbook):
        """Parse ASO playbook from the dataset."""
        parser = MarkdownParser()
        result = await parser.parse(aso_playbook)
        assert "ASO" in result
        assert "Playbook" in result

    @pytest.mark.asyncio
    async def test_metadata_is_ignored(self):
        """Markdown parser doesn't use metadata."""
        parser = MarkdownParser()
        result = await parser.parse("# Test", metadata={"key": "val"})
        assert result == "# Test"


# ── WebParser ─────────────────────────────────────────────────────

class TestWebParserValidation:
    """URL validation logic."""

    def test_valid_https(self):
        parser = WebParser()
        assert parser.validate_input("https://example.com") is True

    def test_valid_http(self):
        parser = WebParser()
        assert parser.validate_input("http://example.com") is True

    def test_invalid_no_scheme(self):
        parser = WebParser()
        assert parser.validate_input("example.com") is False

    def test_invalid_local_path(self):
        parser = WebParser()
        assert parser.validate_input("/local/path") is False

    def test_invalid_empty(self):
        parser = WebParser()
        assert parser.validate_input("") is False

    def test_invalid_ftp(self):
        parser = WebParser()
        assert parser.validate_input("ftp://files.com/doc") is False


class TestWebParserParse:
    @pytest.mark.asyncio
    async def test_raises_on_invalid_url(self):
        parser = WebParser()
        with pytest.raises(ParsingError, match="Invalid URL"):
            await parser.parse("not-a-url")

    @pytest.mark.asyncio
    async def test_scrapes_successfully(self):
        parser = WebParser()
        mock_result = MagicMock()
        mock_result.markdown = "# Scraped Content\n\nSome text here."
        parser.client = MagicMock()
        parser.client.scrape = MagicMock(return_value=mock_result)

        result = await parser.parse("https://example.com")
        assert "Scraped Content" in result

    @pytest.mark.asyncio
    async def test_raises_on_empty_scrape(self):
        parser = WebParser()
        mock_result = MagicMock()
        mock_result.markdown = ""
        parser.client = MagicMock()
        parser.client.scrape = MagicMock(return_value=mock_result)

        with pytest.raises(ParsingError, match="No content extracted"):
            await parser.parse("https://example.com")

    @pytest.mark.asyncio
    async def test_raises_on_scrape_error(self):
        parser = WebParser()
        parser.client = MagicMock()
        parser.client.scrape = MagicMock(side_effect=Exception("Network error"))

        with pytest.raises(ParsingError, match="Web scraping failed"):
            await parser.parse("https://example.com")


# ── VideoParser ───────────────────────────────────────────────────

class TestVideoParserValidation:
    """YouTube URL validation patterns."""

    @pytest.mark.parametrize("url", [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://www.youtube.com/embed/dQw4w9WgXcQ",
        "http://youtube.com/watch?v=abc123_-XYZ",
    ])
    def test_valid_youtube_urls(self, url):
        parser = VideoParser()
        assert parser.validate_input(url) is True

    @pytest.mark.parametrize("url", [
        "https://example.com",
        "https://vimeo.com/123",
        "not-a-url",
        "",
    ])
    def test_invalid_urls(self, url):
        parser = VideoParser()
        assert parser.validate_input(url) is False


class TestVideoParserIdExtraction:
    @pytest.mark.parametrize("url, expected_id", [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtube.com/watch?v=abc123_-XYZ", "abc123_-XYZ"),
    ])
    def test_extract_video_id(self, url, expected_id):
        parser = VideoParser()
        assert parser.extract_video_id(url) == expected_id

    def test_raises_on_bad_url(self):
        parser = VideoParser()
        with pytest.raises(ParsingError, match="Could not extract"):
            parser.extract_video_id("https://example.com/no-video")


class TestVideoParserParse:
    @pytest.mark.asyncio
    async def test_raises_on_invalid_url(self):
        parser = VideoParser()
        with pytest.raises(ParsingError, match="Invalid YouTube URL"):
            await parser.parse("https://example.com")

    @pytest.mark.asyncio
    async def test_successful_transcript(self):
        parser = VideoParser()

        # mock the transcript API
        mock_entry_1 = MagicMock()
        mock_entry_1.text = "Hello everyone."
        mock_entry_2 = MagicMock()
        mock_entry_2.text = "Welcome to the video."

        mock_api = MagicMock()
        mock_api.fetch = MagicMock(return_value=[mock_entry_1, mock_entry_2])

        with patch("src.services.ingestion.video_parser.YouTubeTranscriptApi", return_value=mock_api):
            result = await parser.parse("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

        assert "YouTube Video Transcript" in result
        assert "dQw4w9WgXcQ" in result
        assert "Hello everyone." in result
        assert "Welcome to the video." in result

    @pytest.mark.asyncio
    async def test_transcript_api_failure(self):
        parser = VideoParser()
        mock_api = MagicMock()
        mock_api.fetch = MagicMock(side_effect=Exception("Transcript disabled"))

        with patch("src.services.ingestion.video_parser.YouTubeTranscriptApi", return_value=mock_api):
            with pytest.raises(ParsingError, match="Video parsing failed"):
                await parser.parse("https://www.youtube.com/watch?v=dQw4w9WgXcQ")


# ── DocumentParser ────────────────────────────────────────────────

class TestDocumentParserValidation:
    def test_invalid_nonexistent_file(self):
        parser = DocumentParser()
        assert parser.validate_input("/nonexistent/file.pdf") is False

    def test_invalid_extension(self, tmp_path):
        txt_file = tmp_path / "file.txt"
        txt_file.write_text("hello")
        parser = DocumentParser()
        assert parser.validate_input(str(txt_file)) is False

    def test_valid_pdf_exists(self, tmp_path):
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake content")
        parser = DocumentParser()
        assert parser.validate_input(str(pdf_file)) is True

    def test_valid_pptx_exists(self, tmp_path):
        pptx_file = tmp_path / "deck.pptx"
        pptx_file.write_bytes(b"PK fake")
        parser = DocumentParser()
        assert parser.validate_input(str(pptx_file)) is True


class TestDocumentParserParse:
    @pytest.mark.asyncio
    async def test_raises_on_invalid_file(self):
        parser = DocumentParser()
        with pytest.raises(ParsingError, match="Invalid document"):
            await parser.parse("/nonexistent.pdf")

    @pytest.mark.asyncio
    async def test_successful_parse(self, tmp_path):
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF fake")
        parser = DocumentParser()

        mock_doc = MagicMock()
        mock_doc.text = "# Parsed PDF Content"
        parser.client = MagicMock()
        parser.client.load_data = MagicMock(return_value=[mock_doc])

        result = await parser.parse(str(pdf_file))
        assert "Parsed PDF Content" in result

    @pytest.mark.asyncio
    async def test_multi_page_document(self, tmp_path):
        pdf_file = tmp_path / "multi.pdf"
        pdf_file.write_bytes(b"%PDF fake")
        parser = DocumentParser()

        page1, page2 = MagicMock(), MagicMock()
        page1.text = "# Page 1"
        page2.text = "# Page 2"
        parser.client = MagicMock()
        parser.client.load_data = MagicMock(return_value=[page1, page2])

        result = await parser.parse(str(pdf_file))
        assert "Page 1" in result
        assert "Page 2" in result


# ── ResearchParser ────────────────────────────────────────────────

class TestResearchParserValidation:
    def test_valid_query(self):
        parser = ResearchParser()
        assert parser.validate_input("What is ASO?") is True

    def test_empty_query(self):
        parser = ResearchParser()
        assert parser.validate_input("") is False

    def test_whitespace_query(self):
        parser = ResearchParser()
        assert parser.validate_input("   ") is False


class TestResearchParserParse:
    @pytest.mark.asyncio
    async def test_raises_on_empty(self):
        parser = ResearchParser()
        with pytest.raises(ParsingError, match="cannot be empty"):
            await parser.parse("")

    @pytest.mark.asyncio
    async def test_basic_search(self):
        parser = ResearchParser()
        parser.client = MagicMock()
        parser.client.search = MagicMock(return_value={
            "results": [
                {"title": "ASO Guide", "url": "https://example.com", "content": "ASO tips"},
            ],
            "answer": "ASO is about optimizing app visibility.",
        })

        result = await parser.parse("What is ASO?")
        assert "Research Query: What is ASO?" in result
        assert "ASO Guide" in result
        assert "ASO is about optimizing app visibility." in result

    @pytest.mark.asyncio
    async def test_deep_research_flag(self):
        parser = ResearchParser()
        parser.client = MagicMock()
        parser.client.search = MagicMock(return_value={"results": []})

        await parser.parse("Deep query", metadata={"deep_research": True})
        parser.client.search.assert_called_once_with(
            query="Deep query",
            search_depth="advanced",
            max_results=5,
        )

    @pytest.mark.asyncio
    async def test_basic_search_depth_default(self):
        parser = ResearchParser()
        parser.client = MagicMock()
        parser.client.search = MagicMock(return_value={"results": []})

        await parser.parse("Simple query")
        parser.client.search.assert_called_once_with(
            query="Simple query",
            search_depth="basic",
            max_results=5,
        )

    @pytest.mark.asyncio
    async def test_api_failure(self):
        parser = ResearchParser()
        parser.client = MagicMock()
        parser.client.search = MagicMock(side_effect=Exception("API down"))

        with pytest.raises(ParsingError, match="Research failed"):
            await parser.parse("Query")
