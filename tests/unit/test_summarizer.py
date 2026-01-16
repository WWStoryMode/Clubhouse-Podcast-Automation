"""Unit tests for summarizer module."""

import os
from unittest.mock import patch, MagicMock

import pytest

from src.core.summarizer import (
    generate_descriptions,
    configure_gemini,
    _parse_response,
    _parse_tags,
    SummaryError,
)


class TestConfigureGemini:
    """Tests for configure_gemini function."""

    @patch("src.core.summarizer.genai.configure")
    def test_configure_with_api_key(self, mock_configure):
        """Test configuration with explicit API key."""
        configure_gemini("test-api-key")
        mock_configure.assert_called_once_with(api_key="test-api-key")

    @patch.dict(os.environ, {"GEMINI_API_KEY": "env-api-key"})
    @patch("src.core.summarizer.genai.configure")
    def test_configure_from_env(self, mock_configure):
        """Test configuration from environment variable."""
        configure_gemini()
        mock_configure.assert_called_once_with(api_key="env-api-key")

    @patch.dict(os.environ, {}, clear=True)
    def test_configure_no_key_raises_error(self):
        """Test that missing API key raises SummaryError."""
        os.environ.pop("GEMINI_API_KEY", None)

        with pytest.raises(SummaryError, match="API key not provided"):
            configure_gemini()


class TestGenerateDescriptions:
    """Tests for generate_descriptions function."""

    def test_empty_transcript_raises_error(self):
        """Test that empty transcript raises SummaryError."""
        with pytest.raises(SummaryError, match="Transcript is empty"):
            generate_descriptions("", "Episode Title", api_key="test-key")

    def test_empty_title_raises_error(self):
        """Test that empty title raises SummaryError."""
        with pytest.raises(SummaryError, match="Episode title is empty"):
            generate_descriptions("Some transcript", "", api_key="test-key")

    @patch("src.core.summarizer.genai")
    def test_generate_success(self, mock_genai):
        """Test successful description generation."""
        mock_response = MagicMock()
        mock_response.text = """YOUTUBE_TITLE: Great Episode About Tech
YOUTUBE_DESCRIPTION: This is a great episode about technology.
- Topic 1
- Topic 2
SPOTIFY_TITLE: Tech Talk Episode
SPOTIFY_DESCRIPTION: A conversational episode about tech trends.
TAGS: technology, podcast, innovation, startup"""

        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        result = generate_descriptions(
            "This is a transcript about technology...",
            "Episode 1",
            api_key="test-key",
        )

        assert result["youtube_title"] == "Great Episode About Tech"
        assert "great episode" in result["youtube_description"].lower()
        assert result["spotify_title"] == "Tech Talk Episode"
        assert len(result["tags"]) > 0

    @patch("src.core.summarizer.genai")
    def test_generate_without_tags(self, mock_genai):
        """Test generation with tags disabled."""
        mock_response = MagicMock()
        mock_response.text = """YOUTUBE_TITLE: Test Title
YOUTUBE_DESCRIPTION: Test description.
SPOTIFY_TITLE: Test Title
SPOTIFY_DESCRIPTION: Test description.
TAGS: tag1, tag2"""

        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        result = generate_descriptions(
            "Transcript text",
            "Episode 1",
            api_key="test-key",
            generate_tags=False,
        )

        assert result["tags"] == []

    @patch("src.core.summarizer.genai")
    def test_generate_empty_response(self, mock_genai):
        """Test handling of empty API response."""
        mock_response = MagicMock()
        mock_response.text = ""

        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        with pytest.raises(SummaryError, match="empty response"):
            generate_descriptions(
                "Transcript text",
                "Episode 1",
                api_key="test-key",
            )

    @patch("src.core.summarizer.genai")
    def test_generate_api_error(self, mock_genai):
        """Test handling of API errors."""
        mock_genai.GenerativeModel.side_effect = Exception("API error")

        with pytest.raises(SummaryError, match="Summary generation failed"):
            generate_descriptions(
                "Transcript text",
                "Episode 1",
                api_key="test-key",
            )


class TestParseResponse:
    """Tests for _parse_response function."""

    def test_parse_complete_response(self):
        """Test parsing a complete well-formatted response."""
        response = """YOUTUBE_TITLE: My Great Video
YOUTUBE_DESCRIPTION: This is a description
with multiple lines.
SPOTIFY_TITLE: My Podcast Episode
SPOTIFY_DESCRIPTION: Podcast description here.
TAGS: tag1, tag2, tag3"""

        result = _parse_response(response, "Fallback", True)

        assert result["youtube_title"] == "My Great Video"
        assert "multiple lines" in result["youtube_description"]
        assert result["spotify_title"] == "My Podcast Episode"
        assert result["tags"] == ["tag1", "tag2", "tag3"]

    def test_parse_missing_fields_uses_fallback(self):
        """Test that missing fields use fallback values."""
        response = """YOUTUBE_DESCRIPTION: Only description provided."""

        result = _parse_response(response, "Fallback Title", True)

        assert result["youtube_title"] == "Fallback Title"
        assert result["spotify_title"] == "Fallback Title"
        assert "Only description" in result["youtube_description"]

    def test_parse_empty_response(self):
        """Test parsing empty response."""
        result = _parse_response("", "Fallback", True)

        assert result["youtube_title"] == "Fallback"
        assert result["youtube_description"] == ""
        assert result["tags"] == []


class TestParseTags:
    """Tests for _parse_tags function."""

    def test_parse_comma_separated_tags(self):
        """Test parsing comma-separated tags."""
        tags = _parse_tags("tag1, tag2, tag3")
        assert tags == ["tag1", "tag2", "tag3"]

    def test_parse_newline_separated_tags(self):
        """Test parsing newline-separated tags."""
        tags = _parse_tags("tag1\ntag2\ntag3")
        assert tags == ["tag1", "tag2", "tag3"]

    def test_parse_tags_with_hashtags(self):
        """Test that hashtags are stripped."""
        tags = _parse_tags("#tag1, #tag2, #tag3")
        assert tags == ["tag1", "tag2", "tag3"]

    def test_parse_empty_tags(self):
        """Test parsing empty string."""
        tags = _parse_tags("")
        assert tags == []

    def test_parse_tags_with_extra_spaces(self):
        """Test that extra spaces are trimmed."""
        tags = _parse_tags("  tag1  ,  tag2  ,  tag3  ")
        assert tags == ["tag1", "tag2", "tag3"]
