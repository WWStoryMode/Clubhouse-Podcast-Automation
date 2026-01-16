"""Unit tests for downloader module."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.core.downloader import (
    download_clubhouse_video,
    validate_url,
    sanitize_filename,
    DownloadError,
)


class TestValidateUrl:
    """Tests for validate_url function."""

    def test_valid_https_url(self):
        """Test valid HTTPS URL."""
        assert validate_url("https://example.com/video.mp4") is True

    def test_valid_http_url(self):
        """Test valid HTTP URL."""
        assert validate_url("http://example.com/video.mp4") is True

    def test_invalid_url_no_scheme(self):
        """Test URL without scheme."""
        assert validate_url("example.com/video.mp4") is False

    def test_invalid_url_no_host(self):
        """Test URL without host."""
        assert validate_url("https:///video.mp4") is False

    def test_invalid_url_empty(self):
        """Test empty URL."""
        assert validate_url("") is False

    def test_invalid_url_ftp(self):
        """Test FTP URL (not supported)."""
        assert validate_url("ftp://example.com/video.mp4") is False


class TestSanitizeFilename:
    """Tests for sanitize_filename function."""

    def test_sanitize_normal_filename(self):
        """Test that normal filename is unchanged."""
        assert sanitize_filename("video.mp4") == "video.mp4"

    def test_sanitize_removes_invalid_chars(self):
        """Test that invalid characters are replaced."""
        assert sanitize_filename('video<>:"/\\|?*.mp4') == "video_________.mp4"

    def test_sanitize_strips_spaces(self):
        """Test that leading/trailing spaces are removed."""
        assert sanitize_filename("  video.mp4  ") == "video.mp4"

    def test_sanitize_strips_dots(self):
        """Test that leading/trailing dots are removed."""
        assert sanitize_filename("...video.mp4...") == "video.mp4"

    def test_sanitize_long_filename(self):
        """Test that long filenames are truncated."""
        long_name = "a" * 300
        result = sanitize_filename(long_name)
        assert len(result) <= 200

    def test_sanitize_empty_becomes_download(self):
        """Test that empty filename becomes 'download'."""
        assert sanitize_filename("") == "download"
        assert sanitize_filename("   ") == "download"


class TestDownloadClubhouseVideo:
    """Tests for download_clubhouse_video function."""

    def test_download_invalid_url(self, temp_output_dir):
        """Test that invalid URL raises ValueError."""
        with pytest.raises(ValueError, match="Invalid URL"):
            download_clubhouse_video("not-a-url", temp_output_dir)

    def test_download_empty_url(self, temp_output_dir):
        """Test that empty URL raises ValueError."""
        with pytest.raises(ValueError, match="Invalid URL"):
            download_clubhouse_video("", temp_output_dir)

    @patch("src.core.downloader.requests.get")
    def test_download_success(self, mock_get, temp_output_dir):
        """Test successful download."""
        # Mock response
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "1000"}
        mock_response.iter_content.return_value = [b"fake video content"]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = download_clubhouse_video(
            "https://example.com/recording.mp4",
            temp_output_dir,
            show_progress=False,
        )

        assert result.exists()
        assert result.suffix == ".mp4"

    @patch("src.core.downloader.requests.get")
    def test_download_with_custom_filename(self, mock_get, temp_output_dir):
        """Test download with custom filename."""
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "1000"}
        mock_response.iter_content.return_value = [b"fake video content"]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = download_clubhouse_video(
            "https://example.com/video",
            temp_output_dir,
            filename="my_episode",
            show_progress=False,
        )

        assert result.name == "my_episode.mp4"

    @patch("src.core.downloader.requests.get")
    def test_download_timeout(self, mock_get, temp_output_dir):
        """Test download timeout handling."""
        import requests

        mock_get.side_effect = requests.exceptions.Timeout()

        with pytest.raises(DownloadError, match="timed out"):
            download_clubhouse_video(
                "https://example.com/video.mp4",
                temp_output_dir,
            )

    @patch("src.core.downloader.requests.get")
    def test_download_http_error(self, mock_get, temp_output_dir):
        """Test HTTP error handling."""
        import requests

        mock_response = MagicMock()
        mock_response.status_code = 404
        error = requests.exceptions.HTTPError(response=mock_response)
        mock_get.return_value.raise_for_status.side_effect = error

        with pytest.raises(DownloadError, match="HTTP error"):
            download_clubhouse_video(
                "https://example.com/video.mp4",
                temp_output_dir,
            )

    @patch("src.core.downloader.requests.get")
    def test_download_connection_error(self, mock_get, temp_output_dir):
        """Test connection error handling."""
        import requests

        mock_get.side_effect = requests.exceptions.ConnectionError()

        with pytest.raises(DownloadError, match="Connection error"):
            download_clubhouse_video(
                "https://example.com/video.mp4",
                temp_output_dir,
            )

    @patch("src.core.downloader.requests.get")
    def test_download_creates_output_directory(self, mock_get, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "1000"}
        mock_response.iter_content.return_value = [b"fake video content"]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        nested_dir = tmp_path / "nested" / "output" / "dir"

        result = download_clubhouse_video(
            "https://example.com/video.mp4",
            nested_dir,
            show_progress=False,
        )

        assert nested_dir.exists()
        assert result.exists()

    @patch("src.core.downloader.requests.get")
    def test_download_extracts_filename_from_url(self, mock_get, temp_output_dir):
        """Test that filename is extracted from URL path."""
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "1000"}
        mock_response.iter_content.return_value = [b"fake video content"]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = download_clubhouse_video(
            "https://example.com/path/to/my_recording.mp4?token=abc",
            temp_output_dir,
            show_progress=False,
        )

        assert "my_recording" in result.name
