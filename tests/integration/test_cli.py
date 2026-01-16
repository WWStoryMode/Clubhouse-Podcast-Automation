"""Integration tests for CLI module."""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner

from src.cli import cli, load_config


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_default_config_when_file_missing(self):
        """Test that defaults are returned when config file doesn't exist."""
        config = load_config(Path("/nonexistent/config.yaml"))

        assert config["mode"] == "local"
        assert "output_dir" in config["local"]

    def test_load_config_from_file(self, tmp_path):
        """Test loading config from YAML file."""
        config_content = """
mode: local
local:
  output_dir: ./custom_output
  ffmpeg_path: /usr/bin/ffmpeg
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        config = load_config(config_file)

        assert config["mode"] == "local"
        assert config["local"]["output_dir"] == "./custom_output"


class TestCLI:
    """Tests for CLI commands."""

    def test_cli_help(self):
        """Test that --help works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "Clubhouse-Podcast-Automation" in result.output

    def test_download_command_help(self):
        """Test download command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["download", "--help"])

        assert result.exit_code == 0
        assert "--url" in result.output

    def test_extract_command_help(self):
        """Test extract command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["extract", "--help"])

        assert result.exit_code == 0
        assert "--input" in result.output

    def test_transcribe_command_help(self):
        """Test transcribe command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["transcribe", "--help"])

        assert result.exit_code == 0
        assert "--input" in result.output
        assert "--language" in result.output

    def test_summarize_command_help(self):
        """Test summarize command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["summarize", "--help"])

        assert result.exit_code == 0
        assert "--title" in result.output

    def test_process_command_help(self):
        """Test process command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["process", "--help"])

        assert result.exit_code == 0
        assert "--url" in result.output
        assert "--title" in result.output


class TestDownloadCommand:
    """Tests for download command."""

    @patch("src.cli.download_clubhouse_video")
    def test_download_success(self, mock_download, tmp_path):
        """Test successful download."""
        mock_download.return_value = tmp_path / "video.mp4"

        runner = CliRunner()
        result = runner.invoke(cli, [
            "download",
            "--url", "https://example.com/video.mp4",
            "--output", str(tmp_path),
        ])

        assert result.exit_code == 0
        assert "Downloaded to" in result.output

    @patch("src.cli.download_clubhouse_video")
    def test_download_invalid_url(self, mock_download):
        """Test download with invalid URL."""
        from src.core.downloader import DownloadError
        mock_download.side_effect = ValueError("Invalid URL")

        runner = CliRunner()
        result = runner.invoke(cli, [
            "download",
            "--url", "not-a-url",
        ])

        assert result.exit_code == 1
        assert "Error" in result.output


class TestExtractCommand:
    """Tests for extract command."""

    @patch("src.cli.extract_audio")
    def test_extract_success(self, mock_extract, tmp_path):
        """Test successful extraction."""
        # Create dummy input file
        input_file = tmp_path / "video.mp4"
        input_file.write_bytes(b"fake video")

        mock_extract.return_value = tmp_path / "video.mp3"

        runner = CliRunner()
        result = runner.invoke(cli, [
            "extract",
            "--input", str(input_file),
        ])

        assert result.exit_code == 0
        assert "Extracted to" in result.output

    def test_extract_file_not_found(self, tmp_path):
        """Test extraction with missing input file."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            "extract",
            "--input", str(tmp_path / "nonexistent.mp4"),
        ])

        assert result.exit_code == 2  # Click's error for invalid path


class TestTranscribeCommand:
    """Tests for transcribe command."""

    @patch("src.cli.transcribe_audio")
    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
    def test_transcribe_success(self, mock_transcribe, tmp_path):
        """Test successful transcription."""
        # Create dummy input file
        input_file = tmp_path / "audio.mp3"
        input_file.write_bytes(b"fake audio")

        mock_transcribe.return_value = "This is the transcript."

        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, [
                "transcribe",
                "--input", str(input_file),
                "--output", str(tmp_path / "transcript.txt"),
            ])

        assert result.exit_code == 0
        assert "Transcript saved" in result.output

    def test_transcribe_missing_api_key(self, tmp_path):
        """Test transcription without API key."""
        input_file = tmp_path / "audio.mp3"
        input_file.write_bytes(b"fake audio")

        runner = CliRunner()
        # Ensure no API key in environment
        env = os.environ.copy()
        env.pop("GEMINI_API_KEY", None)

        with patch.dict(os.environ, env, clear=True):
            result = runner.invoke(cli, [
                "transcribe",
                "--input", str(input_file),
            ])

        assert result.exit_code == 1
        assert "GEMINI_API_KEY" in result.output


class TestSummarizeCommand:
    """Tests for summarize command."""

    @patch("src.cli.generate_descriptions")
    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
    def test_summarize_success(self, mock_generate, tmp_path):
        """Test successful summary generation."""
        # Create dummy transcript file
        transcript_file = tmp_path / "transcript.txt"
        transcript_file.write_text("This is a test transcript.")

        mock_generate.return_value = {
            "youtube_title": "Test Title",
            "youtube_description": "Test description",
            "spotify_title": "Test Title",
            "spotify_description": "Test description",
            "tags": ["test", "podcast"],
        }

        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, [
                "summarize",
                "--input", str(transcript_file),
                "--title", "Test Episode",
                "--output", str(tmp_path),
            ])

        assert result.exit_code == 0
        assert "Descriptions saved" in result.output


class TestProcessCommand:
    """Tests for process command (full pipeline)."""

    @patch("src.cli.generate_descriptions")
    @patch("src.cli.transcribe_audio")
    @patch("src.cli.extract_audio")
    @patch("src.cli.download_clubhouse_video")
    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
    def test_process_full_pipeline(
        self, mock_download, mock_extract, mock_transcribe, mock_generate, tmp_path
    ):
        """Test full processing pipeline."""
        # Set up mocks
        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"fake video")
        mock_download.return_value = video_path

        audio_path = tmp_path / "video.mp3"
        audio_path.write_bytes(b"fake audio")
        mock_extract.return_value = audio_path

        mock_transcribe.return_value = "This is the transcript."

        mock_generate.return_value = {
            "youtube_title": "Test Title",
            "youtube_description": "Test description",
            "spotify_title": "Test Title",
            "spotify_description": "Test description",
            "tags": ["test", "podcast"],
        }

        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, [
                "process",
                "--url", "https://example.com/video.mp4",
                "--title", "Test Episode",
                "--output", str(tmp_path),
            ])

        assert result.exit_code == 0
        assert "Processing complete" in result.output
        assert "[1/4]" in result.output
        assert "[4/4]" in result.output

    def test_process_missing_api_key(self):
        """Test process command without API key."""
        runner = CliRunner()
        env = os.environ.copy()
        env.pop("GEMINI_API_KEY", None)

        with patch.dict(os.environ, env, clear=True):
            result = runner.invoke(cli, [
                "process",
                "--url", "https://example.com/video.mp4",
                "--title", "Test Episode",
            ])

        assert result.exit_code == 1
        assert "GEMINI_API_KEY" in result.output
