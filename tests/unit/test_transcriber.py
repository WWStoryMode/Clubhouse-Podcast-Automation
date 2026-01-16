"""Unit tests for transcriber module."""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.core.transcriber import (
    transcribe_audio,
    configure_gemini,
    TranscriptionError,
)


class TestConfigureGemini:
    """Tests for configure_gemini function."""

    @patch("src.core.transcriber.genai.configure")
    def test_configure_with_api_key(self, mock_configure):
        """Test configuration with explicit API key."""
        configure_gemini("test-api-key")
        mock_configure.assert_called_once_with(api_key="test-api-key")

    @patch.dict(os.environ, {"GEMINI_API_KEY": "env-api-key"})
    @patch("src.core.transcriber.genai.configure")
    def test_configure_from_env(self, mock_configure):
        """Test configuration from environment variable."""
        configure_gemini()
        mock_configure.assert_called_once_with(api_key="env-api-key")

    @patch.dict(os.environ, {}, clear=True)
    def test_configure_no_key_raises_error(self):
        """Test that missing API key raises TranscriptionError."""
        # Remove GEMINI_API_KEY if it exists
        os.environ.pop("GEMINI_API_KEY", None)

        with pytest.raises(TranscriptionError, match="API key not provided"):
            configure_gemini()


class TestTranscribeAudio:
    """Tests for transcribe_audio function."""

    def test_transcribe_file_not_found(self, temp_output_dir):
        """Test that FileNotFoundError is raised for missing input."""
        fake_path = temp_output_dir / "nonexistent.mp3"

        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            transcribe_audio(fake_path, api_key="test-key")

    def test_transcribe_path_is_directory(self, temp_output_dir):
        """Test that FileNotFoundError is raised when path is a directory."""
        with pytest.raises(FileNotFoundError, match="Path is not a file"):
            transcribe_audio(temp_output_dir, api_key="test-key")

    @patch("src.core.transcriber.genai")
    def test_transcribe_success(self, mock_genai, temp_output_dir):
        """Test successful transcription with mocked API."""
        # Create a dummy audio file
        audio_path = temp_output_dir / "test.mp3"
        audio_path.write_bytes(b"fake audio content")

        # Mock the API responses
        mock_file = MagicMock()
        mock_genai.upload_file.return_value = mock_file

        mock_response = MagicMock()
        mock_response.text = "This is the transcribed text."

        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        result = transcribe_audio(audio_path, api_key="test-key")

        assert result == "This is the transcribed text."
        mock_genai.configure.assert_called_once_with(api_key="test-key")
        mock_genai.upload_file.assert_called_once()
        mock_model.generate_content.assert_called_once()

    @patch("src.core.transcriber.genai")
    def test_transcribe_with_timestamps(self, mock_genai, temp_output_dir):
        """Test transcription with timestamps option."""
        audio_path = temp_output_dir / "test.mp3"
        audio_path.write_bytes(b"fake audio content")

        mock_file = MagicMock()
        mock_genai.upload_file.return_value = mock_file

        mock_response = MagicMock()
        mock_response.text = "[00:00] Hello world."

        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        result = transcribe_audio(
            audio_path,
            api_key="test-key",
            include_timestamps=True,
        )

        assert "[00:00]" in result

        # Verify the prompt includes timestamp instructions
        call_args = mock_model.generate_content.call_args
        prompt = call_args[0][0][0]  # First arg, first element
        assert "timestamp" in prompt.lower()

    @patch("src.core.transcriber.genai")
    def test_transcribe_empty_response(self, mock_genai, temp_output_dir):
        """Test handling of empty API response."""
        audio_path = temp_output_dir / "test.mp3"
        audio_path.write_bytes(b"fake audio content")

        mock_file = MagicMock()
        mock_genai.upload_file.return_value = mock_file

        mock_response = MagicMock()
        mock_response.text = ""

        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        with pytest.raises(TranscriptionError, match="empty response"):
            transcribe_audio(audio_path, api_key="test-key")

    @patch("src.core.transcriber.genai")
    def test_transcribe_api_error(self, mock_genai, temp_output_dir):
        """Test handling of API errors."""
        audio_path = temp_output_dir / "test.mp3"
        audio_path.write_bytes(b"fake audio content")

        mock_genai.upload_file.side_effect = Exception("API error occurred")

        with pytest.raises(TranscriptionError, match="Transcription failed"):
            transcribe_audio(audio_path, api_key="test-key")

    @patch("src.core.transcriber.genai")
    def test_transcribe_different_language(self, mock_genai, temp_output_dir):
        """Test transcription with different language setting."""
        audio_path = temp_output_dir / "test.mp3"
        audio_path.write_bytes(b"fake audio content")

        mock_file = MagicMock()
        mock_genai.upload_file.return_value = mock_file

        mock_response = MagicMock()
        mock_response.text = "Transcribed Chinese text"

        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        result = transcribe_audio(
            audio_path,
            api_key="test-key",
            language="zh",
        )

        # Verify the prompt includes the language
        call_args = mock_model.generate_content.call_args
        prompt = call_args[0][0][0]
        assert "zh" in prompt

    def test_transcribe_with_real_file_and_api(self, sample_audio, mock_gemini_api_key):
        """Test transcription with real audio file and API.

        This test requires:
        1. GEMINI_API_KEY environment variable to be set
        2. A sample audio file at tests/fixtures/sample_audio.mp3
        """
        if mock_gemini_api_key == "test-api-key-for-testing":
            pytest.skip("Real GEMINI_API_KEY not set")

        result = transcribe_audio(
            sample_audio,
            api_key=mock_gemini_api_key,
            language="en",
        )

        assert isinstance(result, str)
        assert len(result) > 0
