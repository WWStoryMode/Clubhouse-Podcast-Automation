"""Unit tests for transcriber module."""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.core.transcriber import (
    transcribe_audio,
    get_gemini_client,
    validate_transcript_quality,
    transcribe_audio_chunked,
    TranscriptionError,
)


class TestGetGeminiClient:
    """Tests for get_gemini_client function."""

    @patch("src.core.transcriber.genai.Client")
    def test_configure_with_api_key(self, mock_client_cls):
        """Test configuration with explicit API key."""
        get_gemini_client("test-api-key")
        mock_client_cls.assert_called_once_with(api_key="test-api-key")

    @patch.dict(os.environ, {"GEMINI_API_KEY": "env-api-key"})
    @patch("src.core.transcriber.genai.Client")
    def test_configure_from_env(self, mock_client_cls):
        """Test configuration from environment variable."""
        get_gemini_client()
        mock_client_cls.assert_called_once_with(api_key="env-api-key")

    @patch.dict(os.environ, {}, clear=True)
    def test_configure_no_key_raises_error(self):
        """Test that missing API key raises TranscriptionError."""
        os.environ.pop("GEMINI_API_KEY", None)

        with pytest.raises(TranscriptionError, match="API key not provided"):
            get_gemini_client()


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

    @patch("src.core.transcriber.get_gemini_client")
    def test_transcribe_success(self, mock_get_client, temp_output_dir):
        """Test successful transcription with mocked API."""
        audio_path = temp_output_dir / "test.mp3"
        audio_path.write_bytes(b"fake audio content")

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_file = MagicMock()
        mock_file.name = "files/test123"
        mock_client.files.upload.return_value = mock_file

        mock_response = MagicMock()
        mock_response.text = "This is the transcribed text."
        mock_client.models.generate_content.return_value = mock_response

        result = transcribe_audio(audio_path, api_key="test-key")

        assert result == "This is the transcribed text."
        mock_get_client.assert_called_once_with("test-key")
        mock_client.files.upload.assert_called_once()
        mock_client.models.generate_content.assert_called_once()

    @patch("src.core.transcriber.get_gemini_client")
    def test_transcribe_with_timestamps(self, mock_get_client, temp_output_dir):
        """Test transcription with timestamps option."""
        audio_path = temp_output_dir / "test.mp3"
        audio_path.write_bytes(b"fake audio content")

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_file = MagicMock()
        mock_file.name = "files/test123"
        mock_client.files.upload.return_value = mock_file

        mock_response = MagicMock()
        mock_response.text = "[00:00] Hello world."
        mock_client.models.generate_content.return_value = mock_response

        result = transcribe_audio(
            audio_path,
            api_key="test-key",
            include_timestamps=True,
        )

        assert "[00:00]" in result

        # Verify the prompt includes timestamp instructions
        call_args = mock_client.models.generate_content.call_args
        contents = call_args[1]["contents"] if "contents" in call_args[1] else call_args[0][0]
        prompt = contents[0] if isinstance(contents, list) else contents
        assert "timestamp" in str(prompt).lower()

    @patch("src.core.transcriber.get_gemini_client")
    def test_transcribe_empty_response(self, mock_get_client, temp_output_dir):
        """Test handling of empty API response."""
        audio_path = temp_output_dir / "test.mp3"
        audio_path.write_bytes(b"fake audio content")

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_file = MagicMock()
        mock_file.name = "files/test123"
        mock_client.files.upload.return_value = mock_file

        mock_response = MagicMock()
        mock_response.text = ""
        mock_client.models.generate_content.return_value = mock_response

        with pytest.raises(TranscriptionError, match="empty response"):
            transcribe_audio(audio_path, api_key="test-key")

    @patch("src.core.transcriber.get_gemini_client")
    def test_transcribe_api_error(self, mock_get_client, temp_output_dir):
        """Test handling of API errors."""
        audio_path = temp_output_dir / "test.mp3"
        audio_path.write_bytes(b"fake audio content")

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.files.upload.side_effect = Exception("API error occurred")

        with pytest.raises(TranscriptionError, match="Transcription failed"):
            transcribe_audio(audio_path, api_key="test-key")

    @patch("src.core.transcriber.get_gemini_client")
    def test_transcribe_different_language(self, mock_get_client, temp_output_dir):
        """Test transcription with different language setting."""
        audio_path = temp_output_dir / "test.mp3"
        audio_path.write_bytes(b"fake audio content")

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_file = MagicMock()
        mock_file.name = "files/test123"
        mock_client.files.upload.return_value = mock_file

        mock_response = MagicMock()
        mock_response.text = "Transcribed Chinese text"
        mock_client.models.generate_content.return_value = mock_response

        result = transcribe_audio(
            audio_path,
            api_key="test-key",
            language="zh",
        )

        assert result == "Transcribed Chinese text"

        # Verify the prompt includes the language description
        call_args = mock_client.models.generate_content.call_args
        contents = call_args[1]["contents"] if "contents" in call_args[1] else call_args[0][0]
        prompt = contents[0] if isinstance(contents, list) else contents
        assert "Mandarin" in str(prompt) or "Chinese" in str(prompt)

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


class TestValidateTranscriptQuality:
    """Tests for validate_transcript_quality function."""

    def test_valid_transcript_passes(self):
        """A normal transcript should pass validation."""
        transcript = (
            "Hello everyone, welcome to the show. Today we're going to talk about "
            "technology and its impact on modern society. We have several guests joining us. "
            "Our first topic is artificial intelligence and how it is changing the world. "
            "Many companies are investing heavily in AI research and development. "
            "The implications for healthcare, education, and transportation are enormous. "
            "Let us hear from our first guest who has been working in this field for years. "
            "Thank you for having me, I am excited to share my perspective on these developments."
        )
        is_valid, reason = validate_transcript_quality(transcript, chunk_duration_seconds=600)
        assert is_valid is True
        assert reason == ""

    def test_empty_transcript_fails(self):
        """Empty transcript should fail."""
        is_valid, reason = validate_transcript_quality("", chunk_duration_seconds=600)
        assert is_valid is False
        assert "too short" in reason.lower()

    def test_whitespace_only_fails(self):
        """Whitespace-only transcript should fail."""
        is_valid, reason = validate_transcript_quality("   \n\t  ", chunk_duration_seconds=600)
        assert is_valid is False
        assert "too short" in reason.lower()

    def test_near_empty_fails(self):
        """Transcript with less than 20 chars should fail."""
        is_valid, reason = validate_transcript_quality("Hello world", chunk_duration_seconds=600)
        assert is_valid is False
        assert "too short" in reason.lower()

    def test_short_transcript_for_long_chunk_fails(self):
        """Short transcript for a 10-minute chunk should fail."""
        # 10 minutes = 600 seconds, threshold = 50 * 10 = 500 chars
        transcript = "This is a short transcript that is only about a hundred chars long. Not enough for ten minutes."
        assert len(transcript) < 500
        is_valid, reason = validate_transcript_quality(transcript, chunk_duration_seconds=600)
        assert is_valid is False
        assert "too short for duration" in reason.lower()

    def test_final_chunk_relaxed_threshold(self):
        """Final chunk should use 50% relaxed threshold."""
        # 5 minutes = 300 seconds, normal threshold = 50 * 5 = 250
        # Relaxed threshold = 250 * 0.5 = 125
        # Use unique content to avoid triggering character-pattern check
        transcript = " ".join(f"word{i}" for i in range(25))  # ~140 chars of unique text
        assert 125 < len(transcript) < 250
        is_valid, reason = validate_transcript_quality(
            transcript, chunk_duration_seconds=300, is_final_chunk=True
        )
        assert is_valid is True

    def test_sub_one_minute_final_chunk_skips_length_check(self):
        """Final chunk under 1 minute should skip the length check entirely."""
        transcript = "This is a short final chunk."  # 28 chars, enough to pass near-empty
        is_valid, reason = validate_transcript_quality(
            transcript, chunk_duration_seconds=30, is_final_chunk=True
        )
        assert is_valid is True

    def test_hallucinated_repeated_sentences_detected(self):
        """Transcript with many repeated sentences should be flagged."""
        base_sentence = "The quick brown fox jumps over the lazy dog and runs across the field"
        # 15 repetitions out of 17 total => ratio > 0.4 and count >= 4
        # Also long enough to pass the 500-char threshold for 10 min
        sentences = [base_sentence] * 15 + [
            "Some other thing happens here in the story",
            "And another completely different thing occurs",
        ]
        transcript = ". ".join(sentences) + "."
        assert len(transcript) >= 500
        is_valid, reason = validate_transcript_quality(transcript, chunk_duration_seconds=600)
        assert is_valid is False
        assert "Repeated sentence" in reason

    def test_natural_repetition_not_flagged(self):
        """A few repeats among many unique sentences should NOT be flagged."""
        unique_sentences = [f"This is unique sentence number {i} in the transcript" for i in range(20)]
        repeated = "A sentence that appears a few times"
        # 3 repeats out of 23 total sentences => count < 4, should pass
        sentences = unique_sentences + [repeated] * 3
        transcript = ". ".join(sentences) + "."
        is_valid, reason = validate_transcript_quality(transcript, chunk_duration_seconds=600)
        assert is_valid is True

    def test_cjk_valid_transcript_passes(self):
        """A valid CJK transcript should pass (using proportional chunk duration)."""
        transcript = (
            "大家好，歡迎來到今天的節目。今天我們要討論科技的發展和未來趨勢。"
            "讓我們先從人工智能開始說起，這個領域最近有很多突破。"
            "我們的嘉賓今天會分享他們的觀點和見解。"
            "首先，讓我們來看看最新的研究成果和應用案例。"
            "這些技術將會改變我們的生活方式和工作方式。"
        )
        # CJK characters are dense — use a chunk duration proportional to the content
        # ~150 CJK chars for 2 minutes (threshold = 100) is realistic
        is_valid, reason = validate_transcript_quality(transcript, chunk_duration_seconds=120)
        assert is_valid is True

    def test_cjk_repeated_pattern_detected(self):
        """CJK text with repeated single character should be flagged."""
        # Single character repeated — every 20-char sliding window is identical
        transcript = "啊" * 600
        is_valid, reason = validate_transcript_quality(transcript, chunk_duration_seconds=600)
        assert is_valid is False
        assert "Repeated character pattern" in reason

    def test_short_filler_phrases_not_counted_for_repetition(self):
        """Short filler phrases like 'Yeah', 'OK' should not trigger sentence repetition."""
        fillers = ["Yeah", "OK", "Right", "Uh huh", "Sure", "Mm hmm"]
        unique_sentences = [f"This is an important point about topic number {i}" for i in range(15)]
        # Add many short filler phrases — they are <=10 chars so filtered out
        all_parts = unique_sentences + fillers * 5
        transcript = ". ".join(all_parts) + "."
        is_valid, reason = validate_transcript_quality(transcript, chunk_duration_seconds=600)
        assert is_valid is True


class TestTranscribeAudioChunkedValidation:
    """Tests for validation integration in transcribe_audio_chunked."""

    @patch("src.core.transcriber.transcribe_audio")
    @patch("src.core.transcriber.get_audio_duration")
    @patch("src.core.transcriber.split_audio")
    def test_validation_failure_triggers_retry(
        self, mock_split, mock_duration, mock_transcribe, tmp_path
    ):
        """When first transcription fails validation, it should retry and use the good result."""
        audio_path = tmp_path / "test.mp3"
        audio_path.write_bytes(b"fake audio")

        mock_duration.return_value = 1200.0  # 20 minutes -> will be chunked

        chunk1 = tmp_path / "chunk_000.mp3"
        chunk2 = tmp_path / "chunk_001.mp3"
        chunk1.write_bytes(b"chunk1")
        chunk2.write_bytes(b"chunk2")
        mock_split.return_value = [chunk1, chunk2]

        short_transcript = "Too short"
        good_transcript = (
            "Welcome to the show everyone. Today we are going to discuss many interesting topics. "
            "Our first guest has a lot of experience in this field. Let's hear what they have to say. "
            "Thank you for joining us today, it is a pleasure to have you here. "
            "I have been working in this industry for over twenty years now. "
            "That is a very impressive career, can you tell us more about your journey. "
            "Of course, I started out as a junior developer and worked my way up. "
            "Along the way I learned many valuable lessons about teamwork and leadership."
        )

        # chunk1: first call returns short, second returns good
        # chunk2: returns good immediately
        mock_transcribe.side_effect = [
            short_transcript,  # chunk1, attempt 1 — fails validation
            good_transcript,   # chunk1, attempt 2 — passes validation
            good_transcript,   # chunk2, attempt 1 — passes validation
        ]

        result = transcribe_audio_chunked(
            audio_path=audio_path,
            api_key="test-key",
            chunk_duration_minutes=10,
            delay_between_chunks=0,
            show_progress=False,
        )

        # Should have called transcribe_audio 3 times (2 for chunk1, 1 for chunk2)
        assert mock_transcribe.call_count == 3
        # The good transcript should be in the result, not the short one
        assert good_transcript in result
        assert short_transcript not in result
