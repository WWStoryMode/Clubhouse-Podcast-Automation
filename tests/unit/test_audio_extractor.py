"""Unit tests for audio_extractor module."""

import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.core.audio_extractor import (
    extract_audio,
    check_ffmpeg,
    get_audio_duration,
    AudioExtractionError,
)


class TestCheckFfmpeg:
    """Tests for check_ffmpeg function."""

    def test_ffmpeg_available(self):
        """Test that ffmpeg is detected when available."""
        # This test requires ffmpeg to be installed
        result = check_ffmpeg()
        # Don't fail if ffmpeg isn't installed, just skip
        if not result:
            pytest.skip("ffmpeg not installed on this system")
        assert result is True

    def test_ffmpeg_not_found(self):
        """Test that missing ffmpeg returns False."""
        result = check_ffmpeg("/nonexistent/path/to/ffmpeg")
        assert result is False

    @patch("subprocess.run")
    def test_ffmpeg_check_with_mock(self, mock_run):
        """Test ffmpeg check with mocked subprocess."""
        mock_run.return_value = MagicMock(returncode=0)
        assert check_ffmpeg() is True
        mock_run.assert_called_once()


class TestExtractAudio:
    """Tests for extract_audio function."""

    def test_extract_audio_file_not_found(self, temp_output_dir):
        """Test that FileNotFoundError is raised for missing input."""
        fake_path = temp_output_dir / "nonexistent.mp4"

        with pytest.raises(FileNotFoundError, match="Video file not found"):
            extract_audio(fake_path)

    def test_extract_audio_path_is_directory(self, temp_output_dir):
        """Test that FileNotFoundError is raised when path is a directory."""
        with pytest.raises(FileNotFoundError, match="Path is not a file"):
            extract_audio(temp_output_dir)

    @patch("src.core.audio_extractor.check_ffmpeg")
    def test_extract_audio_ffmpeg_not_available(
        self, mock_check, temp_output_dir
    ):
        """Test that AudioExtractionError is raised when ffmpeg is missing."""
        mock_check.return_value = False

        # Create a dummy video file
        video_path = temp_output_dir / "test.mp4"
        video_path.write_bytes(b"fake video content")

        with pytest.raises(AudioExtractionError, match="ffmpeg not found"):
            extract_audio(video_path)

    @patch("src.core.audio_extractor.check_ffmpeg")
    def test_extract_audio_output_exists_no_overwrite(
        self, mock_check, temp_output_dir
    ):
        """Test that error is raised when output exists and overwrite=False."""
        mock_check.return_value = True

        # Create dummy input and output files
        video_path = temp_output_dir / "test.mp4"
        video_path.write_bytes(b"fake video content")
        output_path = temp_output_dir / "test.mp3"
        output_path.write_bytes(b"existing audio")

        with pytest.raises(AudioExtractionError, match="already exists"):
            extract_audio(video_path, output_path, overwrite=False)

    @patch("subprocess.run")
    @patch("src.core.audio_extractor.check_ffmpeg")
    def test_extract_audio_ffmpeg_failure(
        self, mock_check, mock_run, temp_output_dir
    ):
        """Test handling of ffmpeg extraction failure."""
        mock_check.return_value = True
        mock_run.return_value = MagicMock(
            returncode=1,
            stderr="Error: invalid input file"
        )

        video_path = temp_output_dir / "test.mp4"
        video_path.write_bytes(b"fake video content")

        with pytest.raises(AudioExtractionError, match="ffmpeg extraction failed"):
            extract_audio(video_path)

    @patch("subprocess.run")
    @patch("src.core.audio_extractor.check_ffmpeg")
    def test_extract_audio_timeout(
        self, mock_check, mock_run, temp_output_dir
    ):
        """Test handling of ffmpeg timeout."""
        mock_check.return_value = True
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ffmpeg", timeout=3600)

        video_path = temp_output_dir / "test.mp4"
        video_path.write_bytes(b"fake video content")

        with pytest.raises(AudioExtractionError, match="timed out"):
            extract_audio(video_path)

    def test_extract_audio_with_real_file(self, sample_video, temp_output_dir):
        """Test actual audio extraction with a real video file.

        This test requires:
        1. ffmpeg to be installed
        2. A sample video at tests/fixtures/sample_video.mp4
        """
        if not check_ffmpeg():
            pytest.skip("ffmpeg not installed")

        output_path = temp_output_dir / "extracted_audio.mp3"

        result = extract_audio(
            sample_video,
            output_path,
            overwrite=True,
        )

        assert result == output_path
        assert result.exists()
        assert result.stat().st_size > 0

    def test_extract_audio_default_output_path(self, sample_video, temp_output_dir):
        """Test that default output path uses .mp3 extension."""
        if not check_ffmpeg():
            pytest.skip("ffmpeg not installed")

        # Copy sample to temp dir to avoid polluting fixtures
        import shutil
        temp_video = temp_output_dir / "test_video.mp4"
        shutil.copy(sample_video, temp_video)

        result = extract_audio(temp_video, overwrite=True)

        expected_output = temp_video.with_suffix(".mp3")
        assert result == expected_output
        assert result.exists()


class TestGetAudioDuration:
    """Tests for get_audio_duration function."""

    def test_get_duration_with_real_file(self, sample_audio):
        """Test getting duration from a real audio file."""
        if not check_ffmpeg():
            pytest.skip("ffmpeg not installed")

        duration = get_audio_duration(sample_audio)

        assert isinstance(duration, float)
        assert duration > 0

    @patch("subprocess.run")
    def test_get_duration_ffprobe_failure(self, mock_run, temp_output_dir):
        """Test handling of ffprobe failure."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stderr="Error: could not find stream"
        )

        audio_path = temp_output_dir / "test.mp3"
        audio_path.write_bytes(b"fake audio")

        with pytest.raises(AudioExtractionError, match="ffprobe failed"):
            get_audio_duration(audio_path)

    @patch("subprocess.run")
    def test_get_duration_invalid_output(self, mock_run, temp_output_dir):
        """Test handling of invalid ffprobe output."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="not a number"
        )

        audio_path = temp_output_dir / "test.mp3"
        audio_path.write_bytes(b"fake audio")

        with pytest.raises(AudioExtractionError, match="Failed to get audio duration"):
            get_audio_duration(audio_path)
