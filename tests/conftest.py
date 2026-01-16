"""Shared pytest fixtures for all tests."""

import os
import pytest
from pathlib import Path


# Paths
FIXTURES_DIR = Path(__file__).parent / "fixtures"
PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the path to the fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture
def sample_video(fixtures_dir) -> Path:
    """Return path to sample video file.

    Place a sample MP4 file at: tests/fixtures/sample_video.mp4
    """
    path = fixtures_dir / "sample_video.mp4"
    if not path.exists():
        pytest.skip(f"Sample video not found: {path}")
    return path


@pytest.fixture
def sample_audio(fixtures_dir) -> Path:
    """Return path to sample audio file.

    Place a sample MP3 file at: tests/fixtures/sample_audio.mp3
    """
    path = fixtures_dir / "sample_audio.mp3"
    if not path.exists():
        pytest.skip(f"Sample audio not found: {path}")
    return path


@pytest.fixture
def sample_transcript(fixtures_dir) -> Path:
    """Return path to sample transcript file.

    Place a sample transcript at: tests/fixtures/sample_transcript.txt
    """
    path = fixtures_dir / "sample_transcript.txt"
    if not path.exists():
        pytest.skip(f"Sample transcript not found: {path}")
    return path


@pytest.fixture
def temp_output_dir(tmp_path) -> Path:
    """Return a temporary output directory for test outputs."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def test_config() -> dict:
    """Return a test configuration dictionary."""
    return {
        "mode": "local",
        "local": {
            "output_dir": "./output",
            "ffmpeg_path": "ffmpeg",
        },
        "templates": {
            "background": "templates/background.png",
            "icon": "templates/icon.png",
            "font": "",
        },
        "video": {
            "resolution": [1920, 1080],
            "fps": 30,
            "waveform_color": "#00FF88",
            "waveform_height": 200,
            "text_color": "#FFFFFF",
        },
        "transcription": {
            "language": "en",
            "include_timestamps": False,
        },
        "summary": {
            "youtube_max_length": 5000,
            "spotify_max_length": 4000,
            "generate_tags": True,
            "max_tags": 10,
        },
        "youtube": {
            "default_privacy": "private",
            "category_id": "22",
            "default_language": "en",
        },
    }


@pytest.fixture
def mock_gemini_api_key() -> str:
    """Return a mock Gemini API key for testing."""
    return os.environ.get("GEMINI_API_KEY", "test-api-key-for-testing")
