"""Unit tests for video_generator module — bitrate and encoding settings."""

import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock, ANY

import pytest

from src.core.video_generator import generate_video, VideoGenerationError


def _get_ffmpeg_cmd(mock_run):
    """Extract the ffmpeg command from mock subprocess.run calls.

    The function calls subprocess.run twice: once for ffprobe (duration),
    once for ffmpeg (encoding). Return the ffmpeg call's command list.
    """
    for call in mock_run.call_args_list:
        cmd = call[0][0] if call[0] else call[1].get("args", [])
        if cmd and cmd[0] != "ffprobe" and "ffprobe" not in str(cmd[0]):
            return cmd
    raise AssertionError("No ffmpeg call found in subprocess.run calls")


@pytest.fixture
def video_setup(tmp_path):
    """Create common test fixtures for video generation tests."""
    audio_path = tmp_path / "test.mp3"
    audio_path.write_bytes(b"fake audio")
    output_path = tmp_path / "output.mp4"
    return audio_path, output_path


class TestVideoBitrate:
    """Tests for video bitrate settings and overrides."""

    @patch("src.core.video_generator.subprocess.run")
    @patch("src.core.video_generator.create_base_image")
    @patch("src.core.video_generator.get_audio_duration")
    @patch("src.core.video_generator.load_video_config")
    def test_videotoolbox_default_fast(
        self, mock_config, mock_duration, mock_image, mock_run, video_setup
    ):
        """VideoToolbox fast mode should use 2M bitrate."""
        audio_path, output_path = video_setup
        mock_config.return_value = {}
        mock_duration.return_value = 60.0
        mock_image.return_value = MagicMock()
        mock_run.return_value = MagicMock(returncode=0)

        generate_video(
            audio_path=audio_path,
            output_path=output_path,
            encoder="videotoolbox",
            compact=False,
            show_progress=False,
        )

        cmd = _get_ffmpeg_cmd(mock_run)
        assert "-b:v" in cmd
        idx = cmd.index("-b:v")
        assert cmd[idx + 1] == "2M"

    @patch("src.core.video_generator.subprocess.run")
    @patch("src.core.video_generator.create_base_image")
    @patch("src.core.video_generator.get_audio_duration")
    @patch("src.core.video_generator.load_video_config")
    def test_videotoolbox_default_compact(
        self, mock_config, mock_duration, mock_image, mock_run, video_setup
    ):
        """VideoToolbox compact mode should use 1M bitrate."""
        audio_path, output_path = video_setup
        mock_config.return_value = {}
        mock_duration.return_value = 60.0
        mock_image.return_value = MagicMock()
        mock_run.return_value = MagicMock(returncode=0)

        generate_video(
            audio_path=audio_path,
            output_path=output_path,
            encoder="videotoolbox",
            compact=True,
            show_progress=False,
        )

        cmd = _get_ffmpeg_cmd(mock_run)
        assert "-b:v" in cmd
        idx = cmd.index("-b:v")
        assert cmd[idx + 1] == "1M"

    @patch("src.core.video_generator.subprocess.run")
    @patch("src.core.video_generator.create_base_image")
    @patch("src.core.video_generator.get_audio_duration")
    @patch("src.core.video_generator.load_video_config")
    def test_video_bitrate_override_videotoolbox(
        self, mock_config, mock_duration, mock_image, mock_run, video_setup
    ):
        """--video-bitrate should override VideoToolbox default."""
        audio_path, output_path = video_setup
        mock_config.return_value = {}
        mock_duration.return_value = 60.0
        mock_image.return_value = MagicMock()
        mock_run.return_value = MagicMock(returncode=0)

        generate_video(
            audio_path=audio_path,
            output_path=output_path,
            encoder="videotoolbox",
            video_bitrate="3M",
            show_progress=False,
        )

        cmd = _get_ffmpeg_cmd(mock_run)
        assert "-b:v" in cmd
        idx = cmd.index("-b:v")
        assert cmd[idx + 1] == "3M"

    @patch("src.core.video_generator.subprocess.run")
    @patch("src.core.video_generator.create_base_image")
    @patch("src.core.video_generator.get_audio_duration")
    @patch("src.core.video_generator.load_video_config")
    def test_video_bitrate_override_cpu(
        self, mock_config, mock_duration, mock_image, mock_run, video_setup
    ):
        """--video-bitrate with CPU encoder should use -b:v instead of -crf, but keep -preset."""
        audio_path, output_path = video_setup
        mock_config.return_value = {}
        mock_duration.return_value = 60.0
        mock_image.return_value = MagicMock()
        mock_run.return_value = MagicMock(returncode=0)

        generate_video(
            audio_path=audio_path,
            output_path=output_path,
            encoder="cpu",
            video_bitrate="1500k",
            show_progress=False,
        )

        cmd = _get_ffmpeg_cmd(mock_run)
        # Should have -b:v with the override value
        assert "-b:v" in cmd
        idx = cmd.index("-b:v")
        assert cmd[idx + 1] == "1500k"
        # Should NOT have -crf (bitrate overrides CRF)
        assert "-crf" not in cmd
        # Should still have -preset for encoding speed
        assert "-preset" in cmd

    @patch("src.core.video_generator.subprocess.run")
    @patch("src.core.video_generator.create_base_image")
    @patch("src.core.video_generator.get_audio_duration")
    @patch("src.core.video_generator.load_video_config")
    def test_cpu_default_unchanged(
        self, mock_config, mock_duration, mock_image, mock_run, video_setup
    ):
        """CPU encoder without --video-bitrate should use CRF as before."""
        audio_path, output_path = video_setup
        mock_config.return_value = {}
        mock_duration.return_value = 60.0
        mock_image.return_value = MagicMock()
        mock_run.return_value = MagicMock(returncode=0)

        generate_video(
            audio_path=audio_path,
            output_path=output_path,
            encoder="cpu",
            compact=False,
            show_progress=False,
        )

        cmd = _get_ffmpeg_cmd(mock_run)
        assert "-crf" in cmd
        idx = cmd.index("-crf")
        assert cmd[idx + 1] == "23"
        assert "-preset" in cmd
        pidx = cmd.index("-preset")
        assert cmd[pidx + 1] == "fast"
        # Should NOT have -b:v
        assert "-b:v" not in cmd

    @patch("src.core.video_generator.subprocess.run")
    @patch("src.core.video_generator.create_base_image")
    @patch("src.core.video_generator.get_audio_duration")
    @patch("src.core.video_generator.load_video_config")
    def test_video_bitrate_override_nvenc(
        self, mock_config, mock_duration, mock_image, mock_run, video_setup
    ):
        """--video-bitrate with NVENC should use -b:v instead of -rc vbr -cq."""
        audio_path, output_path = video_setup
        mock_config.return_value = {}
        mock_duration.return_value = 60.0
        mock_image.return_value = MagicMock()
        mock_run.return_value = MagicMock(returncode=0)

        generate_video(
            audio_path=audio_path,
            output_path=output_path,
            encoder="nvenc",
            video_bitrate="2M",
            show_progress=False,
        )

        cmd = _get_ffmpeg_cmd(mock_run)
        assert "-b:v" in cmd
        idx = cmd.index("-b:v")
        assert cmd[idx + 1] == "2M"
        # Should NOT have the default NVENC quality flags
        assert "-rc" not in cmd
        assert "-cq" not in cmd
