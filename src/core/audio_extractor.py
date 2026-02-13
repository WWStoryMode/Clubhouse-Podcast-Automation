"""Extract audio tracks from video files using ffmpeg."""

import subprocess
import shutil
from pathlib import Path
from typing import Optional


class AudioExtractionError(Exception):
    """Raised when audio extraction fails."""

    pass


def check_ffmpeg(ffmpeg_path: str = "ffmpeg") -> bool:
    """
    Check if ffmpeg is available.

    Args:
        ffmpeg_path: Path to ffmpeg binary

    Returns:
        True if ffmpeg is available, False otherwise
    """
    try:
        result = subprocess.run(
            [ffmpeg_path, "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def get_audio_duration(audio_path: Path, ffmpeg_path: str = "ffmpeg") -> float:
    """
    Get the duration of an audio file in seconds.

    Args:
        audio_path: Path to audio file
        ffmpeg_path: Path to ffmpeg binary

    Returns:
        Duration in seconds

    Raises:
        AudioExtractionError: If duration cannot be determined
    """
    ffprobe_path = ffmpeg_path.replace("ffmpeg", "ffprobe")

    try:
        result = subprocess.run(
            [
                ffprobe_path,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            raise AudioExtractionError(f"ffprobe failed: {result.stderr}")

        return float(result.stdout.strip())

    except (subprocess.SubprocessError, ValueError) as e:
        raise AudioExtractionError(f"Failed to get audio duration: {e}")


def parse_timestamp(timestamp: str) -> str:
    """
    Parse and validate a timestamp string.

    Accepts formats:
    - HH:MM:SS (e.g., "01:30:00")
    - MM:SS (e.g., "05:30")
    - SS (e.g., "90" for 90 seconds)

    Args:
        timestamp: Timestamp string

    Returns:
        Normalized timestamp string for ffmpeg

    Raises:
        ValueError: If timestamp format is invalid
    """
    if not timestamp:
        raise ValueError("Empty timestamp")

    parts = timestamp.split(":")
    if len(parts) == 1:
        # Just seconds
        seconds = float(parts[0])
        return str(seconds)
    elif len(parts) == 2:
        # MM:SS
        minutes, seconds = int(parts[0]), float(parts[1])
        return f"{minutes}:{seconds:05.2f}"
    elif len(parts) == 3:
        # HH:MM:SS
        hours, minutes, seconds = int(parts[0]), int(parts[1]), float(parts[2])
        return f"{hours}:{minutes:02d}:{seconds:05.2f}"
    else:
        raise ValueError(f"Invalid timestamp format: {timestamp}")


def extract_audio(
    video_path: Path,
    output_path: Optional[Path] = None,
    ffmpeg_path: str = "ffmpeg",
    audio_codec: str = "libmp3lame",
    audio_quality: str = "2",
    overwrite: bool = False,
    trim_start: Optional[str] = None,
    trim_end: Optional[str] = None,
    normalize: bool = True,
) -> Path:
    """
    Extract audio track from video file to MP3 using ffmpeg.

    Args:
        video_path: Path to input video file (MP4, etc.)
        output_path: Optional output path for MP3 file.
                     Defaults to same directory as video with .mp3 extension.
        ffmpeg_path: Path to ffmpeg binary
        audio_codec: Audio codec to use (default: libmp3lame for MP3)
        audio_quality: Audio quality setting (0-9, lower is better)
        overwrite: Whether to overwrite existing output file
        trim_start: Start time for trimming (HH:MM:SS, MM:SS, or seconds)
        trim_end: End time for trimming (HH:MM:SS, MM:SS, or seconds)
        normalize: Apply loudnorm filter for consistent audio levels (default: True)

    Returns:
        Path to extracted MP3 file

    Raises:
        FileNotFoundError: If video file doesn't exist
        AudioExtractionError: If ffmpeg is not available or extraction fails
    """
    video_path = Path(video_path)

    # Validate input file
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not video_path.is_file():
        raise FileNotFoundError(f"Path is not a file: {video_path}")

    # Check ffmpeg availability
    if not check_ffmpeg(ffmpeg_path):
        raise AudioExtractionError(
            f"ffmpeg not found at '{ffmpeg_path}'. "
            "Please install ffmpeg: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)"
        )

    # Determine output path
    if output_path is None:
        output_path = video_path.with_suffix(".mp3")
    else:
        output_path = Path(output_path)

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if output already exists
    if output_path.exists() and not overwrite:
        raise AudioExtractionError(
            f"Output file already exists: {output_path}. "
            "Use overwrite=True to replace it."
        )

    # Build ffmpeg command
    cmd = [ffmpeg_path]

    # Input file
    cmd.extend(["-i", str(video_path)])

    # Add trim start (after input for accurate seeking)
    if trim_start:
        try:
            start_time = parse_timestamp(trim_start)
            cmd.extend(["-ss", start_time])
        except ValueError as e:
            raise AudioExtractionError(f"Invalid trim_start format: {e}")

    # Add trim end (after input)
    if trim_end:
        try:
            end_time = parse_timestamp(trim_end)
            cmd.extend(["-to", end_time])
        except ValueError as e:
            raise AudioExtractionError(f"Invalid trim_end format: {e}")

    # Add loudnorm filter for audio normalization
    if normalize:
        cmd.extend(["-af", "loudnorm=I=-16:TP=-3:LRA=11"])

    # Audio settings
    cmd.extend([
        "-vn",                       # No video
        "-acodec", audio_codec,      # Audio codec
        "-ar", "44100",              # Sample rate (required for loudnorm)
        "-q:a", audio_quality,       # Audio quality
        "-y" if overwrite else "-n", # Overwrite flag
        str(output_path),            # Output file
    ])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout for long videos
        )

        if result.returncode != 0:
            raise AudioExtractionError(
                f"ffmpeg extraction failed (code {result.returncode}): {result.stderr}"
            )

        # Verify output file was created
        if not output_path.exists():
            raise AudioExtractionError(
                f"ffmpeg completed but output file not found: {output_path}"
            )

        return output_path

    except subprocess.TimeoutExpired:
        raise AudioExtractionError("ffmpeg timed out after 1 hour")
    except subprocess.SubprocessError as e:
        raise AudioExtractionError(f"ffmpeg subprocess error: {e}")
