"""Generate video with waveform visualization for podcast episodes using ffmpeg."""

import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


class VideoGenerationError(Exception):
    """Raised when video generation fails."""
    pass


def get_audio_duration(audio_path: Path, ffprobe_path: str = "ffprobe") -> float:
    """
    Get the duration of an audio file in seconds.

    Args:
        audio_path: Path to audio file
        ffprobe_path: Path to ffprobe executable

    Returns:
        Duration in seconds

    Raises:
        VideoGenerationError: If duration cannot be determined
    """
    try:
        result = subprocess.run(
            [
                ffprobe_path,
                "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())
    except Exception as e:
        raise VideoGenerationError(f"Failed to get audio duration: {e}")


def create_base_image(
    width: int,
    height: int,
    background_path: Optional[Path] = None,
    icon_path: Optional[Path] = None,
    title: str = "",
    bg_color: Tuple[int, int, int] = (20, 20, 30),
    icon_size: Tuple[int, int] = (200, 200),
    icon_y_position: int = 150,
    title_font_size: int = 48,
    title_color: Tuple[int, int, int] = (255, 255, 255),
    title_y_position: Optional[int] = None,
) -> Image.Image:
    """
    Create base image with background, icon, and title.

    Args:
        width: Image width
        height: Image height
        background_path: Optional path to background image
        icon_path: Optional path to icon/logo image
        title: Title text to display
        bg_color: Fallback background color (RGB)
        icon_size: Size to scale icon to (width, height)
        icon_y_position: Y position for icon (centered horizontally)
        title_font_size: Font size for title
        title_color: Color for title text (RGB)
        title_y_position: Y position for title (default: near bottom)

    Returns:
        PIL Image with all static elements composited
    """
    # Create or load background
    if background_path and Path(background_path).exists():
        bg = Image.open(background_path).convert('RGBA')
        bg = bg.resize((width, height), Image.Resampling.LANCZOS)
    else:
        bg = Image.new('RGBA', (width, height), (*bg_color, 255))

    # Add icon if provided
    if icon_path and Path(icon_path).exists():
        icon = Image.open(icon_path).convert('RGBA')
        icon = icon.resize(icon_size, Image.Resampling.LANCZOS)
        icon_x = (width - icon_size[0]) // 2
        bg.paste(icon, (icon_x, icon_y_position), icon)

    # Add title if provided
    if title:
        draw = ImageDraw.Draw(bg)

        # Try to load a system font with Chinese support
        font = None
        font_paths = [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, title_font_size)
                break
            except (OSError, IOError):
                continue

        if font is None:
            font = ImageFont.load_default()

        # Calculate title position
        if title_y_position is None:
            title_y_position = height - 150

        # Word wrap for title
        max_width = width - 200
        lines = []
        current_line = ""

        for char in title:
            test_line = current_line + char
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if bbox[2] - bbox[0] <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = char

        if current_line:
            lines.append(current_line)

        # Draw each line centered
        y = title_y_position
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_width = bbox[2] - bbox[0]
            x = (width - line_width) // 2
            draw.text((x, y), line, font=font, fill=title_color)
            y += bbox[3] - bbox[1] + 10

    return bg


def rgb_to_hex(color: Tuple[int, int, int]) -> str:
    """Convert RGB tuple to hex string for ffmpeg."""
    return f"0x{color[0]:02X}{color[1]:02X}{color[2]:02X}"


def generate_video(
    audio_path: Path,
    output_path: Path,
    title: str = "",
    background_path: Optional[Path] = None,
    icon_path: Optional[Path] = None,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    waveform_color: Tuple[int, int, int] = (0, 200, 255),
    waveform_width: int = 960,
    waveform_height: int = 200,
    bg_color: Tuple[int, int, int] = (20, 20, 30),
    icon_size: Tuple[int, int] = (200, 200),
    show_progress: bool = True,
    ffmpeg_path: str = "ffmpeg",
    ffprobe_path: str = "ffprobe",
    # Legacy parameters (kept for backward compatibility)
    num_bars: int = 64,
    bar_color: Optional[Tuple[int, int, int]] = None,
) -> Path:
    """
    Generate a video with waveform visualization using ffmpeg.

    Args:
        audio_path: Path to audio file
        output_path: Path for output video
        title: Title text to display
        background_path: Optional background image
        icon_path: Optional icon/logo image
        width: Video width
        height: Video height
        fps: Frames per second
        waveform_color: RGB color for waveform
        waveform_width: Width of waveform visualization
        waveform_height: Height of waveform visualization
        bg_color: RGB background color (if no background image)
        icon_size: Size for icon/logo
        show_progress: Whether to show progress
        ffmpeg_path: Path to ffmpeg executable
        ffprobe_path: Path to ffprobe executable
        num_bars: Deprecated, kept for compatibility
        bar_color: Deprecated, use waveform_color instead

    Returns:
        Path to generated video

    Raises:
        VideoGenerationError: If video generation fails
    """
    audio_path = Path(audio_path)
    output_path = Path(output_path)

    # Handle legacy bar_color parameter
    if bar_color is not None:
        waveform_color = bar_color

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Get audio duration
        if show_progress:
            print("Getting audio duration...")
        duration = get_audio_duration(audio_path, ffprobe_path)

        if show_progress:
            print(f"Audio duration: {duration:.1f} seconds")

        # Create base image with background, icon, and title
        if show_progress:
            print("Creating base frame...")

        base_image = create_base_image(
            width=width,
            height=height,
            background_path=background_path,
            icon_path=icon_path,
            title=title,
            bg_color=bg_color,
            icon_size=icon_size,
        )

        # Save base image to temp file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            base_image_path = Path(tmp.name)

        base_image.save(base_image_path, 'PNG')

        if show_progress:
            print("Generating video with ffmpeg...")

        # Build ffmpeg filter complex
        waveform_color_hex = rgb_to_hex(waveform_color)

        # Calculate waveform position (centered, slightly below middle)
        wave_x = (width - waveform_width) // 2
        wave_y = (height - waveform_height) // 2 + 50

        filter_complex = (
            f"[1:v]loop=loop=-1:size=1:start=0,trim=duration={duration},fps={fps}[bg];"
            f"[0:a]showwaves=s={waveform_width}x{waveform_height}:mode=cline:rate={fps}:"
            f"colors={waveform_color_hex}:scale=cbrt:draw=full[wave];"
            f"[bg][wave]overlay={wave_x}:{wave_y}[out]"
        )

        # Build ffmpeg command
        cmd = [
            ffmpeg_path, "-y",
            "-i", str(audio_path),
            "-i", str(base_image_path),
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-map", "0:a",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            str(output_path),
        ]

        # Run ffmpeg
        if show_progress:
            print(f"Rendering video to {output_path}...")
            # Run with output visible
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )
        else:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )

        # Clean up temp file
        base_image_path.unlink(missing_ok=True)

        if result.returncode != 0:
            raise VideoGenerationError(f"ffmpeg failed: {result.stderr}")

        if show_progress:
            print(f"Video saved to: {output_path}")

        return output_path

    except VideoGenerationError:
        raise
    except Exception as e:
        raise VideoGenerationError(f"Failed to generate video: {e}")
