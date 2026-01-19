"""Generate video with waveform visualization for podcast episodes using ffmpeg."""

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import yaml
from PIL import Image, ImageDraw, ImageFont


class VideoGenerationError(Exception):
    """Raised when video generation fails."""
    pass


# Encoder configurations
ENCODERS: Dict[str, Dict] = {
    "cpu": {
        "codec": "libx264",
        "description": "CPU encoding (portable, consistent quality)",
        "extra_args": [],
    },
    "videotoolbox": {
        "codec": "h264_videotoolbox",
        "description": "macOS GPU acceleration",
        "extra_args": ["-allow_sw", "1"],  # Allow software fallback
    },
    "nvenc": {
        "codec": "h264_nvenc",
        "description": "NVIDIA GPU acceleration",
        "extra_args": [],
    },
    "vaapi": {
        "codec": "h264_vaapi",
        "description": "Linux VA-API (Intel/AMD) acceleration",
        "extra_args": ["-vaapi_device", "/dev/dri/renderD128"],
    },
    "qsv": {
        "codec": "h264_qsv",
        "description": "Intel Quick Sync Video acceleration",
        "extra_args": [],
    },
    "amf": {
        "codec": "h264_amf",
        "description": "AMD GPU acceleration",
        "extra_args": [],
    },
}


# Default layout configuration
DEFAULT_LAYOUT = {
    "logo": {
        "x": "center",
        "y": 150,
        "size": 200,
    },
    "title": {
        "x": "center",
        "y": "auto",
        "font_size": 48,
        "color": [255, 255, 255],
        "max_width": 1720,
    },
    "waveform": {
        "x": "center",
        "y": "center_offset",
        "width": 960,
        "height": 200,
        "color": [240, 240, 240],
    },
}


def load_video_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load video layout configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses default layout.

    Returns:
        Configuration dictionary
    """
    config = DEFAULT_LAYOUT.copy()

    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            user_config = yaml.safe_load(f)

        # Deep merge user config into defaults
        for section in ["logo", "title", "waveform"]:
            if section in user_config:
                if section not in config:
                    config[section] = {}
                config[section] = {**DEFAULT_LAYOUT.get(section, {}), **user_config[section]}

    return config


def resolve_position(value: Any, dimension: int, element_size: int = 0) -> int:
    """
    Resolve position value to pixels.

    Args:
        value: Position value ("center", "center_offset", "auto", or int)
        dimension: Total dimension (width or height)
        element_size: Size of element being positioned

    Returns:
        Position in pixels
    """
    if isinstance(value, int):
        return value
    if value == "center":
        return (dimension - element_size) // 2
    if value == "center_offset":
        return (dimension - element_size) // 2 + 50
    return 0  # Default fallback


def get_available_encoders(ffmpeg_path: str = "ffmpeg") -> List[str]:
    """
    Detect available hardware encoders on the system.

    Args:
        ffmpeg_path: Path to ffmpeg executable

    Returns:
        List of available encoder names
    """
    available = ["cpu"]  # CPU is always available

    try:
        result = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
        )
        encoder_output = result.stdout

        # Check for each hardware encoder
        encoder_checks = {
            "videotoolbox": "h264_videotoolbox",
            "nvenc": "h264_nvenc",
            "vaapi": "h264_vaapi",
            "qsv": "h264_qsv",
            "amf": "h264_amf",
        }

        for name, codec in encoder_checks.items():
            if codec in encoder_output:
                available.append(name)

    except Exception:
        pass  # If detection fails, just return cpu

    return available


def detect_best_encoder(ffmpeg_path: str = "ffmpeg") -> str:
    """
    Auto-detect the best available encoder for the current system.

    Priority order:
    1. videotoolbox (macOS)
    2. nvenc (NVIDIA GPU)
    3. qsv (Intel Quick Sync)
    4. vaapi (Linux VA-API)
    5. amf (AMD GPU)
    6. cpu (fallback)

    Args:
        ffmpeg_path: Path to ffmpeg executable

    Returns:
        Name of the best available encoder
    """
    available = get_available_encoders(ffmpeg_path)

    # Platform-specific priority
    if sys.platform == "darwin":
        # macOS: prefer VideoToolbox
        priority = ["videotoolbox", "cpu"]
    elif sys.platform == "linux":
        # Linux: prefer NVENC, then VAAPI, then QSV
        priority = ["nvenc", "vaapi", "qsv", "amf", "cpu"]
    else:
        # Windows: prefer NVENC, then QSV, then AMF
        priority = ["nvenc", "qsv", "amf", "cpu"]

    for encoder in priority:
        if encoder in available:
            return encoder

    return "cpu"


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
    layout_config: Optional[Dict[str, Any]] = None,
    # Legacy parameters (used if no config provided)
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
        layout_config: Layout configuration dict (overrides legacy params)
        icon_size: Size to scale icon to (width, height)
        icon_y_position: Y position for icon (centered horizontally)
        title_font_size: Font size for title
        title_color: Color for title text (RGB)
        title_y_position: Y position for title (default: below logo)

    Returns:
        PIL Image with all static elements composited
    """
    # Use config if provided, otherwise use legacy parameters
    if layout_config:
        logo_cfg = layout_config.get("logo", {})
        title_cfg = layout_config.get("title", {})

        logo_size = logo_cfg.get("size", 200)
        icon_size = (logo_size, logo_size)
        icon_x_cfg = logo_cfg.get("x", "center")
        icon_y_position = logo_cfg.get("y", 150)

        title_font_size = title_cfg.get("font_size", 48)
        title_color = tuple(title_cfg.get("color", [255, 255, 255]))
        title_x_cfg = title_cfg.get("x", "center")
        title_y_cfg = title_cfg.get("y", "auto")
        title_max_width = title_cfg.get("max_width", width - 200)
    else:
        icon_x_cfg = "center"
        title_x_cfg = "center"
        title_y_cfg = "auto" if title_y_position is None else title_y_position
        title_max_width = width - 200

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

        # Resolve X position
        if icon_x_cfg == "center":
            icon_x = (width - icon_size[0]) // 2
        else:
            icon_x = int(icon_x_cfg)

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

        # Calculate title Y position
        if title_y_cfg == "auto":
            # Position below logo with padding
            actual_title_y = icon_y_position + icon_size[1] + 40
        else:
            actual_title_y = int(title_y_cfg)

        # Word wrap for title
        lines = []
        current_line = ""

        for char in title:
            test_line = current_line + char
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if bbox[2] - bbox[0] <= title_max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = char

        if current_line:
            lines.append(current_line)

        # Draw each line
        y = actual_title_y
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_width = bbox[2] - bbox[0]

            # Resolve X position
            if title_x_cfg == "center":
                x = (width - line_width) // 2
            else:
                x = int(title_x_cfg)

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
    waveform_color: Tuple[int, int, int] = (240, 240, 240),
    waveform_width: int = 960,
    waveform_height: int = 200,
    bg_color: Tuple[int, int, int] = (20, 20, 30),
    icon_size: Tuple[int, int] = (200, 200),
    compact: bool = False,
    encoder: str = "auto",
    video_config_path: Optional[Path] = None,
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
        compact: Use compact encoding (smaller file, slower encoding)
        encoder: Encoder to use (auto, cpu, videotoolbox, nvenc, vaapi, qsv, amf)
        video_config_path: Path to video layout config YAML file
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
        # Load video config if provided
        layout_config = load_video_config(video_config_path)

        if video_config_path and show_progress:
            print(f"Using video config: {video_config_path}")

        # Get waveform settings from config or parameters
        waveform_cfg = layout_config.get("waveform", {})
        if video_config_path:
            waveform_width = waveform_cfg.get("width", waveform_width)
            waveform_height = waveform_cfg.get("height", waveform_height)
            waveform_color = tuple(waveform_cfg.get("color", list(waveform_color)))

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
            layout_config=layout_config if video_config_path else None,
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

        # Calculate waveform position from config or default
        wave_x_cfg = waveform_cfg.get("x", "center")
        wave_y_cfg = waveform_cfg.get("y", "center_offset")

        if wave_x_cfg == "center":
            wave_x = (width - waveform_width) // 2
        else:
            wave_x = int(wave_x_cfg)

        if wave_y_cfg == "center":
            wave_y = (height - waveform_height) // 2
        elif wave_y_cfg == "center_offset":
            wave_y = (height - waveform_height) // 2 + 50
        else:
            wave_y = int(wave_y_cfg)

        filter_complex = (
            f"[1:v]loop=loop=-1:size=1:start=0,trim=duration={duration},fps={fps}[bg];"
            f"[0:a]showwaves=s={waveform_width}x{waveform_height}:mode=cline:rate={fps}:"
            f"colors={waveform_color_hex}:scale=cbrt:draw=full[wave];"
            f"[bg][wave]overlay={wave_x}:{wave_y}[out]"
        )

        # Determine encoder
        if encoder == "auto":
            selected_encoder = detect_best_encoder(ffmpeg_path)
        else:
            selected_encoder = encoder

        if selected_encoder not in ENCODERS:
            raise VideoGenerationError(f"Unknown encoder: {selected_encoder}")

        encoder_config = ENCODERS[selected_encoder]

        if show_progress:
            print(f"Using encoder: {selected_encoder} ({encoder_config['description']})")

        # Encoding settings based on compact mode
        if compact:
            # Compact: smaller file, slower encoding
            audio_bitrate = "128k"
            if selected_encoder == "cpu":
                preset = "medium"
                crf = "28"
            else:
                # Hardware encoders use different quality settings
                preset = "default"
                crf = "28"
        else:
            # Fast: faster encoding, larger file
            audio_bitrate = "192k"
            if selected_encoder == "cpu":
                preset = "fast"
                crf = "23"
            else:
                preset = "default"
                crf = "23"

        # Build ffmpeg command
        cmd = [
            ffmpeg_path, "-y",
            "-i", str(audio_path),
            "-i", str(base_image_path),
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-map", "0:a",
            "-c:v", encoder_config["codec"],
        ]

        # Add encoder-specific arguments
        cmd.extend(encoder_config["extra_args"])

        # Add quality settings (different for CPU vs hardware)
        if selected_encoder == "cpu":
            cmd.extend(["-preset", preset, "-crf", crf])
        elif selected_encoder == "videotoolbox":
            # VideoToolbox uses bitrate mode
            bitrate = "2M" if compact else "4M"
            cmd.extend(["-b:v", bitrate])
        elif selected_encoder == "nvenc":
            # NVENC uses -cq for constant quality mode
            cmd.extend(["-rc", "vbr", "-cq", crf, "-preset", "p4" if compact else "p1"])
        elif selected_encoder in ("vaapi", "qsv", "amf"):
            # Use global quality for other hardware encoders
            cmd.extend(["-global_quality", crf])

        # Add audio settings
        cmd.extend([
            "-c:a", "aac",
            "-b:a", audio_bitrate,
            "-shortest",
            str(output_path),
        ])

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
