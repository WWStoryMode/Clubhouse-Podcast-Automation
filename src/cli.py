"""Command-line interface for Clubhouse-Podcast-Automation."""

import os
import sys
from pathlib import Path
from typing import Optional

import click
import yaml
from dotenv import load_dotenv

from .core.downloader import download_clubhouse_video, DownloadError
from .core.audio_extractor import extract_audio, AudioExtractionError
from .core.transcriber import transcribe_audio, transcribe_audio_chunked, TranscriptionError
from .core.summarizer import generate_descriptions, SummaryError
from .core.video_generator import generate_video, generate_preview_frame, VideoGenerationError, get_available_encoders


# Load environment variables
load_dotenv()


def load_config(config_path: Optional[Path] = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path("config/config.yaml")

    if not config_path.exists():
        # Return defaults if no config file
        return {
            "mode": "local",
            "local": {
                "output_dir": "./output",
                "ffmpeg_path": "ffmpeg",
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
        }

    with open(config_path) as f:
        return yaml.safe_load(f)


@click.group()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to config file",
)
@click.pass_context
def cli(ctx, config):
    """Clubhouse-Podcast-Automation CLI.

    Automate publishing Clubhouse recordings to Spotify and YouTube.
    """
    ctx.ensure_object(dict)
    config_path = Path(config) if config else None
    ctx.obj["config"] = load_config(config_path)


@cli.command()
@click.option("--url", "-u", required=True, help="Clubhouse recording URL")
@click.option("--output", "-o", type=click.Path(), help="Output directory")
@click.option("--filename", "-f", help="Custom filename (without extension)")
@click.pass_context
def download(ctx, url, output, filename):
    """Download a Clubhouse recording."""
    config = ctx.obj["config"]
    output_dir = Path(output) if output else Path(config["local"]["output_dir"]) / "audio"

    click.echo(f"Downloading from: {url}")

    try:
        result = download_clubhouse_video(
            url=url,
            output_dir=output_dir,
            filename=filename,
        )
        click.echo(f"Downloaded to: {result}")
    except (ValueError, DownloadError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True), help="Input video file")
@click.option("--output", "-o", type=click.Path(), help="Output audio file")
@click.pass_context
def extract(ctx, input_path, output):
    """Extract audio from video file."""
    config = ctx.obj["config"]
    ffmpeg_path = config["local"].get("ffmpeg_path", "ffmpeg")

    click.echo(f"Extracting audio from: {input_path}")

    try:
        output_path = Path(output) if output else None
        result = extract_audio(
            video_path=Path(input_path),
            output_path=output_path,
            ffmpeg_path=ffmpeg_path,
            overwrite=True,
        )
        click.echo(f"Extracted to: {result}")
    except (FileNotFoundError, AudioExtractionError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True), help="Input audio file")
@click.option("--output", "-o", type=click.Path(), help="Output transcript file")
@click.option("--language", "-l", default="en", help="Language code: en, yue, zh, zh-HK, zh-TW, ja, ko, etc. (default: en)")
@click.option("--timestamps", "-t", is_flag=True, help="Include timestamps in transcript")
@click.option("--chunked", is_flag=True, help="Use chunked transcription for long audio files")
@click.option("--chunk-minutes", default=10, type=int, help="Chunk duration in minutes (default: 10)")
@click.option("--model", "-m", default="gemini-2.5-flash", help="Gemini model to use (default: gemini-2.5-flash)")
@click.pass_context
def transcribe(ctx, input_path, output, language, timestamps, chunked, chunk_minutes, model):
    """Transcribe audio file using Gemini API."""
    config = ctx.obj["config"]
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        click.echo("Error: GEMINI_API_KEY environment variable not set", err=True)
        sys.exit(1)

    click.echo(f"Transcribing: {input_path}")
    click.echo(f"Using model: {model}")

    try:
        if chunked:
            click.echo(f"Using chunked transcription ({chunk_minutes} min chunks)...")
            transcript = transcribe_audio_chunked(
                audio_path=Path(input_path),
                api_key=api_key,
                language=language,
                chunk_duration_minutes=chunk_minutes,
                include_timestamps=timestamps,
                model_name=model,
                show_progress=True,
            )
        else:
            transcript = transcribe_audio(
                audio_path=Path(input_path),
                api_key=api_key,
                language=language,
                include_timestamps=timestamps or config.get("transcription", {}).get("include_timestamps", False),
                model_name=model,
            )

        # Save transcript
        if output:
            output_path = Path(output)
        else:
            output_dir = Path(config["local"]["output_dir"]) / "transcripts"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{Path(input_path).stem}_transcript.txt"

        output_path.write_text(transcript)
        click.echo(f"Transcript saved to: {output_path}")

    except (FileNotFoundError, TranscriptionError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True), help="Input transcript file")
@click.option("--title", "-t", required=True, help="Episode title")
@click.option("--output", "-o", type=click.Path(), help="Output directory for descriptions")
@click.pass_context
def summarize(ctx, input_path, title, output):
    """Generate descriptions from transcript using Gemini API."""
    config = ctx.obj["config"]
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        click.echo("Error: GEMINI_API_KEY environment variable not set", err=True)
        sys.exit(1)

    click.echo(f"Generating descriptions for: {title}")

    try:
        transcript = Path(input_path).read_text()
        summary_config = config.get("summary", {})

        descriptions = generate_descriptions(
            transcript=transcript,
            episode_title=title,
            api_key=api_key,
            youtube_max_length=summary_config.get("youtube_max_length", 5000),
            spotify_max_length=summary_config.get("spotify_max_length", 4000),
            generate_tags=summary_config.get("generate_tags", True),
            max_tags=summary_config.get("max_tags", 10),
        )

        # Save descriptions
        if output:
            output_dir = Path(output)
        else:
            output_dir = Path(config["local"]["output_dir"]) / "descriptions"

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save as YAML for easy reading
        output_path = output_dir / f"{Path(input_path).stem}_descriptions.yaml"
        with open(output_path, "w") as f:
            yaml.dump(descriptions, f, default_flow_style=False, allow_unicode=True)

        click.echo(f"Descriptions saved to: {output_path}")
        click.echo(f"\nYouTube Title: {descriptions['youtube_title']}")
        click.echo(f"Tags: {', '.join(descriptions['tags'])}")

    except (FileNotFoundError, SummaryError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def parse_color(color_str: str) -> tuple:
    """Parse color string in format 'R,G,B' to tuple."""
    try:
        parts = [int(x.strip()) for x in color_str.split(",")]
        if len(parts) != 3:
            raise ValueError("Color must have 3 components")
        for p in parts:
            if not 0 <= p <= 255:
                raise ValueError("Color values must be 0-255")
        return tuple(parts)
    except Exception as e:
        raise click.BadParameter(f"Invalid color format. Use 'R,G,B' (e.g., '0,200,255'): {e}")


@cli.command("generate-video")
@click.option("--input", "-i", "input_path", type=click.Path(exists=True), help="Input audio file")
@click.option("--output", "-o", type=click.Path(), help="Output video file")
@click.option("--title", "-t", default="", help="Title text to display on video")
@click.option("--background", "-bg", type=click.Path(exists=True), help="Background image (default: templates/background.png)")
@click.option("--icon", type=click.Path(exists=True), help="Logo image (default: templates/logo.png)")
@click.option("--video-config", type=click.Path(exists=True), help="Video layout config YAML (default: templates/video_config.yaml)")
@click.option("--width", default=1920, type=int, help="Video width (default: 1920)")
@click.option("--height", default=1080, type=int, help="Video height (default: 1080)")
@click.option("--fps", default=30, type=int, help="Frames per second (default: 30)")
@click.option("--waveform-width", default=960, type=int, help="Waveform width (default: 960)")
@click.option("--waveform-height", default=200, type=int, help="Waveform height (default: 200)")
@click.option("--waveform-color", default="240,240,240", help="Waveform color as R,G,B (default: 240,240,240)")
@click.option("--bg-color", default="20,20,30", help="Background color as R,G,B (default: 20,20,30)")
@click.option("--compact", is_flag=True, help="Use compact encoding (smaller file, slower encoding)")
@click.option("--encoder", "-e", default="auto",
              type=click.Choice(["auto", "cpu", "videotoolbox", "nvenc", "vaapi", "qsv", "amf"]),
              help="Video encoder (default: auto-detect)")
@click.option("--list-encoders", is_flag=True, help="List available encoders and exit")
@click.pass_context
def generate_video_cmd(ctx, input_path, output, title, background, icon, video_config, width, height, fps, waveform_width, waveform_height, waveform_color, bg_color, compact, encoder, list_encoders):
    """Generate video with waveform visualization from audio using ffmpeg."""
    # Handle --list-encoders
    if list_encoders:
        available = get_available_encoders()
        click.echo("Available encoders:")
        for enc in available:
            marker = " (recommended)" if enc == available[0] and enc != "cpu" else ""
            click.echo(f"  - {enc}{marker}")
        return

    # Require input_path if not listing encoders
    if not input_path:
        raise click.UsageError("Missing option '--input' / '-i'.")

    config = ctx.obj["config"]

    # Parse colors
    waveform_color_tuple = parse_color(waveform_color)
    bg_color_tuple = parse_color(bg_color)

    # Resolve background image path
    if background:
        background_path = Path(background)
    else:
        # Check for default background.png, fall back to placeholder
        default_bg = Path("templates/background.png")
        placeholder_bg = Path("templates/background_placeholder.png")
        if default_bg.exists():
            background_path = default_bg
            click.echo(f"Using background: {default_bg}")
        elif placeholder_bg.exists():
            background_path = placeholder_bg
            click.echo(f"Note: templates/background.png not found, using placeholder")
        else:
            background_path = None

    # Resolve icon/logo image path
    if icon:
        icon_path = Path(icon)
    else:
        # Check for default logo.png, fall back to placeholder
        default_logo = Path("templates/logo.png")
        placeholder_logo = Path("templates/logo_placeholder.png")
        if default_logo.exists():
            icon_path = default_logo
            click.echo(f"Using logo: {default_logo}")
        elif placeholder_logo.exists():
            icon_path = placeholder_logo
            click.echo(f"Note: templates/logo.png not found, using placeholder")
        else:
            icon_path = None

    # Resolve video config path
    if video_config:
        video_config_path = Path(video_config)
        click.echo(f"Using video config: {video_config_path}")
    else:
        # Check for default video_config.yaml
        default_config = Path("templates/video_config.yaml")
        if default_config.exists():
            video_config_path = default_config
        else:
            video_config_path = None

    # Determine output path
    if output:
        output_path = Path(output)
    else:
        output_dir = Path(config["local"]["output_dir"]) / "videos"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{Path(input_path).stem}_video.mp4"

    click.echo(f"Generating video from: {input_path}")
    click.echo(f"Resolution: {width}x{height} @ {fps}fps")
    click.echo(f"Waveform: {waveform_width}x{waveform_height}")
    click.echo(f"Encoding: {'compact' if compact else 'fast'}")
    if title:
        click.echo(f"Title: {title}")

    try:
        result = generate_video(
            audio_path=Path(input_path),
            output_path=output_path,
            title=title,
            background_path=background_path,
            icon_path=icon_path,
            width=width,
            height=height,
            fps=fps,
            waveform_width=waveform_width,
            waveform_height=waveform_height,
            waveform_color=waveform_color_tuple,
            bg_color=bg_color_tuple,
            compact=compact,
            encoder=encoder,
            video_config_path=video_config_path,
            show_progress=True,
        )
        click.echo(f"Video saved to: {result}")
    except (FileNotFoundError, VideoGenerationError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command("preview-frame")
@click.option("--output", "-o", type=click.Path(), help="Output image file (default: output/preview.png)")
@click.option("--title", "-t", default="Sample Title Text", help="Title text to preview")
@click.option("--background", "-bg", type=click.Path(exists=True), help="Background image (default: templates/background.png)")
@click.option("--icon", type=click.Path(exists=True), help="Logo image (default: templates/logo.png)")
@click.option("--video-config", type=click.Path(exists=True), help="Video layout config YAML (default: templates/video_config.yaml)")
@click.option("--width", default=1920, type=int, help="Frame width (default: 1920)")
@click.option("--height", default=1080, type=int, help="Frame height (default: 1080)")
@click.option("--waveform-width", default=960, type=int, help="Waveform width (default: 960)")
@click.option("--waveform-height", default=200, type=int, help="Waveform height (default: 200)")
@click.option("--waveform-color", default="240,240,240", help="Waveform color as R,G,B (default: 240,240,240)")
@click.option("--bg-color", default="20,20,30", help="Background color as R,G,B (default: 20,20,30)")
@click.option("--open/--no-open", default=True, help="Auto-open preview after generation (default: --open)")
@click.pass_context
def preview_frame_cmd(ctx, output, title, background, icon, video_config, width, height, waveform_width, waveform_height, waveform_color, bg_color, open):
    """Generate a single preview frame to check layout before rendering video."""
    config = ctx.obj["config"]

    # Parse colors
    waveform_color_tuple = parse_color(waveform_color)
    bg_color_tuple = parse_color(bg_color)

    # Resolve background image path
    if background:
        background_path = Path(background)
    else:
        default_bg = Path("templates/background.png")
        placeholder_bg = Path("templates/background_placeholder.png")
        if default_bg.exists():
            background_path = default_bg
        elif placeholder_bg.exists():
            background_path = placeholder_bg
        else:
            background_path = None

    # Resolve icon/logo image path
    if icon:
        icon_path = Path(icon)
    else:
        default_logo = Path("templates/logo.png")
        placeholder_logo = Path("templates/logo_placeholder.png")
        if default_logo.exists():
            icon_path = default_logo
        elif placeholder_logo.exists():
            icon_path = placeholder_logo
        else:
            icon_path = None

    # Resolve video config path
    if video_config:
        video_config_path = Path(video_config)
        click.echo(f"Using video config: {video_config_path}")
    else:
        default_config = Path("templates/video_config.yaml")
        if default_config.exists():
            video_config_path = default_config
            click.echo(f"Using video config: {video_config_path}")
        else:
            video_config_path = None

    # Determine output path
    if output:
        output_path = Path(output)
    else:
        output_dir = Path(config["local"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "preview.png"

    click.echo(f"Generating preview frame...")

    try:
        result = generate_preview_frame(
            output_path=output_path,
            title=title,
            background_path=background_path,
            icon_path=icon_path,
            width=width,
            height=height,
            waveform_width=waveform_width,
            waveform_height=waveform_height,
            waveform_color=waveform_color_tuple,
            bg_color=bg_color_tuple,
            video_config_path=video_config_path,
        )
        click.echo(f"Preview saved to: {result}")

        # Auto-open the preview
        if open:
            import subprocess
            subprocess.run(["open", str(result)], check=False)

    except VideoGenerationError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--url", "-u", required=True, help="Clubhouse recording URL")
@click.option("--title", "-t", required=True, help="Episode title")
@click.option("--output", "-o", type=click.Path(), help="Output directory")
@click.pass_context
def process(ctx, url, title, output):
    """Run the full processing pipeline (download -> extract -> transcribe -> summarize)."""
    config = ctx.obj["config"]
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        click.echo("Error: GEMINI_API_KEY environment variable not set", err=True)
        sys.exit(1)

    output_base = Path(output) if output else Path(config["local"]["output_dir"])

    click.echo(f"Processing: {title}")
    click.echo("=" * 50)

    try:
        # Step 1: Download
        click.echo("\n[1/4] Downloading video...")
        audio_dir = output_base / "audio"
        video_path = download_clubhouse_video(
            url=url,
            output_dir=audio_dir,
        )
        click.echo(f"      Downloaded: {video_path}")

        # Step 2: Extract audio
        click.echo("\n[2/4] Extracting audio...")
        audio_path = extract_audio(
            video_path=video_path,
            ffmpeg_path=config["local"].get("ffmpeg_path", "ffmpeg"),
            overwrite=True,
        )
        click.echo(f"      Extracted: {audio_path}")

        # Step 3: Transcribe
        click.echo("\n[3/4] Transcribing audio...")
        transcript = transcribe_audio(
            audio_path=audio_path,
            api_key=api_key,
            language=config.get("transcription", {}).get("language", "en"),
        )

        transcript_dir = output_base / "transcripts"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = transcript_dir / f"{audio_path.stem}_transcript.txt"
        transcript_path.write_text(transcript)
        click.echo(f"      Transcript: {transcript_path}")

        # Step 4: Generate descriptions
        click.echo("\n[4/4] Generating descriptions...")
        summary_config = config.get("summary", {})
        descriptions = generate_descriptions(
            transcript=transcript,
            episode_title=title,
            api_key=api_key,
            youtube_max_length=summary_config.get("youtube_max_length", 5000),
            spotify_max_length=summary_config.get("spotify_max_length", 4000),
            generate_tags=summary_config.get("generate_tags", True),
            max_tags=summary_config.get("max_tags", 10),
        )

        desc_dir = output_base / "descriptions"
        desc_dir.mkdir(parents=True, exist_ok=True)
        desc_path = desc_dir / f"{audio_path.stem}_descriptions.yaml"
        with open(desc_path, "w") as f:
            yaml.dump(descriptions, f, default_flow_style=False, allow_unicode=True)
        click.echo(f"      Descriptions: {desc_path}")

        # Summary
        click.echo("\n" + "=" * 50)
        click.echo("Processing complete!")
        click.echo(f"\nOutputs:")
        click.echo(f"  Video:       {video_path}")
        click.echo(f"  Audio:       {audio_path}")
        click.echo(f"  Transcript:  {transcript_path}")
        click.echo(f"  Descriptions: {desc_path}")
        click.echo(f"\nYouTube Title: {descriptions['youtube_title']}")
        click.echo(f"Tags: {', '.join(descriptions['tags'])}")

    except (ValueError, DownloadError) as e:
        click.echo(f"\nDownload error: {e}", err=True)
        sys.exit(1)
    except (FileNotFoundError, AudioExtractionError) as e:
        click.echo(f"\nExtraction error: {e}", err=True)
        sys.exit(1)
    except TranscriptionError as e:
        click.echo(f"\nTranscription error: {e}", err=True)
        sys.exit(1)
    except SummaryError as e:
        click.echo(f"\nSummary error: {e}", err=True)
        sys.exit(1)


def main():
    """Entry point for the CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
