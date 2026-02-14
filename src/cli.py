"""Command-line interface for Clubhouse-Podcast-Automation."""

import os
import subprocess
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
from .core.pipeline import PipelineState, get_episode_id_from_url
from .core.video_generator import generate_video, generate_preview_frame, VideoGenerationError, get_available_encoders
from .core.youtube_uploader import upload_video as youtube_upload, load_metadata_from_yaml, YouTubeUploadError


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
@click.option("--trim-start", "-ss", help="Start time for trimming (HH:MM:SS, MM:SS, or seconds)")
@click.option("--trim-end", "-to", help="End time for trimming (HH:MM:SS, MM:SS, or seconds)")
@click.option("--normalize/--no-normalize", default=True, help="Apply loudnorm filter for consistent audio levels (default: --normalize)")
@click.pass_context
def extract(ctx, input_path, output, trim_start, trim_end, normalize):
    """Extract audio from video file with optional trimming and normalization."""
    config = ctx.obj["config"]
    ffmpeg_path = config["local"].get("ffmpeg_path", "ffmpeg")

    click.echo(f"Extracting audio from: {input_path}")
    if trim_start or trim_end:
        trim_info = []
        if trim_start:
            trim_info.append(f"start={trim_start}")
        if trim_end:
            trim_info.append(f"end={trim_end}")
        click.echo(f"Trimming: {', '.join(trim_info)}")
    if normalize:
        click.echo("Normalizing audio levels (loudnorm)")

    try:
        output_path = Path(output) if output else None
        result = extract_audio(
            video_path=Path(input_path),
            output_path=output_path,
            ffmpeg_path=ffmpeg_path,
            overwrite=True,
            trim_start=trim_start,
            trim_end=trim_end,
            normalize=normalize,
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
@click.option("--model", "-m", default="gemini-2.5-flash",
              help="Gemini model: gemini-2.5-flash (default), gemini-2.5-pro, gemini-2.0-flash, gemini-3-pro-preview")
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
@click.option("--video-bitrate", default=None,
              help="Override video bitrate (e.g., '2M', '1500k'). Overrides default CRF/bitrate settings.")
@click.option("--list-encoders", is_flag=True, help="List available encoders and exit")
@click.pass_context
def generate_video_cmd(ctx, input_path, output, title, background, icon, video_config, width, height, fps, waveform_width, waveform_height, waveform_color, bg_color, compact, encoder, video_bitrate, list_encoders):
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

    # Resolve video bitrate: CLI flag > config file > built-in default (None)
    if video_bitrate is None:
        video_bitrate = config.get("video", {}).get("bitrate", None)

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
    if video_bitrate:
        click.echo(f"Video bitrate: {video_bitrate}")
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
            video_bitrate=video_bitrate,
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


@cli.command("upload-youtube")
@click.option("--video", "-v", "video_path", required=True, type=click.Path(exists=True), help="Video file to upload")
@click.option("--title", "-t", help="Video title (or use --metadata)")
@click.option("--description", "-d", default="", help="Video description")
@click.option("--tags", help="Comma-separated tags (e.g., 'tag1,tag2,tag3')")
@click.option("--metadata", "-m", type=click.Path(exists=True), help="Load title/description/tags from descriptions YAML")
@click.option("--privacy", default="private", type=click.Choice(["private", "unlisted", "public"]), help="Privacy setting (default: private)")
@click.option("--category", default="22", help="YouTube category ID (default: 22 = People & Blogs)")
@click.option("--credentials", type=click.Path(exists=True), help="Path to OAuth client_secret.json (default: credentials/client_secret.json)")
@click.pass_context
def upload_youtube_cmd(ctx, video_path, title, description, tags, metadata, privacy, category, credentials):
    """Upload video to YouTube."""
    # Load metadata from YAML if provided
    if metadata:
        click.echo(f"Loading metadata from: {metadata}")
        try:
            meta = load_metadata_from_yaml(Path(metadata))
            if not title:
                title = meta.get("title", "")
            if not description:
                description = meta.get("description", "")
            if not tags:
                tags = ",".join(meta.get("tags", []))
        except YouTubeUploadError as e:
            click.echo(f"Error loading metadata: {e}", err=True)
            sys.exit(1)

    # Require title
    if not title:
        click.echo("Error: --title is required (or use --metadata with a descriptions YAML)", err=True)
        sys.exit(1)

    # Parse tags
    tag_list = [t.strip() for t in tags.split(",")] if tags else []

    # Resolve credentials path
    credentials_path = Path(credentials) if credentials else Path("credentials/client_secret.json")

    click.echo(f"Uploading to YouTube: {video_path}")
    click.echo(f"Title: {title}")
    click.echo(f"Privacy: {privacy}")
    if tag_list:
        click.echo(f"Tags: {', '.join(tag_list)}")

    try:
        result = youtube_upload(
            video_path=Path(video_path),
            title=title,
            description=description,
            tags=tag_list,
            category_id=category,
            privacy=privacy,
            client_secrets_path=credentials_path,
            show_progress=True,
        )
        click.echo(f"\nSuccess! Video uploaded:")
        click.echo(f"  URL: {result['url']}")
        click.echo(f"  Video ID: {result['video_id']}")
    except (FileNotFoundError, YouTubeUploadError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--url", "-u", required=True, help="Clubhouse recording URL")
@click.option("--title", "-t", required=True, help="Episode title")
@click.option("--output", "-o", type=click.Path(), help="Output directory")
@click.option("--resume", is_flag=True, help="Resume from last successful step (skip steps with existing output files)")
@click.option("--interactive", "-I", is_flag=True, help="Review each step's output before continuing")
@click.pass_context
def process(ctx, url, title, output, resume, interactive):
    """Run the full processing pipeline (download -> extract -> transcribe -> summarize -> generate video)."""
    config = ctx.obj["config"]
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        click.echo("Error: GEMINI_API_KEY environment variable not set", err=True)
        sys.exit(1)

    output_base = Path(output) if output else Path(config["local"]["output_dir"])

    episode_id = get_episode_id_from_url(url)
    state = PipelineState(episode_id, output_base)

    click.echo(f"Processing: {title}")
    click.echo(f"Episode ID: {episode_id}")
    click.echo("=" * 50)

    if resume:
        completed = state.get_completed_steps()
        if completed:
            click.echo(f"Resuming — completed steps: {', '.join(completed)}")
        next_step = state.get_next_step()
        if next_step is None:
            click.echo("\nAll steps already complete. Nothing to do.")
            return

    def _confirm_step(step_label, file_path):
        """Open output in Finder and prompt the user to confirm before continuing."""
        path = Path(file_path)
        subprocess.run(["open", "-R", str(path)])
        if not click.confirm(f"{step_label} result OK? Continue?", default=True):
            click.echo("Pipeline aborted by user.")
            sys.exit(0)

    try:
        # Step 1: Download
        if resume and state.video_path.exists():
            click.echo("\n[1/5] Downloading video... SKIPPED (exists)")
            video_path = state.video_path
        else:
            click.echo("\n[1/5] Downloading video...")
            audio_dir = output_base / "audio"
            video_path = download_clubhouse_video(
                url=url,
                output_dir=audio_dir,
            )
            click.echo(f"      Downloaded: {video_path}")
            if interactive:
                _confirm_step("[1/5] Download", video_path)

        # Step 2: Extract audio
        if resume and state.audio_path.exists():
            click.echo("\n[2/5] Extracting audio... SKIPPED (exists)")
            audio_path = state.audio_path
        else:
            click.echo("\n[2/5] Extracting audio...")
            audio_path = extract_audio(
                video_path=video_path,
                ffmpeg_path=config["local"].get("ffmpeg_path", "ffmpeg"),
                overwrite=True,
            )
            click.echo(f"      Extracted: {audio_path}")
            if interactive:
                _confirm_step("[2/5] Extract", audio_path)

        # Step 3: Transcribe
        if resume and state.transcript_path.exists():
            click.echo("\n[3/5] Transcribing audio... SKIPPED (exists)")
            transcript_path = state.transcript_path
            transcript = transcript_path.read_text()
        else:
            click.echo("\n[3/5] Transcribing audio...")
            transcript = transcribe_audio(
                audio_path=audio_path,
                api_key=api_key,
                language=config.get("transcription", {}).get("language", "en"),
            )

            transcript_dir = output_base / "transcripts"
            transcript_dir.mkdir(parents=True, exist_ok=True)
            transcript_path = state.transcript_path
            transcript_path.write_text(transcript)
            click.echo(f"      Transcript: {transcript_path}")
            if interactive:
                _confirm_step("[3/5] Transcribe", transcript_path)

        # Step 4: Generate descriptions
        if resume and state.descriptions_path.exists():
            click.echo("\n[4/5] Generating descriptions... SKIPPED (exists)")
            desc_path = state.descriptions_path
            with open(desc_path) as f:
                descriptions = yaml.safe_load(f)
        else:
            click.echo("\n[4/5] Generating descriptions...")
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
            desc_path = state.descriptions_path
            with open(desc_path, "w") as f:
                yaml.dump(descriptions, f, default_flow_style=False, allow_unicode=True)
            click.echo(f"      Descriptions: {desc_path}")
            if interactive:
                _confirm_step("[4/5] Summarize", desc_path)

        # Step 5: Generate video
        if resume and state.generated_video_path.exists():
            click.echo("\n[5/5] Generating video... SKIPPED (exists)")
            generated_video_path = state.generated_video_path
        else:
            click.echo("\n[5/5] Generating video...")
            video_config = config.get("video", {})

            # Resolve template paths
            background_path = None
            for bg_candidate in [Path("templates/background.png"), Path("templates/background_placeholder.png")]:
                if bg_candidate.exists():
                    background_path = bg_candidate
                    break

            icon_path = None
            for icon_candidate in [Path("templates/logo.png"), Path("templates/logo_placeholder.png")]:
                if icon_candidate.exists():
                    icon_path = icon_candidate
                    break

            video_config_path = None
            default_video_config = Path("templates/video_config.yaml")
            if default_video_config.exists():
                video_config_path = default_video_config

            videos_dir = output_base / "videos"
            videos_dir.mkdir(parents=True, exist_ok=True)
            generated_video_path = state.generated_video_path

            generated_video_path = generate_video(
                audio_path=audio_path,
                output_path=generated_video_path,
                title=title,
                background_path=background_path,
                icon_path=icon_path,
                width=video_config.get("width", 1920),
                height=video_config.get("height", 1080),
                fps=video_config.get("fps", 30),
                waveform_width=video_config.get("waveform_width", 960),
                waveform_height=video_config.get("waveform_height", 200),
                waveform_color=parse_color(video_config.get("waveform_color", "240,240,240")),
                bg_color=parse_color(video_config.get("bg_color", "20,20,30")),
                compact=video_config.get("compact", False),
                encoder=video_config.get("encoder", "auto"),
                video_bitrate=video_config.get("bitrate", None),
                video_config_path=video_config_path,
                show_progress=True,
            )
            click.echo(f"      Video: {generated_video_path}")
            if interactive:
                _confirm_step("[5/5] Generate video", generated_video_path)

        # Summary
        click.echo("\n" + "=" * 50)
        click.echo("Processing complete!")
        click.echo(f"\nOutputs:")
        click.echo(f"  Video:       {state.video_path}")
        click.echo(f"  Audio:       {state.audio_path}")
        click.echo(f"  Transcript:  {state.transcript_path}")
        click.echo(f"  Descriptions: {state.descriptions_path}")
        click.echo(f"  Generated:   {state.generated_video_path}")
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
    except VideoGenerationError as e:
        click.echo(f"\nVideo generation error: {e}", err=True)
        sys.exit(1)


def main():
    """Entry point for the CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
