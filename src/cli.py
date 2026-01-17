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
