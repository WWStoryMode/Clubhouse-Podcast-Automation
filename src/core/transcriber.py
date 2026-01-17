"""Transcribe audio using Gemini API."""

import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional, List

import google.generativeai as genai


class TranscriptionError(Exception):
    """Raised when transcription fails."""

    pass


def configure_gemini(api_key: Optional[str] = None) -> None:
    """
    Configure the Gemini API with the provided key.

    Args:
        api_key: Gemini API key. If None, reads from GEMINI_API_KEY env var.

    Raises:
        TranscriptionError: If no API key is provided or found.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY")

    if not key:
        raise TranscriptionError(
            "Gemini API key not provided. "
            "Set GEMINI_API_KEY environment variable or pass api_key parameter."
        )

    genai.configure(api_key=key)


def transcribe_audio(
    audio_path: Path,
    api_key: Optional[str] = None,
    language: str = "en",
    include_timestamps: bool = False,
    model_name: str = "gemini-2.5-flash",
) -> str:
    """
    Transcribe audio using Gemini API.

    Args:
        audio_path: Path to audio file (MP3, WAV, etc.)
        api_key: Gemini API key. If None, reads from GEMINI_API_KEY env var.
        language: Language code (ISO 639-1) for transcription
        include_timestamps: Whether to include timestamps in transcript
        model_name: Gemini model to use

    Returns:
        Full transcript text

    Raises:
        FileNotFoundError: If audio file doesn't exist
        TranscriptionError: If transcription fails
    """
    audio_path = Path(audio_path)

    # Validate input file
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if not audio_path.is_file():
        raise FileNotFoundError(f"Path is not a file: {audio_path}")

    # Configure API
    configure_gemini(api_key)

    # Map language codes to human-readable descriptions
    language_map = {
        "yue": "Cantonese (Hong Kong)",
        "zh-HK": "Cantonese (Hong Kong)",
        "zh": "Mandarin Chinese",
        "zh-TW": "Traditional Chinese (Taiwan)",
        "en": "English",
    }
    language_desc = language_map.get(language, language)

    # Build the prompt
    if include_timestamps:
        prompt = f"""Transcribe the following audio file accurately.

Background: This is an audio recording from an online live show or podcast conversation with multiple speakers.

Language: {language_desc}

Instructions:
- Transcribe all spoken content into clear, complete sentences
- Include timestamps in the format [MM:SS] at the beginning of each paragraph or when the speaker changes
- Preserve the natural flow of conversation
- Keep colloquial expressions and slang as spoken
- Separate different speakers' dialogue into distinct paragraphs
- Provide only the transcript, no additional commentary or summaries"""
    else:
        prompt = f"""Transcribe the following audio file accurately.

Background: This is an audio recording from an online live show or podcast conversation with multiple speakers.

Language: {language_desc}

Instructions:
- Transcribe all spoken content into clear, complete sentences
- Preserve the natural flow of conversation
- Keep colloquial expressions and slang as spoken
- Separate different speakers' dialogue into distinct paragraphs
- Provide only the transcript text, no timestamps or additional commentary"""

    try:
        # Upload the audio file
        audio_file = genai.upload_file(str(audio_path))

        # Create the model and generate transcription
        model = genai.GenerativeModel(model_name)

        response = model.generate_content(
            [prompt, audio_file],
            generation_config=genai.GenerationConfig(
                temperature=0.1,  # Low temperature for accuracy
                max_output_tokens=8192,
            ),
        )

        # Clean up uploaded file
        try:
            audio_file.delete()
        except Exception:
            pass  # Ignore cleanup errors

        if not response.text:
            raise TranscriptionError("Gemini returned empty response")

        return response.text.strip()

    except TranscriptionError:
        # Re-raise our own errors
        raise
    except Exception as e:
        error_msg = str(e).lower()
        if "api key" in error_msg:
            raise TranscriptionError(f"Gemini API key error: {e}")
        elif "blocked" in error_msg or "safety" in error_msg:
            raise TranscriptionError(f"Content blocked by Gemini safety filters: {e}")
        raise TranscriptionError(f"Transcription failed: {e}")


def get_audio_duration(audio_path: Path, ffprobe_path: str = "ffprobe") -> float:
    """
    Get the duration of an audio file in seconds.

    Args:
        audio_path: Path to audio file
        ffprobe_path: Path to ffprobe executable

    Returns:
        Duration in seconds

    Raises:
        TranscriptionError: If duration cannot be determined
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
        raise TranscriptionError(f"Failed to get audio duration: {e}")


def split_audio(
    audio_path: Path,
    output_dir: Path,
    chunk_duration_seconds: int,
    ffmpeg_path: str = "ffmpeg",
) -> List[Path]:
    """
    Split an audio file into chunks of specified duration.

    Args:
        audio_path: Path to input audio file
        output_dir: Directory to save chunks
        chunk_duration_seconds: Duration of each chunk in seconds
        ffmpeg_path: Path to ffmpeg executable

    Returns:
        List of paths to chunk files

    Raises:
        TranscriptionError: If splitting fails
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get total duration
    total_duration = get_audio_duration(audio_path)

    chunks = []
    chunk_num = 0
    start_time = 0

    while start_time < total_duration:
        chunk_path = output_dir / f"chunk_{chunk_num:03d}.mp3"

        try:
            subprocess.run(
                [
                    ffmpeg_path,
                    "-y",  # Overwrite
                    "-i", str(audio_path),
                    "-ss", str(start_time),
                    "-t", str(chunk_duration_seconds),
                    "-acodec", "libmp3lame",
                    "-q:a", "2",
                    str(chunk_path),
                ],
                capture_output=True,
                check=True,
            )
            chunks.append(chunk_path)
        except subprocess.CalledProcessError as e:
            raise TranscriptionError(f"Failed to split audio at {start_time}s: {e.stderr.decode()}")

        start_time += chunk_duration_seconds
        chunk_num += 1

    return chunks


def transcribe_audio_chunked(
    audio_path: Path,
    api_key: Optional[str] = None,
    language: str = "en",
    chunk_duration_minutes: int = 10,
    model_name: str = "gemini-2.5-flash",
    delay_between_chunks: int = 5,
    include_timestamps: bool = False,
    show_progress: bool = True,
) -> str:
    """
    Transcribe long audio files by processing in chunks.

    For very long audio files (> 1 hour), this function splits the audio
    and transcribes each chunk separately.

    Args:
        audio_path: Path to audio file
        api_key: Gemini API key
        language: Language code
        chunk_duration_minutes: Duration of each chunk in minutes
        model_name: Gemini model to use
        delay_between_chunks: Seconds to wait between API calls (rate limiting)
        include_timestamps: Whether to include timestamps in transcript
        show_progress: Whether to print progress messages

    Returns:
        Full concatenated transcript
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Get total duration
    total_duration = get_audio_duration(audio_path)
    chunk_duration_seconds = chunk_duration_minutes * 60

    # If audio is short enough, use regular transcription
    if total_duration <= chunk_duration_seconds:
        if show_progress:
            print(f"Audio is {total_duration/60:.1f} minutes, using single transcription...")
        return transcribe_audio(
            audio_path=audio_path,
            api_key=api_key,
            language=language,
            include_timestamps=include_timestamps,
            model_name=model_name,
        )

    # Calculate number of chunks
    num_chunks = int(total_duration / chunk_duration_seconds) + 1
    if show_progress:
        print(f"Audio is {total_duration/60:.1f} minutes, splitting into {num_chunks} chunks...")

    # Create temp directory for chunks
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Split audio
        if show_progress:
            print("Splitting audio into chunks...")
        chunks = split_audio(audio_path, temp_path, chunk_duration_seconds)

        if show_progress:
            print(f"Created {len(chunks)} chunks")

        # Transcribe each chunk
        transcripts = []
        for i, chunk_path in enumerate(chunks):
            if show_progress:
                print(f"Transcribing chunk {i+1}/{len(chunks)}...")

            # Calculate chunk start time for timestamp adjustment
            chunk_start_minutes = i * chunk_duration_minutes

            try:
                transcript = transcribe_audio(
                    audio_path=chunk_path,
                    api_key=api_key,
                    language=language,
                    include_timestamps=include_timestamps,
                    model_name=model_name,
                )

                # Add chunk marker with time offset if using timestamps
                if include_timestamps and chunk_start_minutes > 0:
                    transcript = f"[Chunk {i+1} - starts at {chunk_start_minutes}:00]\n{transcript}"

                transcripts.append(transcript)

                if show_progress:
                    print(f"  Chunk {i+1} completed ({len(transcript)} chars)")

            except TranscriptionError as e:
                if show_progress:
                    print(f"  Chunk {i+1} failed: {e}")
                transcripts.append(f"[Transcription failed for chunk {i+1}]")

            # Rate limiting delay between chunks
            if i < len(chunks) - 1 and delay_between_chunks > 0:
                if show_progress:
                    print(f"  Waiting {delay_between_chunks}s before next chunk...")
                time.sleep(delay_between_chunks)

    # Combine transcripts
    full_transcript = "\n\n".join(transcripts)

    if show_progress:
        print(f"Transcription complete. Total: {len(full_transcript)} chars")

    return full_transcript
