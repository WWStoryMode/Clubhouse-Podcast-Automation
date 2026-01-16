"""Transcribe audio using Gemini API."""

import os
from pathlib import Path
from typing import Optional

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
    model_name: str = "gemini-1.5-flash",
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

    # Build the prompt
    if include_timestamps:
        prompt = f"""Transcribe the following audio file accurately.
Include timestamps in the format [MM:SS] at the beginning of each paragraph or when the speaker changes.
The audio is in {language} language.
Provide only the transcript, no additional commentary."""
    else:
        prompt = f"""Transcribe the following audio file accurately.
The audio is in {language} language.
Provide only the transcript text, no timestamps or additional commentary."""

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


def transcribe_audio_chunked(
    audio_path: Path,
    api_key: Optional[str] = None,
    language: str = "en",
    chunk_duration_minutes: int = 30,
    model_name: str = "gemini-1.5-flash",
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

    Returns:
        Full concatenated transcript

    Note:
        This is a placeholder for future implementation.
        Currently delegates to regular transcribe_audio.
    """
    # TODO: Implement chunked transcription for very long files
    # For now, use regular transcription
    return transcribe_audio(
        audio_path=audio_path,
        api_key=api_key,
        language=language,
        include_timestamps=False,
        model_name=model_name,
    )
