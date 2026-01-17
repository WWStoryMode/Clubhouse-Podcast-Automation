"""Generate descriptions for YouTube and Spotify using Gemini API."""

import os
from typing import Optional, List

import google.generativeai as genai


class SummaryError(Exception):
    """Raised when summary generation fails."""

    pass


def configure_gemini(api_key: Optional[str] = None) -> None:
    """
    Configure the Gemini API with the provided key.

    Args:
        api_key: Gemini API key. If None, reads from GEMINI_API_KEY env var.

    Raises:
        SummaryError: If no API key is provided or found.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY")

    if not key:
        raise SummaryError(
            "Gemini API key not provided. "
            "Set GEMINI_API_KEY environment variable or pass api_key parameter."
        )

    genai.configure(api_key=key)


def generate_descriptions(
    transcript: str,
    episode_title: str,
    api_key: Optional[str] = None,
    youtube_max_length: int = 5000,
    spotify_max_length: int = 4000,
    generate_tags: bool = True,
    max_tags: int = 10,
    model_name: str = "gemini-3-flash-preview",
) -> dict:
    """
    Generate platform-specific descriptions using Gemini API.

    Args:
        transcript: Full transcript text
        episode_title: Title of the episode
        api_key: Gemini API key. If None, reads from GEMINI_API_KEY env var.
        youtube_max_length: Max characters for YouTube description
        spotify_max_length: Max characters for Spotify description
        generate_tags: Whether to generate tags
        max_tags: Maximum number of tags to generate
        model_name: Gemini model to use

    Returns:
        Dictionary with:
        {
            "youtube_title": str,
            "youtube_description": str,
            "spotify_title": str,
            "spotify_description": str,
            "tags": list[str]
        }

    Raises:
        SummaryError: If summary generation fails
    """
    if not transcript or not transcript.strip():
        raise SummaryError("Transcript is empty")

    if not episode_title or not episode_title.strip():
        raise SummaryError("Episode title is empty")

    # Configure API
    configure_gemini(api_key)

    # Build the prompt
    prompt = f"""You are a podcast content assistant. Based on the following transcript, generate content for publishing this episode.

Episode Title: {episode_title}

Transcript:
{transcript[:10000]}  # Limit transcript length for API

Please generate:

1. YOUTUBE_TITLE: A catchy, SEO-friendly title for YouTube (max 100 characters). Keep it engaging but informative.

2. YOUTUBE_DESCRIPTION: A comprehensive description for YouTube (max {youtube_max_length} characters) that includes:
   - A brief summary of the episode (2-3 sentences)
   - Key topics discussed (bullet points)
   - Timestamps for major sections if identifiable
   - A call to action to subscribe

3. SPOTIFY_TITLE: The episode title for Spotify (can be same as original or slightly modified, max 100 characters)

4. SPOTIFY_DESCRIPTION: A description for Spotify (max {spotify_max_length} characters) that includes:
   - A concise summary of the episode
   - Key takeaways
   - Keep it more conversational than YouTube

5. TAGS: {max_tags} relevant tags/keywords for discoverability (comma-separated)

Format your response exactly as follows:
YOUTUBE_TITLE: [title here]
YOUTUBE_DESCRIPTION: [description here]
SPOTIFY_TITLE: [title here]
SPOTIFY_DESCRIPTION: [description here]
TAGS: [tag1, tag2, tag3, ...]
"""

    try:
        model = genai.GenerativeModel(model_name)

        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.7,
                max_output_tokens=4096,
            ),
        )

        if not response.text:
            raise SummaryError("Gemini returned empty response")

        # Parse the response
        return _parse_response(
            response.text,
            episode_title,
            generate_tags,
        )

    except SummaryError:
        raise
    except Exception as e:
        error_msg = str(e).lower()
        if "api key" in error_msg:
            raise SummaryError(f"Gemini API key error: {e}")
        raise SummaryError(f"Summary generation failed: {e}")


def _parse_response(
    response_text: str,
    fallback_title: str,
    include_tags: bool,
) -> dict:
    """
    Parse the Gemini response into structured data.

    Args:
        response_text: Raw response from Gemini
        fallback_title: Title to use if parsing fails
        include_tags: Whether to include tags in result

    Returns:
        Parsed dictionary with all fields
    """
    result = {
        "youtube_title": fallback_title,
        "youtube_description": "",
        "spotify_title": fallback_title,
        "spotify_description": "",
        "tags": [],
    }

    lines = response_text.strip().split("\n")
    current_field = None
    current_content = []

    field_mapping = {
        "YOUTUBE_TITLE:": "youtube_title",
        "YOUTUBE_DESCRIPTION:": "youtube_description",
        "SPOTIFY_TITLE:": "spotify_title",
        "SPOTIFY_DESCRIPTION:": "spotify_description",
        "TAGS:": "tags",
    }

    for line in lines:
        line_upper = line.strip().upper()

        # Check if this line starts a new field
        found_field = None
        for prefix, field_name in field_mapping.items():
            if line_upper.startswith(prefix.upper()):
                found_field = field_name
                # Get content after the prefix
                content = line[len(prefix):].strip()
                break

        if found_field:
            # Save previous field content
            if current_field and current_content:
                if current_field == "tags":
                    result[current_field] = _parse_tags("\n".join(current_content))
                else:
                    result[current_field] = "\n".join(current_content).strip()

            # Start new field
            current_field = found_field
            current_content = [content] if content else []
        elif current_field:
            # Continue current field
            current_content.append(line)

    # Save last field
    if current_field and current_content:
        if current_field == "tags":
            result[current_field] = _parse_tags("\n".join(current_content))
        else:
            result[current_field] = "\n".join(current_content).strip()

    # Clean up tags if not requested
    if not include_tags:
        result["tags"] = []

    return result


def _parse_tags(tags_text: str) -> List[str]:
    """
    Parse tags from text.

    Args:
        tags_text: Comma or newline separated tags

    Returns:
        List of cleaned tags
    """
    # Handle both comma and newline separated
    tags_text = tags_text.replace("\n", ",")
    tags = [tag.strip().strip("#") for tag in tags_text.split(",")]
    return [tag for tag in tags if tag]  # Remove empty tags
