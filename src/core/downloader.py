"""Download Clubhouse recordings from direct links."""

from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from tqdm import tqdm


class DownloadError(Exception):
    """Raised when download fails."""

    pass


def validate_url(url: str) -> bool:
    """
    Validate if the URL is properly formatted.

    Args:
        url: URL string to validate

    Returns:
        True if URL is valid, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme in ("http", "https"), result.netloc])
    except Exception:
        return False


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing invalid characters.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename safe for filesystem
    """
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "_")

    # Remove leading/trailing spaces and dots
    filename = filename.strip(". ")

    # Limit length
    if len(filename) > 200:
        filename = filename[:200]

    return filename or "download"


def download_clubhouse_video(
    url: str,
    output_dir: Path,
    filename: Optional[str] = None,
    timeout: int = 3600,
    chunk_size: int = 8192,
    show_progress: bool = True,
) -> Path:
    """
    Download MP4 from Clubhouse recording link.

    Args:
        url: Direct download URL for the Clubhouse recording
        output_dir: Directory to save the downloaded file
        filename: Optional custom filename (without extension)
        timeout: Download timeout in seconds
        chunk_size: Size of download chunks in bytes
        show_progress: Whether to show progress bar

    Returns:
        Path to downloaded MP4 file

    Raises:
        ValueError: If URL is invalid
        DownloadError: If download fails
    """
    # Validate URL
    if not url or not validate_url(url):
        raise ValueError(f"Invalid URL: {url}")

    # Prepare output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename if not provided
    if not filename:
        # Try to extract a meaningful name from URL
        parsed = urlparse(url)
        path_parts = parsed.path.strip("/").split("/")
        if path_parts and path_parts[-1]:
            base_name = path_parts[-1].split("?")[0]  # Remove query params
            filename = sanitize_filename(base_name)
        else:
            filename = "clubhouse_recording"

    # Ensure .mp4 extension
    if not filename.endswith(".mp4"):
        filename = f"{filename}.mp4"

    output_path = output_dir / filename

    try:
        response = requests.get(
            url,
            stream=True,
            timeout=timeout,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; PodcastAutomation/1.0)"
            },
        )
        response.raise_for_status()

        # Get total file size for progress bar
        total_size = int(response.headers.get("content-length", 0))

        # Download with optional progress bar
        if show_progress and total_size > 0:
            progress_bar = tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=f"Downloading {filename}",
            )
        else:
            progress_bar = None

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    if progress_bar:
                        progress_bar.update(len(chunk))

        if progress_bar:
            progress_bar.close()

        # Verify file was created
        if not output_path.exists():
            raise DownloadError("Download completed but file not found")

        if output_path.stat().st_size == 0:
            output_path.unlink()  # Remove empty file
            raise DownloadError("Downloaded file is empty")

        return output_path

    except requests.exceptions.Timeout:
        raise DownloadError(f"Download timed out after {timeout} seconds")
    except requests.exceptions.HTTPError as e:
        raise DownloadError(f"HTTP error: {e.response.status_code} - {e}")
    except requests.exceptions.ConnectionError as e:
        raise DownloadError(f"Connection error: {e}")
    except requests.exceptions.RequestException as e:
        raise DownloadError(f"Download failed: {e}")
