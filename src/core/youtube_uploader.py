"""Upload videos to YouTube using YouTube Data API v3."""

import os
import pickle
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError


class YouTubeUploadError(Exception):
    """Raised when YouTube upload fails."""
    pass


# OAuth scopes for YouTube upload
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

# Retry settings for resumable uploads
MAX_RETRIES = 10
RETRIABLE_STATUS_CODES = [500, 502, 503, 504]


def get_authenticated_service(
    client_secrets_path: Path,
    token_path: Optional[Path] = None,
) -> Any:
    """
    Authenticate with YouTube API and return service object.

    First run opens browser for OAuth consent. Subsequent runs use cached token.

    Args:
        client_secrets_path: Path to OAuth client secrets JSON file
        token_path: Path to store/load cached token (default: credentials/youtube_token.pickle)

    Returns:
        YouTube API service object

    Raises:
        YouTubeUploadError: If authentication fails
    """
    client_secrets_path = Path(client_secrets_path)
    if not client_secrets_path.exists():
        raise YouTubeUploadError(
            f"Client secrets file not found: {client_secrets_path}\n"
            "Please download OAuth credentials from Google Cloud Console."
        )

    if token_path is None:
        token_path = client_secrets_path.parent / "youtube_token.pickle"
    else:
        token_path = Path(token_path)

    credentials = None

    # Load cached token if exists
    if token_path.exists():
        try:
            with open(token_path, "rb") as token_file:
                credentials = pickle.load(token_file)
        except Exception:
            pass  # Will re-authenticate

    # Refresh or get new credentials
    if credentials and credentials.expired and credentials.refresh_token:
        try:
            credentials.refresh(Request())
        except Exception:
            credentials = None

    if not credentials or not credentials.valid:
        try:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(client_secrets_path),
                SCOPES,
            )
            credentials = flow.run_local_server(
                port=0,
                prompt="consent",
                success_message="Authentication successful! You can close this window.",
            )
        except Exception as e:
            raise YouTubeUploadError(f"OAuth authentication failed: {e}")

        # Save token for future use
        token_path.parent.mkdir(parents=True, exist_ok=True)
        with open(token_path, "wb") as token_file:
            pickle.dump(credentials, token_file)

    try:
        return build("youtube", "v3", credentials=credentials)
    except Exception as e:
        raise YouTubeUploadError(f"Failed to build YouTube service: {e}")


def upload_video(
    video_path: Path,
    title: str,
    description: str = "",
    tags: Optional[List[str]] = None,
    category_id: str = "22",  # People & Blogs
    privacy: str = "private",
    client_secrets_path: Optional[Path] = None,
    token_path: Optional[Path] = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Upload a video to YouTube.

    Args:
        video_path: Path to video file
        title: Video title
        description: Video description
        tags: List of tags/keywords
        category_id: YouTube category ID (default: 22 = People & Blogs)
        privacy: Privacy status: private, unlisted, or public
        client_secrets_path: Path to OAuth client secrets JSON
        token_path: Path to cached OAuth token
        show_progress: Whether to show upload progress

    Returns:
        Dict with video ID and URL

    Raises:
        YouTubeUploadError: If upload fails
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Default credentials path
    if client_secrets_path is None:
        client_secrets_path = Path("credentials/client_secret.json")

    if tags is None:
        tags = []

    # Validate privacy setting
    if privacy not in ("private", "unlisted", "public"):
        raise YouTubeUploadError(
            f"Invalid privacy setting: {privacy}. "
            "Must be 'private', 'unlisted', or 'public'."
        )

    try:
        if show_progress:
            print("Authenticating with YouTube...")

        youtube = get_authenticated_service(client_secrets_path, token_path)

        # Video metadata
        body = {
            "snippet": {
                "title": title,
                "description": description,
                "tags": tags,
                "categoryId": category_id,
            },
            "status": {
                "privacyStatus": privacy,
                "selfDeclaredMadeForKids": False,
            },
        }

        # Create upload request
        media = MediaFileUpload(
            str(video_path),
            mimetype="video/mp4",
            resumable=True,
            chunksize=1024 * 1024,  # 1MB chunks
        )

        request = youtube.videos().insert(
            part="snippet,status",
            body=body,
            media_body=media,
        )

        if show_progress:
            print(f"Uploading: {video_path.name}")

        # Execute upload with retry logic
        response = None
        retry = 0

        while response is None:
            try:
                status, response = request.next_chunk()
                if status and show_progress:
                    progress = int(status.progress() * 100)
                    print(f"Upload progress: {progress}%")
            except HttpError as e:
                if e.resp.status in RETRIABLE_STATUS_CODES:
                    retry += 1
                    if retry > MAX_RETRIES:
                        raise YouTubeUploadError(
                            f"Upload failed after {MAX_RETRIES} retries: {e}"
                        )
                    sleep_time = 2 ** retry
                    if show_progress:
                        print(f"Retry {retry}/{MAX_RETRIES} in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    raise YouTubeUploadError(f"YouTube API error: {e}")

        video_id = response["id"]
        video_url = f"https://youtu.be/{video_id}"

        if show_progress:
            print(f"Upload complete!")
            print(f"Video ID: {video_id}")
            print(f"Video URL: {video_url}")

        return {
            "video_id": video_id,
            "url": video_url,
            "title": title,
            "privacy": privacy,
        }

    except YouTubeUploadError:
        raise
    except FileNotFoundError:
        raise
    except Exception as e:
        raise YouTubeUploadError(f"Upload failed: {e}")


def load_metadata_from_yaml(yaml_path: Path) -> Dict[str, Any]:
    """
    Load video metadata from descriptions YAML file.

    Args:
        yaml_path: Path to descriptions YAML file

    Returns:
        Dict with title, description, and tags

    Raises:
        YouTubeUploadError: If file cannot be loaded
    """
    import yaml

    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise YouTubeUploadError(f"Metadata file not found: {yaml_path}")

    try:
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        return {
            "title": data.get("youtube_title", ""),
            "description": data.get("youtube_description", ""),
            "tags": data.get("tags", []),
        }
    except Exception as e:
        raise YouTubeUploadError(f"Failed to load metadata: {e}")
