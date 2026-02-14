"""Pipeline state management for resumable processing."""

from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

STEP_NAMES = ["download", "extract", "transcribe", "summarize", "generate_video"]


def get_episode_id_from_url(url: str) -> str:
    """Extract episode ID from a Clubhouse URL.

    Takes the last non-empty path segment and strips query params and file extensions.
    e.g. "https://www.clubhouse.com/room/19204125?foo=bar" -> "19204125"
         "https://...s3.amazonaws.com/.../19453826.mp4?..." -> "19453826"
    """
    parsed = urlparse(url)
    path_parts = parsed.path.strip("/").split("/")
    for part in reversed(path_parts):
        segment = part.split("?")[0]
        if segment:
            # Strip file extension (e.g. ".mp4") to get the raw ID
            stem = Path(segment).stem
            return stem if stem else segment
    raise ValueError(f"Cannot extract episode ID from URL: {url}")


class PipelineState:
    """Tracks pipeline step completion via output file existence."""

    def __init__(self, episode_id: str, output_base: Path):
        self.episode_id = episode_id
        self.output_base = Path(output_base)

    @property
    def video_path(self) -> Path:
        return self.output_base / "audio" / f"{self.episode_id}.mp4"

    @property
    def audio_path(self) -> Path:
        return self.output_base / "audio" / f"{self.episode_id}.mp3"

    @property
    def transcript_path(self) -> Path:
        return self.output_base / "transcripts" / f"{self.episode_id}_transcript.txt"

    @property
    def descriptions_path(self) -> Path:
        return self.output_base / "descriptions" / f"{self.episode_id}_descriptions.yaml"

    @property
    def generated_video_path(self) -> Path:
        return self.output_base / "videos" / f"{self.episode_id}_video.mp4"

    def _step_file(self, step: str) -> Path:
        return {
            "download": self.video_path,
            "extract": self.audio_path,
            "transcribe": self.transcript_path,
            "summarize": self.descriptions_path,
            "generate_video": self.generated_video_path,
        }[step]

    def get_completed_steps(self) -> List[str]:
        """Return list of step names whose output files exist."""
        return [step for step in STEP_NAMES if self._step_file(step).exists()]

    def get_next_step(self) -> Optional[str]:
        """Return the first incomplete step, or None if all done."""
        for step in STEP_NAMES:
            if not self._step_file(step).exists():
                return step
        return None
