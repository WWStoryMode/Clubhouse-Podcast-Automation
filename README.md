# Clubhouse-Podcast-Automation

An open-source tool to automate publishing Clubhouse recordings to Spotify and YouTube. Supports both **local** and **cloud** deployment from a single codebase.

## Features

- **Download** Clubhouse recordings from dynamic links
- **Extract** audio track from video (MP4 → MP3) with optional trimming and loudness normalization
- **Transcribe** audio using Gemini API (supports chunked transcription for long recordings)
- **Generate descriptions** for YouTube and Spotify with auto-generated tags
- **Create video** with waveform visualization using ffmpeg (hardware-accelerated encoding)
- **Upload to YouTube** with metadata, privacy settings, and playlist support
- **Full pipeline** with `--resume` (skip completed steps) and `--interactive` (review each step)

## Deployment Options

| Mode | Best For | Infrastructure | Status |
|------|----------|----------------|--------|
| **Local** | Individual creators, development | Your machine | Available |
| **Cloud** | Teams, automation, scheduled jobs | GCP Cloud Run Jobs | *To be implemented* |

## Quick Start (Local)

### Prerequisites

- Python 3.10+
- ffmpeg (`brew install ffmpeg` on macOS)
- Gemini API key ([Get one here](https://aistudio.google.com/apikey))
- YouTube API credentials (for uploads — see [YouTube Setup](#youtube-upload-setup) below)

### Installation

```bash
# Clone the repository
git clone https://github.com/WWStoryMode/Clubhouse-Podcast-Automation.git
cd Clubhouse-Podcast-Automation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy configuration templates
cp config/config.example.yaml config/config.yaml
cp .env.example .env

# Edit .env with your API keys
```

### Configuration

1. Edit `.env` with your API keys:
   ```
   GEMINI_API_KEY=your_gemini_api_key
   ```

2. Edit `config/config.yaml` with your settings (see `config/config.example.yaml` for all options)

3. Add your template assets to `templates/`:
   - `background.png` — Video background image (1920x1080 recommended)
   - `logo.png` — Logo/icon overlay
   - `video_config.yaml` — Video layout configuration (optional)

   Placeholder files are included and used as fallbacks if custom assets are not provided.

### Usage

#### Full pipeline (one command)

```bash
python -m src.cli process -u "https://clubhouse.com/room/12345" -t "Episode Title"
```

The pipeline runs 5 steps: **download → extract audio → transcribe → generate descriptions → generate video**.

#### Resume an interrupted run

```bash
python -m src.cli process -u "URL" -t "Title" --resume
```

Skips steps whose output files already exist.

#### Interactive mode (review each step)

```bash
python -m src.cli process -u "URL" -t "Title" --interactive
```

Opens each step's output in Finder for review before continuing.

#### Individual steps

```bash
# Download recording
python -m src.cli download -u "https://clubhouse.com/room/12345"

# Extract audio (with optional trimming and normalization)
python -m src.cli extract -i output/audio/12345.mp4
python -m src.cli extract -i output/audio/12345.mp4 --trim-start 00:01:00 --trim-end 01:30:00 --no-normalize

# Transcribe (supports language selection and chunked mode for long files)
python -m src.cli transcribe -i output/audio/12345.mp3
python -m src.cli transcribe -i output/audio/12345.mp3 -l yue --chunked --model gemini-2.5-pro

# Generate descriptions
python -m src.cli summarize -i output/transcripts/12345_transcript.txt -t "Episode Title"

# Generate video with waveform visualization
python -m src.cli generate-video -i output/audio/12345.mp3 -t "Episode Title"
python -m src.cli generate-video --list-encoders  # Show available hardware encoders

# Preview video frame layout (without rendering full video)
python -m src.cli preview-frame -t "Episode Title"

# Upload to YouTube
python -m src.cli upload-youtube -v output/videos/12345_video.mp4 -t "Episode Title"
python -m src.cli upload-youtube -v output/videos/12345_video.mp4 -m output/descriptions/12345_descriptions.yaml
```

### YouTube Upload Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project and enable **YouTube Data API v3**
3. Go to Credentials → Create Credentials → **OAuth 2.0 Client ID** (Desktop app)
4. Download the JSON and save as `credentials/client_secret.json`
5. The first upload will open a browser for authorization (token cached in `credentials/youtube_token.pickle`)

## Cloud Deployment (GCP)

> **Status:** To be implemented
>
> Cloud deployment via GCP Cloud Run Jobs is planned for a future release. This will enable:
> - Scheduled/automated processing
> - Team collaboration
> - Webhook triggers
> - Cloud storage integration

## Project Structure

```
Clubhouse-Podcast-Automation/
├── src/
│   ├── cli.py                    # CLI entry point (all commands)
│   └── core/                     # Business logic
│       ├── pipeline.py           # Pipeline state & resume logic
│       ├── downloader.py         # Download from Clubhouse
│       ├── audio_extractor.py    # MP4 → MP3 with trimming & normalization
│       ├── transcriber.py        # Gemini API transcription
│       ├── summarizer.py         # Gemini API description generation
│       ├── video_generator.py    # Waveform video via ffmpeg
│       └── youtube_uploader.py   # YouTube Data API v3
│
├── config/                       # Configuration
│   └── config.example.yaml       # Template (copy to config.yaml)
├── credentials/                  # OAuth credentials (gitignored)
│   └── client_secret.json        # YouTube API OAuth client
├── templates/                    # Video template assets
│   ├── background.png            # Video background (1920x1080)
│   ├── logo.png                  # Logo overlay
│   ├── video_config.yaml         # Video layout config
│   └── fonts/                    # Custom fonts
├── output/                       # Generated files
│   ├── audio/                    # Downloaded video + extracted MP3
│   ├── transcripts/              # Text transcripts
│   ├── descriptions/             # YouTube/Spotify descriptions (YAML)
│   └── videos/                   # Generated videos
└── tests/                        # Test suite
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| CLI | Click |
| Audio extraction | ffmpeg |
| Transcription | Gemini API (google-genai SDK) |
| Summarization | Gemini API (google-genai SDK) |
| Video generation | ffmpeg (with hardware acceleration) |
| Audio analysis | librosa |
| YouTube upload | YouTube Data API v3 |
| Cloud deployment | GCP Cloud Run Jobs *(planned)* |

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.
