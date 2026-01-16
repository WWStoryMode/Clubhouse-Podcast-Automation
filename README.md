# Clubhouse-Podcast-Automation

An open-source tool to automate publishing Clubhouse recordings to Spotify and YouTube. Supports both **local** and **cloud** deployment from a single codebase.

## Features

- **Download** Clubhouse recordings from dynamic links
- **Extract** audio track from video (MP4 to MP3)
- **Transcribe** audio using Gemini API (Google AI Studio)
- **Generate descriptions** for YouTube and Spotify
- **Create video** with waveform visualization using MoviePy
- **Upload to YouTube** with scheduled publishing
- **Prepare files** for Spotify manual upload

## Deployment Options

| Mode | Best For | Infrastructure |
|------|----------|----------------|
| **Local** | Individual creators, development | Your machine |
| **Cloud** | Teams, automation, scheduled jobs | GCP Cloud Run Jobs |

## Quick Start (Local)

### Prerequisites

- Python 3.10+
- ffmpeg (`brew install ffmpeg` on macOS)
- Gemini API key ([Get one here](https://aistudio.google.com/apikey))
- YouTube API credentials ([Setup guide](docs/api-setup.md))

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/Clubhouse-Podcast-Automation.git
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

2. Edit `config/config.yaml` with your settings

3. Add your template assets to `templates/`:
   - `background.png` - Video background image (1920x1080)
   - `icon.png` - Logo/icon overlay

### Usage

```bash
# Full pipeline
python -m src.cli process --url "https://clubhouse.com/..." --title "Episode 1"

# Individual steps
python -m src.cli download --url "https://clubhouse.com/..."
python -m src.cli extract --input output/audio/raw.mp4
python -m src.cli transcribe --input output/audio/audio.mp3
python -m src.cli summarize --input output/transcripts/transcript.txt
python -m src.cli generate-video --input output/audio/audio.mp3 --title "Episode 1"
python -m src.cli upload-youtube --input output/videos/video.mp4
```

## Cloud Deployment (GCP)

See [Cloud Deployment Guide](docs/cloud-deployment.md) for detailed instructions.

```bash
# Quick start
gcloud run jobs execute podcast-automation \
  --args="process,--url,https://clubhouse.com/...,--title,Episode 1"
```

## Documentation

- [Getting Started](docs/getting-started.md)
- [Local Setup](docs/local-setup.md)
- [Cloud Deployment](docs/cloud-deployment.md)
- [Configuration](docs/configuration.md)
- [API Setup](docs/api-setup.md)

## Project Structure

```
Clubhouse-Podcast-Automation/
├── src/
│   ├── core/                 # Business logic
│   │   ├── pipeline.py       # Orchestration
│   │   ├── downloader.py     # Download from Clubhouse
│   │   ├── audio_extractor.py
│   │   ├── transcriber.py    # Gemini API
│   │   ├── summarizer.py     # Gemini API
│   │   ├── video_generator.py # MoviePy
│   │   └── uploader.py       # YouTube API
│   │
│   ├── adapters/             # Platform adapters
│   │   ├── storage/          # Local/Cloud storage
│   │   └── runtime/          # Local/Cloud execution
│   │
│   └── cli.py                # CLI entry point
│
├── config/                   # Configuration
├── templates/                # Video templates
├── output/                   # Generated files
└── deploy/                   # Deployment configs
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| Audio extraction | ffmpeg |
| Transcription | Gemini API |
| Summarization | Gemini API |
| Video generation | MoviePy + librosa |
| YouTube upload | YouTube Data API v3 |
| Cloud deployment | GCP Cloud Run Jobs |

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.
