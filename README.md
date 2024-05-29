# Snakeman CLI

Python script that processes a video, extracts frames, generates descriptions for those frames, summarizes the descriptions, and creates a final video with text-to-speech (TTS) audio overlayed on a rap beat.

## Features

- Downloads a YouTube video.
- Extracts frames from the video.
- Generates frame descriptions using an image-to-text model.
- Summarizes the descriptions.
- Generates TTS audio for the summary.
- Combines the TTS audio with a rap beat.
- Creates a final video clip with the TTS audio overlayed.

## Installation

### Prerequisites

- Python 3.6 or higher
- pip (Python package installer)

### Install Dependencies

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  

# Install dependencies using pip
pip install yt-dlp opencv-python pytesseract Pillow gtts ffmpeg-python pydub transformers torch torchvision torchaudio

# For Arch Linux
sudo pacman -S python-pip opencv tesseract
pip install yt-dlp opencv-python pytesseract Pillow gtts ffmpeg-python pydub transformers torch torchvision torchaudio

# For Gentoo
sudo emerge dev-python/pip opencv tesseract
pip install yt-dlp opencv-python pytesseract Pillow gtts ffmpeg-python pydub transformers torch torchvision torchaudio
```

Usage
```bash
Copy code
# Clone the repository
git clone https://github.com/yourusername/snakeman-video-processing.git
cd snakeman-video-processing

# Activate the virtual environment
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Run the script
python snakeman.py

# Follow the prompts to enter the YouTube video URL and a brief description of the video. If left blank will default to video title
```
