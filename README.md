# Audio Tracker

Offline real-time translator for Windows — spoken English (mic) → text, and ASL signs (camera/video) → text. (WIP)

## Features
- **Voice mode**: Live speech transcription from mic using Vosk (fully offline)
- **ASL mode** (WIP): Word-level ASL sign recognition from camera or video files
  - MediaPipe holistic landmarks (hands + pose + face)
  - 345-feature vectors per frame (21×3 per hand + 33×3 pose + 40×3 face)
  - Sequence-based classification with rolling frame buffer
- Timestamped transcript output
- Save transcripts to file
- Start/stop hotkey (Ctrl+Shift+S) for voice mode

## Requirements
- Python 3.10+
- Windows (tested), macOS/Linux should work
- Webcam (for ASL camera mode)
- ~1.5 GB disk space (for WLASL dataset + models)
- yt-dlp (for WLASL video downloading)

## Setup

```bash
git clone <repo-url>
cd "Audio tracker"
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### Voice Transcription (Vosk)
Download a Vosk English model and unzip it:
https://alphacephei.com/vosk/models

Suggested: `vosk-model-small-en-us-0.15`

The app auto-detects models placed next to `app.py` or in a `vosk/` subfolder.
You can also set the path explicitly:

```powershell
$env:VOSK_MODEL_PATH = "C:\path\to\vosk-model-small-en-us-0.15"
```

### ASL Recognition

#### 1. Build the training dataset (WLASL)

MediaPipe model files are auto-downloaded on first run.

```bash
# Download WLASL100 videos + extract landmark sequences (full pipeline)
python dataset_builder.py

# Or with more glosses
python dataset_builder.py --num-glosses 300

# Resume an interrupted download
python dataset_builder.py --num-glosses 100

# Re-process already-downloaded videos (skip downloading)
python dataset_builder.py --process-only

# Download videos only (skip processing)
python dataset_builder.py --download-only
```

Output goes to `wlasl_data/`:
- `raw_videos/` — downloaded source clips
- `processed/train.npz`, `val.npz`, `test.npz` — landmark sequences
- `processed/glosses.json` — label index → sign name

#### 2. Test the feature extractor

```bash
python feature_extractor.py
```

Opens a camera window with real-time skeleton overlay showing:
- Green: upper body pose (shoulders, arms, hips)
- Cyan/Orange: hand skeletons (right/left, 21 joints each)
- Magenta: face landmarks (eyebrows, eyes, lips, jaw)
- HUD: detection status dots, frame/sequence counters, buffer progress bar

Press **Q** to quit.

#### 3. Supplemental data from YouTube (optional)

```bash
# Harvest ASL teaching videos (e.g. @aslu channel)
python asl_harvester.py --url "https://www.youtube.com/@aslu/videos" --max-videos 50

# Single video with explicit label
python asl_harvester.py --url "https://www.youtube.com/watch?v=VIDEO_ID" --label HELLO

# Check collected data
python asl_harvester.py --stats
```

## Run

```bash
# Voice transcription
python app.py

# Feature extractor test (camera skeleton overlay)
python feature_extractor.py
```

## Controls

### Voice mode
- **Ctrl+Shift+S**: toggle recording
- **Ctrl+C**: quit

### Feature extractor / ASL camera
- **Q**: quit

## Project Structure

| File | Purpose |
|------|---------|
| `app.py` | Voice-to-text (mic → Vosk → transcript) |
| `feature_extractor.py` | MediaPipe holistic → 345-dim landmark sequences |
| `dataset_builder.py` | WLASL dataset download + landmark processing |
| `asl_harvester.py` | YouTube ASL video data scraper |
| `asl_model.py` | LSTM model definition (planned) |
| `train.py` | Training loop + evaluation (planned) |
| `asl_app.py` | Live ASL inference (planned) |
| `main.py` | Unified launcher + GUI (planned) |

## Architecture

```
Camera frame
  → MediaPipe (hands + pose + face landmarks)
  → 345 features/frame (normalized to shoulder-center, scaled by torso height)
  → rolling sequence buffer (60 frames)
  → LSTM classifier (planned)
  → sign label

Mic audio
  → Vosk (offline speech recognition)
  → timestamped text
```

## Output
Transcripts are saved to `transcripts/` with timestamps.
