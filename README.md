# Audio Tracker

Offline transcription for Windows — speech-to-text from mic and ASL fingerspelling from camera.

## Features
- **Voice mode**: Live speech transcription from mic (Vosk, offline)
- **ASL mode**: Camera-based ASL fingerspelling recognition (MediaPipe + scikit-learn)
- Timestamped segments
- Save transcripts to file
- Start/stop hotkey (Ctrl+Shift+S) for voice mode

## Setup
1. Install Python 3.10+.
2. Create a virtual environment.
3. Install dependencies:

   pip install -r requirements.txt

### Voice Transcription (Vosk)
4. Download a Vosk English model and unzip it:
   https://alphacephei.com/vosk/models

   Suggested model: vosk-model-small-en-us-0.15

5. The app auto-detects models placed next to `app.py` or in a `vosk/` subfolder.
   You can also set the path explicitly:

   PowerShell:  $env:VOSK_MODEL_PATH = "C:\path\to\vosk-model-small-en-us-0.15"
   cmd:         set VOSK_MODEL_PATH=C:\path\to\vosk-model-small-en-us-0.15

### ASL Fingerspelling (Camera)
6. Train the ASL model on your own hand signs:

   python asl_trainer.py

   - A camera window opens.
   - Press a letter key (a-y, skipping j) and hold the corresponding ASL
     sign in front of the camera. Samples are captured each frame.
   - Record at least 50-100 samples per letter for best results.
   - Press **t** to train the model (saved to `asl_model.pkl`).
   - Press **q** to quit.

### Auto-harvest from YouTube (optional)
7. Instead of recording yourself, you can harvest training data from
   ASL teaching videos (e.g. Dr. Bill Vicars' @aslu channel):

   ```
   # Preview what would be downloaded:
   python asl_harvester.py --url "https://www.youtube.com/@aslu/videos" --dry-run

   # Harvest up to 50 videos:
   python asl_harvester.py --url "https://www.youtube.com/@aslu/videos" --max-videos 50

   # Harvest a single video with a specific label:
   python asl_harvester.py --url "https://www.youtube.com/watch?v=VIDEO_ID" --label HELLO

   # Check what you've collected:
   python asl_harvester.py --stats

   # Train the model on harvested data:
   python asl_harvester.py --train
   ```

   The harvester downloads each video, runs MediaPipe hand detection on
   frames, and saves normalized landmarks labeled by the video title.
   You can combine harvested data with your own recordings from `asl_trainer.py`.

## Run

### Voice transcription
python app.py

### ASL transcription
python asl_app.py

## Controls

### Voice mode
- Ctrl+Shift+S: toggle recording
- Ctrl+C: quit

### ASL mode (camera window)
- Hold an ASL sign steady for ~0.8s to register a letter
- Space: insert word break
- Backspace: delete last character
- Enter: new line
- S: save transcript
- Q / Esc: quit

## Output
Transcripts are saved to:
- transcripts\session_YYYYMMDD_HHMMSS.txt (voice)
- transcripts\asl_session_YYYYMMDD_HHMMSS.txt (ASL)
