import datetime
import json
import os
import queue
import sys
import threading
import time

import keyboard
import sounddevice as sd
from vosk import Model, KaldiRecognizer

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 8000
HOTKEY = "ctrl+shift+s"
# Files that indicate a valid Vosk model directory
_MODEL_MARKERS = ("am", "conf", "graph", "ivector")


def _is_valid_model(path: str) -> bool:
    """Return True if *path* looks like a usable Vosk model directory."""
    return os.path.isdir(path) and any(
        os.path.isdir(os.path.join(path, m)) for m in _MODEL_MARKERS
    )


def _unwrap(path: str) -> str:
    """If the model is wrapped in an extra same-named subfolder, descend."""
    while not _is_valid_model(path):
        children = [
            os.path.join(path, d)
            for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d))
        ]
        if len(children) == 1:
            path = children[0]
        else:
            break
    return path


def get_model_path() -> str:
    # 1. CLI argument
    if len(sys.argv) > 1:
        p = _unwrap(sys.argv[1])
        if _is_valid_model(p):
            return p

    # 2. Environment variable
    env = os.environ.get("VOSK_MODEL_PATH", "").strip()
    if env and os.path.isdir(env):
        p = _unwrap(env)
        if _is_valid_model(p):
            return p

    # 3. Auto-detect: look for a model folder next to app.py
    app_dir = os.path.dirname(os.path.abspath(__file__))
    search_dirs = [app_dir, os.path.join(app_dir, "vosk")]
    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue
        for entry in sorted(os.listdir(search_dir)):
            candidate = os.path.join(search_dir, entry)
            if os.path.isdir(candidate) and entry.startswith("vosk-model"):
                candidate = _unwrap(candidate)
                if _is_valid_model(candidate):
                    print(f"Auto-detected model: {candidate}")
                    return candidate

    print("ERROR: No Vosk model found. Provide the path as:")
    print("  python app.py <model_folder>")
    print("  or  $env:VOSK_MODEL_PATH = 'C:\\path\\to\\model'  (PowerShell)")
    print("  or  set VOSK_MODEL_PATH=C:\\path\\to\\model       (cmd)")
    sys.exit(1)


def ensure_output_dir() -> str:
    output_dir = os.path.join(os.getcwd(), "transcripts")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def new_transcript_path(output_dir: str) -> str:
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(output_dir, f"session_{stamp}.txt")


def format_time(seconds: float) -> str:
    whole = int(seconds)
    ms = int((seconds - whole) * 1000)
    return time.strftime("%H:%M:%S", time.gmtime(whole)) + f".{ms:03d}"


class Transcriber:
    def __init__(self, model_path: str) -> None:
        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, SAMPLE_RATE)
        self.audio_queue: queue.Queue[bytes] = queue.Queue()
        self.running = False
        self.output_dir = ensure_output_dir()
        self.output_path = new_transcript_path(self.output_dir)
        self.start_time: float | None = None
        self.writer = open(self.output_path, "w", encoding="utf-8")

    def close(self) -> None:
        self.writer.close()

    def reset_session(self) -> None:
        self.output_path = new_transcript_path(self.output_dir)
        self.writer = open(self.output_path, "w", encoding="utf-8")
        self.start_time = None

    def callback(self, indata, frames, time_info, status) -> None:
        if status:
            print(status, file=sys.stderr)
        if self.running:
            self.audio_queue.put(bytes(indata))

    def toggle(self) -> None:
        if self.running:
            self.running = False
            print("Stopped recording.")
        else:
            if self.start_time is None:
                self.start_time = time.time()
            self.running = True
            print("Started recording.")

    def process_audio(self) -> None:
        while True:
            data = self.audio_queue.get()
            if data is None:
                break
            if self.recognizer.AcceptWaveform(data):
                result = json.loads(self.recognizer.Result())
                text = result.get("text", "").strip()
                if text:
                    ts = self.timestamp()
                    line = f"[{ts}] {text}"
                    print(line)
                    self.writer.write(line + "\n")
                    self.writer.flush()

    def timestamp(self) -> str:
        if self.start_time is None:
            return "00:00:00.000"
        return format_time(time.time() - self.start_time)


def main() -> None:
    model_path = get_model_path()
    transcriber = Transcriber(model_path)

    print("Press Ctrl+Shift+S to start/stop recording.")
    print("Press Ctrl+C to quit.")
    keyboard.add_hotkey(HOTKEY, transcriber.toggle)

    worker = threading.Thread(target=transcriber.process_audio, daemon=True)
    worker.start()

    try:
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            dtype="int16",
            channels=CHANNELS,
            callback=transcriber.callback,
        ):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        transcriber.audio_queue.put(None)
        transcriber.close()


if __name__ == "__main__":
    main()
