"""Live ASL fingerspelling transcription from camera.

Reads the camera feed, detects hand landmarks via MediaPipe,
classifies static ASL letters, and assembles them into words.

Controls (in the camera window):
  Space   : insert a space (finish current word)
  Backspace : delete last character
  Enter   : new line
  S       : save transcript to file
  Q / Esc : quit
"""

from __future__ import annotations

import datetime
import os
import time

import cv2
import numpy as np

from asl_recognizer import ASLRecognizer

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "asl_model.pkl")
TRANSCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transcripts")

# --- Timing parameters for letter buffering ---
HOLD_TIME = 0.8          # seconds a letter must be stable before it's accepted
COOLDOWN = 0.4            # seconds after accepting before next letter can start
DUPLICATE_PAUSE = 1.2     # extra pause required to repeat the same letter
WORD_TIMEOUT = 3.0        # seconds of no detection → auto-insert space


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


class TranscriptBuffer:
    """Maintains the running transcript with word-building logic."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.current_word: str = ""
        self.start_time: float = time.time()

    def add_letter(self, letter: str) -> None:
        self.current_word += letter

    def add_space(self) -> None:
        if self.current_word:
            self.lines.append(self.current_word)
            self.current_word = ""

    def backspace(self) -> None:
        if self.current_word:
            self.current_word = self.current_word[:-1]
        elif self.lines:
            self.current_word = self.lines.pop()

    def newline(self) -> None:
        self.add_space()
        self.lines.append("")

    def get_display_text(self, max_lines: int = 6) -> list[str]:
        """Return recent transcript lines for on-screen display."""
        full = " ".join(self.lines)
        if self.current_word:
            full += " " + self.current_word + "_"
        # Wrap to ~60 chars per line
        words = full.split()
        display_lines: list[str] = [""]
        for w in words:
            if len(display_lines[-1]) + len(w) + 1 > 60:
                display_lines.append("")
            if display_lines[-1]:
                display_lines[-1] += " "
            display_lines[-1] += w
        return display_lines[-max_lines:]

    def get_full_text(self) -> str:
        parts = list(self.lines)
        if self.current_word:
            parts.append(self.current_word)
        return " ".join(parts)

    def timestamp(self) -> str:
        elapsed = time.time() - self.start_time
        whole = int(elapsed)
        ms = int((elapsed - whole) * 1000)
        return time.strftime("%H:%M:%S", time.gmtime(whole)) + f".{ms:03d}"


def save_transcript(buf: TranscriptBuffer) -> str:
    ensure_dir(TRANSCRIPT_DIR)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(TRANSCRIPT_DIR, f"asl_session_{stamp}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"ASL Transcription — {stamp}\n")
        f.write("=" * 40 + "\n\n")
        f.write(buf.get_full_text() + "\n")
    return path


def main() -> None:
    recognizer = ASLRecognizer(model_path=MODEL_PATH, confidence=0.55)
    buf = TranscriptBuffer()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        return

    print("=== ASL Live Transcription ===")
    print("Controls: Space=word break | Backspace=delete | S=save | Q=quit")

    # State for letter stabilization
    candidate_letter: str | None = None
    candidate_start: float = 0.0
    last_accepted: str | None = None
    last_accepted_time: float = 0.0
    last_detection_time: float = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror for natural interaction
        annotated, predictions = recognizer.process_frame(frame)

        now = time.time()

        if predictions:
            letter, conf = predictions[0]  # Use first (dominant) hand
            last_detection_time = now

            if letter == candidate_letter:
                # Same letter held — check if hold time reached
                held_time = now - candidate_start
                if held_time >= HOLD_TIME and (now - last_accepted_time) >= COOLDOWN:
                    # Extra check: same letter as last accepted needs longer pause
                    if letter == last_accepted and (now - last_accepted_time) < DUPLICATE_PAUSE:
                        pass  # Wait longer for duplicate
                    else:
                        buf.add_letter(letter)
                        print(f"[{buf.timestamp()}]  {letter}  (conf={conf:.0%})")
                        last_accepted = letter
                        last_accepted_time = now
                        candidate_letter = None
            else:
                # New letter candidate
                candidate_letter = letter
                candidate_start = now

            # Draw detection info
            cv2.putText(annotated, f"Detected: {letter} ({conf:.0%})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            # Progress bar for hold time
            if candidate_letter:
                progress = min(1.0, (now - candidate_start) / HOLD_TIME)
                bar_w = int(200 * progress)
                cv2.rectangle(annotated, (10, 45), (10 + bar_w, 55), (0, 255, 0), -1)
                cv2.rectangle(annotated, (10, 45), (210, 55), (100, 100, 100), 1)
        else:
            candidate_letter = None
            # Auto-space on timeout
            if buf.current_word and (now - last_detection_time) > WORD_TIMEOUT:
                buf.add_space()
                print(f"[{buf.timestamp()}]  <space>")

        # Draw transcript overlay
        display_lines = buf.get_display_text()
        overlay_y = annotated.shape[0] - 30 * len(display_lines) - 10
        cv2.rectangle(annotated, (0, overlay_y - 10), (annotated.shape[1], annotated.shape[0]),
                      (0, 0, 0), -1)
        for i, line in enumerate(display_lines):
            y = overlay_y + i * 30
            cv2.putText(annotated, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("ASL Transcription", annotated)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):  # q or Esc
            break
        elif key == ord(" "):
            buf.add_space()
            print(f"[{buf.timestamp()}]  <space>")
        elif key == 8:  # Backspace
            buf.backspace()
        elif key == 13:  # Enter
            buf.newline()
        elif key == ord("s"):
            path = save_transcript(buf)
            print(f"Transcript saved to {path}")

    # Auto-save on exit
    if buf.get_full_text().strip():
        path = save_transcript(buf)
        print(f"Transcript saved to {path}")

    cap.release()
    cv2.destroyAllWindows()
    recognizer.close()


if __name__ == "__main__":
    main()
