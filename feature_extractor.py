"""Holistic feature extraction for ASL sign recognition.

Uses MediaPipe Tasks API to extract hand and pose landmarks from video frames.
Produces a per-frame feature vector of 225 dimensions:
  - Left hand:  21 landmarks × 3 coords (x, y, z) =  63
  - Right hand: 21 landmarks × 3 coords (x, y, z) =  63
  - Pose:       33 landmarks × 3 coords (x, y, z) =  99
                                              Total: 225

Landmarks are normalized relative to the mid-shoulder point and scaled by
torso height for signer-invariant features.

Also provides a SequenceBuffer that accumulates frames into fixed-length
sequences suitable for temporal model input.
"""

from __future__ import annotations

import os
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Suppress TF/MediaPipe log noise
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarkerResult,
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarkerResult,
    RunningMode,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELS_DIR = Path(__file__).resolve().parent / "models"

HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"

HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
POSE_MODEL_PATH = MODELS_DIR / "pose_landmarker_lite.task"

NUM_HAND_LANDMARKS = 21
NUM_POSE_LANDMARKS = 33
HAND_FEATURES = NUM_HAND_LANDMARKS * 3   # 63
POSE_FEATURES = NUM_POSE_LANDMARKS * 3   # 99
FRAME_FEATURES = HAND_FEATURES * 2 + POSE_FEATURES  # 225

# Pose landmark indices for normalization
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24

DEFAULT_SEQ_LEN = 30  # frames

# Hand skeleton connections (21 landmarks)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # Index
    (5, 9), (9, 10), (10, 11), (11, 12),     # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (0, 17),                                  # Palm base
]

# Pose connections (upper body — relevant for ASL)
POSE_UPPER_CONNECTIONS = [
    (11, 12),              # Shoulders
    (11, 13), (13, 15),    # Left arm
    (12, 14), (14, 16),    # Right arm
    (11, 23), (12, 24),    # Torso sides
    (23, 24),              # Hips
]

# Pose landmark indices to draw as joints
POSE_DRAW_INDICES = {11, 12, 13, 14, 15, 16, 23, 24}


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------

def _download_model(url: str, dest: Path) -> None:
    """Download a model file if it doesn't exist yet."""
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dest.name} ...")
    urllib.request.urlretrieve(url, str(dest))
    print(f"  Saved to {dest}")


def ensure_models() -> None:
    """Download hand and pose model files if not already present."""
    _download_model(HAND_MODEL_URL, HAND_MODEL_PATH)
    _download_model(POSE_MODEL_URL, POSE_MODEL_PATH)


# ---------------------------------------------------------------------------
# Landmark extraction helpers
# ---------------------------------------------------------------------------

def _hand_landmarks_to_array(landmarks) -> np.ndarray:
    """Convert a list of 21 hand landmarks to shape (21, 3)."""
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)


def _pose_landmarks_to_array(landmarks) -> np.ndarray:
    """Convert a list of 33 pose landmarks to shape (33, 3)."""
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)


def _normalize_frame(
    left_hand: np.ndarray,
    right_hand: np.ndarray,
    pose: np.ndarray,
) -> np.ndarray:
    """Normalize landmarks and flatten to a (225,) feature vector.

    Normalization:
    1. Translate all landmarks so mid-shoulder is the origin.
    2. Scale so torso height (mid-shoulder to mid-hip) = 1.
    """
    # Mid-shoulder and mid-hip from pose
    mid_shoulder = (pose[LEFT_SHOULDER] + pose[RIGHT_SHOULDER]) / 2.0
    mid_hip = (pose[LEFT_HIP] + pose[RIGHT_HIP]) / 2.0
    torso_height = np.linalg.norm(mid_hip - mid_shoulder)
    if torso_height < 1e-6:
        torso_height = 1.0  # fallback to avoid division by zero

    # Translate
    left_hand = left_hand - mid_shoulder
    right_hand = right_hand - mid_shoulder
    pose = pose - mid_shoulder

    # Scale
    left_hand /= torso_height
    right_hand /= torso_height
    pose /= torso_height

    return np.concatenate([
        left_hand.flatten(),
        right_hand.flatten(),
        pose.flatten(),
    ])


# ---------------------------------------------------------------------------
# ExtractionResult
# ---------------------------------------------------------------------------

@dataclass
class ExtractionResult:
    """Carries both the normalized feature vector and raw landmarks for drawing."""
    features: np.ndarray | None = None        # (225,) or None if no pose
    pose_landmarks: list | None = None        # raw MediaPipe pose landmark list
    hand_landmarks: list = field(default_factory=list)  # raw hand landmark lists
    handedness: list = field(default_factory=list)       # handedness per detected hand
    left_detected: bool = False
    right_detected: bool = False
    pose_detected: bool = False


# ---------------------------------------------------------------------------
# FeatureExtractor
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """Extract holistic landmarks from frames using MediaPipe Tasks API.

    Parameters
    ----------
    mode : str
        "image" for single images / offline video processing,
        "video" for sequential video frames with timestamps,
        "live" for live camera stream.
    """

    def __init__(self, mode: str = "video") -> None:
        ensure_models()

        running_mode = {
            "image": RunningMode.IMAGE,
            "video": RunningMode.VIDEO,
            "live": RunningMode.IMAGE,  # we'll call IMAGE per-frame for simplicity
        }[mode]

        self._hand_landmarker = HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
                running_mode=running_mode,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )

        self._pose_landmarker = PoseLandmarker.create_from_options(
            PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
                running_mode=running_mode,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )

        self._mode = mode

    def _detect(self, frame_rgb: np.ndarray, timestamp_ms: int = 0):
        """Run MediaPipe pose + hand detection on a single frame."""
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb,
        )

        if self._mode == "video":
            pose_result = self._pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            hand_result = self._hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        else:
            pose_result = self._pose_landmarker.detect(mp_image)
            hand_result = self._hand_landmarker.detect(mp_image)

        return pose_result, hand_result

    @staticmethod
    def _sort_hands(hand_result):
        """Separate hand landmarks into left/right arrays + detection flags."""
        left_hand = np.zeros((NUM_HAND_LANDMARKS, 3), dtype=np.float32)
        right_hand = np.zeros((NUM_HAND_LANDMARKS, 3), dtype=np.float32)
        left_detected = right_detected = False

        if hand_result.hand_landmarks:
            for i, hand_lms in enumerate(hand_result.hand_landmarks):
                handedness = hand_result.handedness[i][0]
                arr = _hand_landmarks_to_array(hand_lms)
                # MediaPipe reports handedness from camera's perspective (mirrored)
                if handedness.category_name == "Left":
                    right_hand = arr
                    right_detected = True
                else:
                    left_hand = arr
                    left_detected = True

        return left_hand, right_hand, left_detected, right_detected

    def extract(
        self,
        frame_rgb: np.ndarray,
        timestamp_ms: int = 0,
    ) -> np.ndarray | None:
        """Extract a 225-dim feature vector from a single RGB frame.

        Returns (225,) feature vector, or None if pose not detected.
        """
        pose_result, hand_result = self._detect(frame_rgb, timestamp_ms)

        if not pose_result.pose_landmarks:
            return None

        pose = _pose_landmarks_to_array(pose_result.pose_landmarks[0])
        left_hand, right_hand, _, _ = self._sort_hands(hand_result)
        return _normalize_frame(left_hand, right_hand, pose)

    def extract_with_landmarks(
        self,
        frame_rgb: np.ndarray,
        timestamp_ms: int = 0,
    ) -> ExtractionResult:
        """Extract features AND return raw landmarks for visualization.

        Always returns an ExtractionResult; features is None if no pose detected.
        """
        pose_result, hand_result = self._detect(frame_rgb, timestamp_ms)

        raw_hands = list(hand_result.hand_landmarks) if hand_result.hand_landmarks else []
        raw_handedness = list(hand_result.handedness) if hand_result.handedness else []

        if not pose_result.pose_landmarks:
            return ExtractionResult(
                hand_landmarks=raw_hands,
                handedness=raw_handedness,
            )

        pose = _pose_landmarks_to_array(pose_result.pose_landmarks[0])
        left_hand, right_hand, left_det, right_det = self._sort_hands(hand_result)

        return ExtractionResult(
            features=_normalize_frame(left_hand, right_hand, pose),
            pose_landmarks=pose_result.pose_landmarks[0],
            hand_landmarks=raw_hands,
            handedness=raw_handedness,
            left_detected=left_det,
            right_detected=right_det,
            pose_detected=True,
        )

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._hand_landmarker.close()
        self._pose_landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# SequenceBuffer
# ---------------------------------------------------------------------------

class SequenceBuffer:
    """Accumulate per-frame features into fixed-length sequences.

    Parameters
    ----------
    seq_len : int
        Number of frames per sequence.
    overlap : int
        Number of frames shared between consecutive sequences.
        E.g. seq_len=30, overlap=15 means a new sequence is ready
        every 15 frames.
    """

    def __init__(self, seq_len: int = DEFAULT_SEQ_LEN, overlap: int = 15) -> None:
        self.seq_len = seq_len
        self.overlap = overlap
        self._step = seq_len - overlap
        self._buffer: deque[np.ndarray] = deque(maxlen=seq_len)
        self._frames_since_emit = 0

    def add_frame(self, features: np.ndarray) -> np.ndarray | None:
        """Add a frame's features and return a sequence if ready.

        Parameters
        ----------
        features : np.ndarray
            Shape (225,) feature vector for one frame.

        Returns
        -------
        np.ndarray or None
            Shape (seq_len, 225) when a full sequence is ready, else None.
        """
        self._buffer.append(features)
        self._frames_since_emit += 1

        if len(self._buffer) == self.seq_len and self._frames_since_emit >= self._step:
            self._frames_since_emit = 0
            return np.array(self._buffer, dtype=np.float32)
        return None

    def get_current(self) -> np.ndarray | None:
        """Get the current buffer as a padded sequence (may be < seq_len frames).

        Returns None if buffer is empty.
        """
        if not self._buffer:
            return None
        arr = np.array(self._buffer, dtype=np.float32)
        if len(arr) < self.seq_len:
            pad = np.zeros((self.seq_len - len(arr), FRAME_FEATURES), dtype=np.float32)
            arr = np.concatenate([pad, arr], axis=0)
        return arr

    def reset(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()
        self._frames_since_emit = 0


# ---------------------------------------------------------------------------
# Convenience: extract sequences from a video file
# ---------------------------------------------------------------------------

def extract_sequences_from_video(
    video_path: str,
    seq_len: int = DEFAULT_SEQ_LEN,
    target_fps: int = 30,
) -> tuple[list[np.ndarray], float]:
    """Extract landmark sequences from a video file.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    seq_len : int
        Number of frames per sequence.
    target_fps : int
        Resample video to this FPS before extraction.

    Returns
    -------
    sequences : list of np.ndarray
        Each array has shape (seq_len, 225).
    video_fps : float
        Original FPS of the video.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, round(video_fps / target_fps))

    extractor = FeatureExtractor(mode="video")
    frames: list[np.ndarray] = []
    frame_idx = 0
    timestamp_ms = 0
    frame_duration_ms = int(1000 / target_fps)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % frame_interval != 0:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        features = extractor.extract(rgb, timestamp_ms=timestamp_ms)
        timestamp_ms += frame_duration_ms

        if features is not None:
            frames.append(features)
        else:
            # Insert zero frame to maintain timing
            frames.append(np.zeros(FRAME_FEATURES, dtype=np.float32))

    cap.release()
    extractor.close()

    # Split into sequences
    sequences = []
    for start in range(0, len(frames) - seq_len + 1, seq_len):
        seq = np.array(frames[start : start + seq_len], dtype=np.float32)
        sequences.append(seq)

    # Handle trailing frames (pad if > half a sequence)
    remainder = len(frames) % seq_len
    if remainder > seq_len // 2:
        tail = frames[-(remainder):]
        pad = [np.zeros(FRAME_FEATURES, dtype=np.float32)] * (seq_len - remainder)
        sequences.append(np.array(pad + tail, dtype=np.float32))

    return sequences, video_fps


# ---------------------------------------------------------------------------
# Overlay drawing
# ---------------------------------------------------------------------------

def draw_overlay(
    frame: np.ndarray,
    result: ExtractionResult,
    buffer: SequenceBuffer,
    frame_count: int = 0,
    seq_count: int = 0,
) -> None:
    """Draw landmark skeletons and a HUD on a BGR video frame (in-place)."""
    import cv2

    h, w = frame.shape[:2]

    # --- Pose skeleton (green) ---
    if result.pose_detected and result.pose_landmarks is not None:
        pose_lms = result.pose_landmarks
        for i, j in POSE_UPPER_CONNECTIONS:
            pt1 = (int(pose_lms[i].x * w), int(pose_lms[i].y * h))
            pt2 = (int(pose_lms[j].x * w), int(pose_lms[j].y * h))
            cv2.line(frame, pt1, pt2, (0, 200, 0), 2)
        for idx in POSE_DRAW_INDICES:
            pt = (int(pose_lms[idx].x * w), int(pose_lms[idx].y * h))
            cv2.circle(frame, pt, 5, (0, 255, 0), -1)
            cv2.circle(frame, pt, 5, (0, 150, 0), 1)

    # --- Hand skeletons ---
    for hand_idx, hand_lms in enumerate(result.hand_landmarks):
        label = result.handedness[hand_idx][0].category_name
        # Cyan-blue for signer's right (MediaPipe "Left"), orange for left
        if label == "Left":
            color, joint_color = (255, 200, 50), (255, 255, 100)
        else:
            color, joint_color = (50, 180, 255), (100, 220, 255)

        for i, j in HAND_CONNECTIONS:
            pt1 = (int(hand_lms[i].x * w), int(hand_lms[i].y * h))
            pt2 = (int(hand_lms[j].x * w), int(hand_lms[j].y * h))
            cv2.line(frame, pt1, pt2, color, 2)
        for lm in hand_lms:
            pt = (int(lm.x * w), int(lm.y * h))
            cv2.circle(frame, pt, 3, joint_color, -1)

    # --- HUD panel (semi-transparent bar at bottom) ---
    panel_h = 90
    overlay = frame[h - panel_h : h, :].copy()
    cv2.rectangle(frame, (0, h - panel_h), (w, h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.3, frame[h - panel_h : h, :], 0.7, 0,
                    frame[h - panel_h : h, :])

    y_base = h - panel_h + 22

    # Detection status dots
    def _dot(x, y, detected, label_text):
        c = (0, 255, 0) if detected else (0, 0, 180)
        cv2.circle(frame, (x, y - 5), 6, c, -1)
        cv2.putText(frame, label_text, (x + 12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    _dot(15, y_base, result.pose_detected, "Pose")
    _dot(95, y_base, result.left_detected, "L-Hand")
    _dot(195, y_base, result.right_detected, "R-Hand")

    # Counts
    cv2.putText(frame, f"Frames: {frame_count}", (15, y_base + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(frame, f"Sequences: {seq_count}", (140, y_base + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Sequence buffer progress bar
    buf_fill = len(buffer._buffer)
    buf_max = buffer.seq_len
    bar_x, bar_y = 15, y_base + 40
    bar_w, bar_h = w - 30, 16
    fill_w = int(bar_w * buf_fill / buf_max) if buf_max > 0 else 0

    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
    if fill_w > 0:
        bar_color = (0, 255, 180) if buf_fill < buf_max else (0, 255, 255)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h), (150, 150, 150), 1)
    cv2.putText(frame, f"Buffer: {buf_fill}/{buf_max}",
                (bar_x + 5, bar_y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Feature vector norm
    if result.features is not None:
        norm = float(np.linalg.norm(result.features))
        cv2.putText(frame, f"||feat||={norm:.1f}",
                    (bar_x + bar_w - 110, bar_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 255), 1)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import cv2
    import time

    ensure_models()
    print(f"Hand model:  {HAND_MODEL_PATH}")
    print(f"Pose model:  {POSE_MODEL_PATH}")
    print(f"Features per frame: {FRAME_FEATURES}")
    print()

    extractor = FeatureExtractor(mode="image")
    buf = SequenceBuffer(seq_len=30, overlap=15)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No camera available — skipping live test.")
    else:
        print("Camera open. Warming up...")
        for _ in range(30):
            cap.read()
            time.sleep(0.03)

        print("Tracking active. Press Q to quit.")
        frame_count = 0
        seq_count = 0
        fail_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                fail_count += 1
                if fail_count > 30:
                    print("Camera stopped responding.")
                    break
                continue
            fail_count = 0

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = extractor.extract_with_landmarks(rgb)

            if result.features is not None:
                frame_count += 1
                seq = buf.add_frame(result.features)
                if seq is not None:
                    seq_count += 1

            draw_overlay(frame, result, buf, frame_count, seq_count)
            cv2.imshow("ASL Feature Tracker", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        extractor.close()
        print(f"\nProcessed {frame_count} frames, {seq_count} sequences.")
