"""ASL fingerspelling recognizer using MediaPipe hand landmarks.

Extracts 21 hand landmarks per hand, normalizes them relative to the wrist,
and classifies the pose using a trained scikit-learn model.
"""

from __future__ import annotations

import os
import pickle
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

# ASL alphabet labels (A-Z, no J or Z since they involve motion)
ASL_STATIC_LABELS = [
    c for c in "ABCDEFGHIKLMNOPQRSTUVWXY"
]  # 24 static letters

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def extract_landmarks(hand_landmarks) -> np.ndarray:
    """Extract and normalize 21 hand landmarks into a flat feature vector.

    Returns a 1-D array of shape (63,) — 21 landmarks × (x, y, z),
    all relative to the wrist (landmark 0).
    """
    coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32,
    )
    # Normalize: translate so wrist is origin
    wrist = coords[0].copy()
    coords -= wrist
    # Scale so max absolute value is 1
    max_val = np.max(np.abs(coords))
    if max_val > 0:
        coords /= max_val
    return coords.flatten()


class ASLRecognizer:
    """Recognizes ASL fingerspelling from a camera frame.

    Parameters
    ----------
    model_path : str or None
        Path to a pickled scikit-learn classifier.  If None, runs in
        "landmark-only" mode (draws landmarks but doesn't classify).
    confidence : float
        Minimum prediction probability to accept a letter.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence: float = 0.6,
    ) -> None:
        self.confidence = confidence
        self.classifier = None
        self.label_names: list[str] = ASL_STATIC_LABELS

        if model_path and os.path.isfile(model_path):
            with open(model_path, "rb") as f:
                data = pickle.load(f)
            self.classifier = data["model"]
            self.label_names = data.get("labels", ASL_STATIC_LABELS)
            print(f"Loaded ASL model: {model_path}")
        else:
            print(
                "No ASL model loaded — running in landmark-only mode.\n"
                "Run  python asl_trainer.py  to collect data and train a model."
            )

        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

    def process_frame(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, list[tuple[str, float]]]:
        """Process a BGR camera frame.

        Returns
        -------
        annotated : np.ndarray
            Frame with hand landmarks drawn.
        predictions : list of (letter, confidence)
            One entry per detected hand.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        predictions: list[tuple[str, float]] = []

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                # Draw landmarks on frame
                mp_drawing.draw_landmarks(
                    frame,
                    hand_lms,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )
                if self.classifier is not None:
                    features = extract_landmarks(hand_lms).reshape(1, -1)
                    proba = self.classifier.predict_proba(features)[0]
                    idx = int(np.argmax(proba))
                    conf = float(proba[idx])
                    if conf >= self.confidence:
                        letter = self.label_names[idx]
                        predictions.append((letter, conf))

        return frame, predictions

    def close(self) -> None:
        self.hands.close()
