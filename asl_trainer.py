"""Collect hand-landmark data and train an ASL fingerspelling classifier.

Workflow
-------
1. Run this script:  python asl_trainer.py
2. A camera window opens.  Press **a-y** (skipping j) to start recording
   samples for that letter.  Hold the sign steady — it captures one sample
   per frame while the key is held.
3. Press **t** to train the model on everything collected so far.
4. Press **q** to quit.

The trained model is saved to  asl_model.pkl  and loaded automatically by
asl_app.py.
"""

from __future__ import annotations

import os
import pickle
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from asl_recognizer import ASL_STATIC_LABELS, extract_landmarks

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "asl_data")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "asl_model.pkl")

# Map keyboard keys to ASL label indices (skip j and z — motion-based)
_KEY_TO_LABEL: dict[str, int] = {}
_key_iter = ord("a")
for i, letter in enumerate(ASL_STATIC_LABELS):
    _KEY_TO_LABEL[chr(_key_iter)] = i
    _key_iter += 1
    # skip 'j' key (maps to nothing)
    if chr(_key_iter) == "j":
        _key_iter += 1


def save_samples(label: str, features: list[np.ndarray]) -> None:
    """Append feature arrays to a .npy file for the given label."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{label}.npy")
    arr = np.array(features, dtype=np.float32)
    if os.path.exists(path):
        existing = np.load(path)
        arr = np.concatenate([existing, arr], axis=0)
    np.save(path, arr)
    print(f"  Saved {len(features)} samples for '{label}' (total {len(arr)})")


def load_dataset() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load all saved landmark data and return (X, y, label_names).

    Discovers all .npy files in DATA_DIR so it works with both
    fingerspelling letters and full sign labels from the harvester.
    """
    xs, ys = [], []
    label_names: list[str] = []

    if os.path.isdir(DATA_DIR):
        for npy in sorted(Path(DATA_DIR).glob("*.npy")):
            label = npy.stem
            data = np.load(str(npy))
            idx = len(label_names)
            label_names.append(label)
            xs.append(data)
            ys.append(np.full(len(data), idx, dtype=np.int32))

    if not xs:
        return np.empty((0, 63), dtype=np.float32), np.empty(0, dtype=np.int32), []
    return np.concatenate(xs), np.concatenate(ys), label_names


def train_model() -> None:
    """Train a RandomForest on collected data and save to disk."""
    X, y, label_names = load_dataset()
    if len(X) == 0:
        print("No training data found. Record some samples first.")
        return

    unique_labels = np.unique(y)
    print(f"\nTraining on {len(X)} samples across {len(unique_labels)} signs...")
    for i, name in enumerate(label_names):
        count = int(np.sum(y == i))
        print(f"  {name}: {count} samples")

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
    )

    # Quick cross-validation
    if len(unique_labels) >= 2 and len(X) >= 10:
        scores = cross_val_score(clf, X, y, cv=min(5, len(unique_labels)), scoring="accuracy")
        print(f"  Cross-val accuracy: {scores.mean():.1%} (+/- {scores.std():.1%})")

    clf.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": clf, "labels": label_names}, f)
    print(f"  Model saved to {MODEL_PATH}\n")


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        return

    print("=== ASL Training Data Collector ===")
    print("Controls:")
    print("  a-y  : hold key + sign the letter to record samples")
    print("  t    : train model on collected data")
    print("  q    : quit")
    print()

    current_label: str | None = None
    buffer: list[np.ndarray] = []
    sample_count = 0
    last_status = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_lms,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )
                # If recording, capture landmarks
                if current_label is not None:
                    features = extract_landmarks(hand_lms)
                    buffer.append(features)
                    sample_count += 1

        # Status bar
        status = f"Recording: '{ASL_STATIC_LABELS[_KEY_TO_LABEL[current_label]]}' ({sample_count} samples)" if current_label else "Ready — press a letter key + sign"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show sample counts per letter
        X, y, lnames = load_dataset() if os.path.exists(DATA_DIR) else (np.empty((0,)), np.empty((0,)), [])
        info_y = 60
        for i, label in enumerate(lnames):
            count = int(np.sum(y == i)) if len(y) > 0 else 0
            if count > 0:
                cv2.putText(frame, f"{label}:{count}", (10 + (i % 12) * 50, info_y + (i // 12) * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        cv2.imshow("ASL Trainer", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            # Save any remaining buffer
            if current_label and buffer:
                save_samples(ASL_STATIC_LABELS[_KEY_TO_LABEL[current_label]], buffer)
            break
        elif key == ord("t"):
            # Save buffer first
            if current_label and buffer:
                save_samples(ASL_STATIC_LABELS[_KEY_TO_LABEL[current_label]], buffer)
                buffer = []
                current_label = None
                sample_count = 0
            train_model()
        elif chr(key) in _KEY_TO_LABEL:
            new_label = chr(key)
            if new_label != current_label:
                # Save previous buffer
                if current_label and buffer:
                    save_samples(ASL_STATIC_LABELS[_KEY_TO_LABEL[current_label]], buffer)
                buffer = []
                sample_count = 0
                current_label = new_label
                print(f"  Recording for '{ASL_STATIC_LABELS[_KEY_TO_LABEL[new_label]]}'...")
        elif key == 255:
            # No key pressed — keep going
            pass
        else:
            # Any other key stops recording
            if current_label and buffer:
                save_samples(ASL_STATIC_LABELS[_KEY_TO_LABEL[current_label]], buffer)
            buffer = []
            current_label = None
            sample_count = 0

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
