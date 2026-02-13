"""Harvest ASL training data from YouTube videos.

Downloads videos from ASL-related YouTube channels (e.g. @aslu / Dr. Bill
Vicars), extracts frames where hands are detected, computes MediaPipe
landmarks, and saves them as labeled training data.

Usage
-----
# Harvest from a single video (label taken from title):
python asl_harvester.py --url "https://www.youtube.com/watch?v=VIDEO_ID"

# Harvest from a channel or playlist:
python asl_harvester.py --url "https://www.youtube.com/@aslu/videos" --max-videos 50

# Harvest with an explicit label override:
python asl_harvester.py --url "https://www.youtube.com/watch?v=VIDEO_ID" --label HELLO

# List videos that would be downloaded (dry run):
python asl_harvester.py --url "https://www.youtube.com/@aslu/videos" --dry-run

# After harvesting, train the model:
python asl_harvester.py --train

Requirements
------------
pip install yt-dlp opencv-python mediapipe numpy scikit-learn
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# Suppress noisy MediaPipe / TensorFlow Lite warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")

import cv2
import mediapipe as mp
import numpy as np

from asl_recognizer import extract_landmarks

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "asl_data"
MODEL_PATH = APP_DIR / "asl_model.pkl"
VIDEO_CACHE = APP_DIR / "video_cache"

mp_hands = mp.solutions.hands


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_label(title: str) -> str:
    """Extract a usable sign label from a video title.

    ASLU titles typically look like:
      "HELLO"
      "HELLO (side view)"
      "WISE, wisdom"
      "CALCULATOR, calculate"
      "EXTENSION-CORD"
      "REPENT (initialized-SORRY-version)"
      "code of ethics palm version"         ← skip (multi-word description)
      "CLIPBOARD (version 3 clip onto ...)" ← skip (version variant)
      "quiz: what sign is this?"            ← skip
    """
    title = title.strip()

    # ---- Skip rules ----
    # Quiz / question videos
    if re.search(r"\bquiz\b", title, re.IGNORECASE):
        return ""
    if title.rstrip().endswith("?"):
        return ""
    # Screensaver / loop / resolution tip videos
    if re.search(r"\bscreensaver\b|\bloop\b|\bresolution\b", title, re.IGNORECASE):
        return ""
    # Videos about decades / years notation
    if re.search(r"'?\d0'?s", title):
        return ""
    # Videos that are long descriptive phrases (no clear single sign)
    if re.search(r"\bmulling\b|\bprocessing\b|\bthinking\b|\bcogitat", title, re.IGNORECASE):
        return ""
    # "Three wise men brought ..." style sentence videos
    if len(title) > 60 and title[0].isupper() and title[1:2].islower():
        return ""
    # Skip "... requires context" or neologism commentary
    if re.search(r"requires[- ]context|neologism|compare with", title, re.IGNORECASE):
        return ""

    # ---- Strip noise ----
    # Remove parenthetical content: (side view), (palm version), (ASL), etc.
    cleaned = re.sub(r"\s*\([^)]*\)", "", title).strip()

    # Remove common suffixes/noise phrases (case-insensitive)
    noise_phrases = [
        r"\bside view\b", r"\bpalm version\b", r"\bvertical version\b",
        r"\bversion\s*\d*\b", r"\binitialized[\w-]*version\b",
        r"\binitialized\b", r"\buninflected\b", r"\binflected\b",
        r"\bnon-dominant\b", r"\bdepiction\b", r"\bclassifier\b",
        r"\bASL\b", r"\basl\b", r"\bsign\s+for\b", r"\bhow\s+to\s+sign\b",
        r"\bin\s+asl\b",
    ]
    for pat in noise_phrases:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()

    # Strip leading/trailing punctuation and whitespace
    cleaned = re.sub(r"^[\s,\-–—:]+|[\s,\-–—:]+$", "", cleaned)

    # If there are commas, take the first item only (synonyms follow)
    if "," in cleaned:
        cleaned = cleaned.split(",")[0].strip()

    # If there are " / " separators, take the first item
    if " / " in cleaned:
        cleaned = cleaned.split(" / ")[0].strip()

    # Remove any remaining special chars except hyphens and spaces
    cleaned = re.sub(r"[^A-Za-z0-9\- ]", "", cleaned).strip()

    # Uppercase
    label = cleaned.upper()

    # Replace spaces with hyphens for multi-word sign names
    label = re.sub(r"\s+", "-", label)

    # Remove double/trailing hyphens
    label = re.sub(r"-{2,}", "-", label).strip("-")

    # ---- Validation ----
    # Skip if empty, too short, or too long
    if not label or len(label) < 2 or len(label) > 20:
        return ""
    # Skip if it's more than 2 words (likely a description, not a sign name)
    if label.count("-") > 1:
        return ""

    return label


def _run_ytdlp(args: list[str]) -> subprocess.CompletedProcess:
    """Run yt-dlp and return the result."""
    cmd = [sys.executable, "-m", "yt_dlp", *args]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300)


def list_videos(url: str, max_videos: int = 50) -> list[dict]:
    """Return a list of {id, title, url} dicts from a channel/playlist URL."""
    result = _run_ytdlp([
        "--flat-playlist",
        "--print", "%(id)s\t%(title)s",
        "--playlist-end", str(max_videos),
        url,
    ])
    if result.returncode != 0:
        print(f"yt-dlp error:\n{result.stderr}", file=sys.stderr)
        return []

    videos = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("\t", 1)
        if len(parts) == 2:
            vid_id, title = parts
            videos.append({
                "id": vid_id,
                "title": title,
                "url": f"https://www.youtube.com/watch?v={vid_id}",
            })
    return videos


def download_video(url: str, output_dir: Path) -> Path | None:
    """Download a video to output_dir.  Returns path to the file or None."""
    output_dir.mkdir(parents=True, exist_ok=True)
    result = _run_ytdlp([
        "-f", "best[height<=480]",  # Keep small — we only need landmarks
        "--no-playlist",
        "-o", str(output_dir / "%(id)s.%(ext)s"),
        "--print", "after_move:filepath",
        url,
    ])
    if result.returncode != 0:
        print(f"  Download failed: {result.stderr.strip()}", file=sys.stderr)
        return None

    filepath = result.stdout.strip().splitlines()[-1]
    return Path(filepath) if filepath and Path(filepath).exists() else None


def extract_hand_data(
    video_path: Path,
    label: str,
    sample_interval: int = 3,
    max_samples: int = 500,
) -> list[np.ndarray]:
    """Open a video, extract hand landmarks every `sample_interval` frames.

    Parameters
    ----------
    video_path : Path
        Path to the video file.
    label : str
        Sign label (for progress messages only).
    sample_interval : int
        Process every Nth frame (reduces redundancy / speeds up).
    max_samples : int
        Stop after this many successful landmark extractions.

    Returns
    -------
    list of np.ndarray
        Each array is shape (63,) — flattened normalized landmarks.
    """
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Could not open {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    samples: list[np.ndarray] = []
    frame_idx = 0

    while len(samples) < max_samples:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % sample_interval != 0:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                feat = extract_landmarks(hand_lms)
                samples.append(feat)

    cap.release()
    hands.close()
    return samples


def save_samples(label: str, features: list[np.ndarray]) -> None:
    """Append features to the label's .npy file in DATA_DIR."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{label}.npy"
    arr = np.array(features, dtype=np.float32)
    if path.exists():
        existing = np.load(str(path))
        arr = np.concatenate([existing, arr], axis=0)
    np.save(str(path), arr)
    print(f"  Saved {len(features)} samples for '{label}' (total {len(arr)})")


def harvest_single(
    url: str,
    label_override: str | None = None,
    keep_video: bool = False,
) -> None:
    """Download one video and extract training data."""
    # Get title for labeling
    result = _run_ytdlp([
        "--no-playlist",
        "--print", "%(title)s",
        url,
    ])
    title = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    label = label_override or _clean_label(title)

    if not label:
        print(f"  Skipping (no usable label): {title}")
        return

    print(f"  Title : {title}")
    print(f"  Label : {label}")

    # Download
    video_path = download_video(url, VIDEO_CACHE)
    if not video_path:
        return

    # Extract landmarks
    samples = extract_hand_data(video_path, label)
    if samples:
        save_samples(label, samples)
    else:
        print(f"  No hand landmarks detected in video.")

    # Cleanup
    if not keep_video and video_path.exists():
        video_path.unlink()


def harvest_channel(
    url: str,
    max_videos: int = 50,
    label_override: str | None = None,
    keep_videos: bool = False,
) -> None:
    """Download multiple videos from a channel/playlist and extract data."""
    print(f"Fetching video list from {url} ...")
    videos = list_videos(url, max_videos)
    if not videos:
        print("No videos found.")
        return

    print(f"Found {len(videos)} videos.\n")

    for i, vid in enumerate(videos, 1):
        label = label_override or _clean_label(vid["title"])
        if not label:
            print(f"[{i}/{len(videos)}] Skipping (no label): {vid['title']}")
            continue
        print(f"[{i}/{len(videos)}] {vid['title']}  →  {label}")
        harvest_single(vid["url"], label_override=label, keep_video=keep_videos)
        print()


def train_model() -> None:
    """Train classifier on all harvested data (delegates to asl_trainer)."""
    # Import locally to avoid circular issues
    from asl_trainer import train_model as _train
    _train()


def show_dataset_stats() -> None:
    """Print summary of collected data."""
    if not DATA_DIR.exists():
        print("No data collected yet.")
        return

    print(f"\n{'Label':<20} {'Samples':>8}")
    print("-" * 30)
    total = 0
    for npy in sorted(DATA_DIR.glob("*.npy")):
        arr = np.load(str(npy))
        label = npy.stem
        print(f"{label:<20} {len(arr):>8}")
        total += len(arr)
    print("-" * 30)
    print(f"{'TOTAL':<20} {total:>8}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Harvest ASL training data from YouTube videos."
    )
    parser.add_argument("--url", help="YouTube video, channel, or playlist URL")
    parser.add_argument("--label", help="Override label for all downloaded videos")
    parser.add_argument("--max-videos", type=int, default=50,
                        help="Max videos to process from a channel/playlist (default 50)")
    parser.add_argument("--keep-videos", action="store_true",
                        help="Keep downloaded video files (default: delete after processing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List videos without downloading")
    parser.add_argument("--train", action="store_true",
                        help="Train model on collected data")
    parser.add_argument("--stats", action="store_true",
                        help="Show dataset statistics")

    args = parser.parse_args()

    if args.stats:
        show_dataset_stats()
        return

    if args.train:
        train_model()
        return

    if not args.url:
        parser.print_help()
        return

    if args.dry_run:
        videos = list_videos(args.url, args.max_videos)
        print(f"\n{'#':<4} {'Label':<20} {'Title'}")
        print("-" * 70)
        for i, vid in enumerate(videos, 1):
            label = args.label or _clean_label(vid["title"])
            status = label if label else "(skip)"
            print(f"{i:<4} {status:<20} {vid['title']}")
        print(f"\n{len(videos)} videos found.")
        return

    # Detect if URL is a single video or channel/playlist
    if "watch?v=" in args.url and "/playlist" not in args.url:
        harvest_single(args.url, label_override=args.label, keep_video=args.keep_videos)
    else:
        harvest_channel(args.url, max_videos=args.max_videos,
                        label_override=args.label, keep_videos=args.keep_videos)


if __name__ == "__main__":
    main()
