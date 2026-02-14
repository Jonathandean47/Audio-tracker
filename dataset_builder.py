"""WLASL dataset pipeline: download videos, extract landmarks, save sequences.

Downloads the WLASL v0.3 manifest and source videos, runs MediaPipe
landmark extraction on each clip, and saves padded/truncated sequences
as compressed .npz files split into train/val/test.

Usage
-----
    python dataset_builder.py                    # Full pipeline, WLASL100
    python dataset_builder.py --num-glosses 300  # WLASL300
    python dataset_builder.py --download-only    # Download videos only
    python dataset_builder.py --process-only     # Process already-downloaded videos

Output
------
    wlasl_data/
      WLASL_v0.3.json          # cached manifest
      raw_videos/              # downloaded source videos
      processed/
        train.npz              # sequences (N, seq_len, 345) + labels (N,)
        val.npz
        test.npz
        glosses.json           # label index → gloss name
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

# Suppress TF/MediaPipe noise before importing feature_extractor
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")

from feature_extractor import FRAME_FEATURES, FeatureExtractor

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent / "wlasl_data"
MANIFEST_PATH = DATA_DIR / "WLASL_v0.3.json"
RAW_VIDEOS_DIR = DATA_DIR / "raw_videos"
PROCESSED_DIR = DATA_DIR / "processed"

WLASL_JSON_URL = (
    "https://raw.githubusercontent.com/dxli94/WLASL/master/"
    "start_kit/WLASL_v0.3.json"
)

WLASL_DECODE_FPS = 25       # FPS used when WLASL annotated frame_start/end
DEFAULT_SEQ_LEN = 60        # padded/truncated sequence length
DEFAULT_NUM_GLOSSES = 100   # WLASL100


# ---------------------------------------------------------------------------
# 1. Manifest download & parsing
# ---------------------------------------------------------------------------

def download_manifest() -> Path:
    """Download WLASL_v0.3.json if not already cached."""
    if MANIFEST_PATH.exists():
        print(f"Manifest cached: {MANIFEST_PATH}")
        return MANIFEST_PATH

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading WLASL_v0.3.json ...")
    urllib.request.urlretrieve(WLASL_JSON_URL, str(MANIFEST_PATH))
    print(f"  Saved to {MANIFEST_PATH}")
    return MANIFEST_PATH


def parse_manifest(
    num_glosses: int = DEFAULT_NUM_GLOSSES,
) -> tuple[list[dict], list[str]]:
    """Parse the WLASL manifest and return the top-N glosses.

    The JSON is a list sorted by number of instances (most first),
    so slicing [:num_glosses] gives the WLASL-N subset.

    Returns (entries, glosses) where entries is the raw JSON list
    and glosses is the ordered list of gloss label strings.
    """
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        data = json.load(f)

    entries = data[:num_glosses]
    glosses = [e["gloss"] for e in entries]

    total = sum(len(e["instances"]) for e in entries)
    print(f"WLASL{num_glosses}: {len(glosses)} glosses, {total} instances")

    return entries, glosses


# ---------------------------------------------------------------------------
# 2. Video downloading
# ---------------------------------------------------------------------------

def _raw_video_path(instance: dict) -> Path:
    """Determine the expected raw-video path for an instance.

    YouTube videos are stored by their 11-char YouTube ID;
    other videos are stored by the WLASL video_id.
    """
    url = instance["url"]
    vid = instance["video_id"]

    if "youtube" in url or "youtu.be" in url:
        yt_id = url[-11:]
        return RAW_VIDEOS_DIR / f"{yt_id}.mp4"

    return RAW_VIDEOS_DIR / f"{vid}.mp4"


def _download_youtube(url: str, dest: Path) -> bool:
    """Download a YouTube video using yt-dlp."""
    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "--no-warnings",
                "--quiet",
                "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                "--merge-output-format", "mp4",
                "-o", str(dest),
                "--no-playlist",
                url,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return result.returncode == 0 and dest.exists()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _download_direct(url: str, dest: Path) -> bool:
    """Download a video via direct HTTP request."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
            ),
        }
        # aslpro requires a referer
        if "aslpro" in url:
            headers["Referer"] = "http://www.aslpro.com/cgi-bin/aslpro/aslpro.cgi"

        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as resp:
            dest.write_bytes(resp.read())
        return dest.exists() and dest.stat().st_size > 1024
    except Exception:
        return False


def download_videos(entries: list[dict]) -> dict[str, Path]:
    """Download all unique source videos for the given entries.

    Returns a mapping of WLASL video_id → raw video path for
    every instance whose video is available locally.
    """
    RAW_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    # Deduplicate: many instances share the same YouTube source video
    downloads: dict[str, tuple[str, Path]] = {}   # stem → (url, path)
    vid_to_stem: dict[str, str] = {}               # video_id → stem

    for entry in entries:
        for inst in entry["instances"]:
            raw = _raw_video_path(inst)
            vid_to_stem[inst["video_id"]] = raw.stem
            if raw.stem not in downloads:
                downloads[raw.stem] = (inst["url"], raw)

    print(f"\n--- Video download: {len(downloads)} unique sources ---")

    new, cached, failed = 0, 0, 0

    for i, (stem, (url, dest)) in enumerate(downloads.items(), 1):
        # Already on disk?
        if dest.exists() and dest.stat().st_size > 1024:
            cached += 1
            continue

        is_yt = "youtube" in url or "youtu.be" in url

        # Skip Flash (.swf) — can't decode
        if "aslpro" in url and ".swf" in url.lower():
            failed += 1
            continue

        tag = "YT" if is_yt else "DL"
        print(f"  [{i}/{len(downloads)}] {tag} {stem} ", end="", flush=True)

        ok = _download_youtube(url, dest) if is_yt else _download_direct(url, dest)

        if ok:
            print("ok")
            new += 1
        else:
            print("fail")
            failed += 1

        time.sleep(random.uniform(0.3, 1.0))

    # Build availability map: video_id → raw path
    available: dict[str, Path] = {}
    for vid_id, stem in vid_to_stem.items():
        p = RAW_VIDEOS_DIR / f"{stem}.mp4"
        if p.exists() and p.stat().st_size > 1024:
            available[vid_id] = p

    print(f"\nDownloads: {new} new, {cached} cached, {failed} failed")
    print(f"Clips with video available: {len(available)}")
    return available


# ---------------------------------------------------------------------------
# 3. Landmark extraction
# ---------------------------------------------------------------------------

def _resample_sequence(
    frames: list[np.ndarray], seq_len: int
) -> np.ndarray:
    """Pad (zero-prefix) or uniformly sub-sample to fixed length.

    Short clips are zero-padded at the start so that the sign action
    is right-aligned (natural for an LSTM to read left-to-right).
    Long clips are uniformly sampled to preserve temporal coverage.
    """
    n = len(frames)

    if n == seq_len:
        return np.array(frames, dtype=np.float32)

    if n < seq_len:
        pad = [np.zeros(FRAME_FEATURES, dtype=np.float32)] * (seq_len - n)
        return np.array(pad + frames, dtype=np.float32)

    # Uniformly pick seq_len indices from n frames
    indices = np.linspace(0, n - 1, seq_len, dtype=int)
    return np.array([frames[i] for i in indices], dtype=np.float32)


def extract_clip_landmarks(
    video_path: Path,
    frame_start: int,
    frame_end: int,
    seq_len: int,
    extractor: FeatureExtractor,
) -> np.ndarray | None:
    """Extract landmarks from a video clip, return a fixed-length sequence.

    Parameters
    ----------
    video_path : Path
        Path to the raw source video.
    frame_start : int
        1-indexed start frame (WLASL convention, at 25 fps).
    frame_end : int
        1-indexed end frame, or -1 for last frame.
    seq_len : int
        Target output length (frames).
    extractor : FeatureExtractor
        Reusable extractor instance (IMAGE mode).

    Returns
    -------
    np.ndarray of shape (seq_len, FRAME_FEATURES) or None on failure.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    if total_frames < 3:
        cap.release()
        return None

    # Convert from WLASL 25-fps annotation to actual video frame indices
    fps_ratio = video_fps / WLASL_DECODE_FPS
    actual_start = max(0, int((frame_start - 1) * fps_ratio))
    actual_end = (
        total_frames if frame_end == -1
        else min(int(frame_end * fps_ratio), total_frames)
    )

    if actual_start >= actual_end:
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, actual_start)

    features: list[np.ndarray] = []

    for _ in range(actual_end - actual_start):
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        feat = extractor.extract(rgb)

        if feat is not None:
            features.append(feat)
        else:
            features.append(np.zeros(FRAME_FEATURES, dtype=np.float32))

    cap.release()

    if len(features) < 3:
        return None

    return _resample_sequence(features, seq_len)


# ---------------------------------------------------------------------------
# 4. Batch processing
# ---------------------------------------------------------------------------

def process_entries(
    entries: list[dict],
    glosses: list[str],
    available: dict[str, Path],
    seq_len: int = DEFAULT_SEQ_LEN,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Process all available clips into landmark sequences.

    Returns a dict mapping split name → (sequences, labels).
    """
    gloss_to_idx = {g: i for i, g in enumerate(glosses)}

    split_seqs: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    split_labs: dict[str, list[int]] = {"train": [], "val": [], "test": []}

    total = sum(
        1 for e in entries
        for inst in e["instances"]
        if inst["video_id"] in available
    )
    print(f"\n--- Processing {total} clips ---")

    # IMAGE mode avoids the strict timestamp-ordering requirement of VIDEO
    # mode, which would need a fresh extractor per clip.  Quality difference
    # is negligible for training-data extraction.
    extractor = FeatureExtractor(mode="image")
    done, ok = 0, 0

    try:
        for entry in entries:
            gloss = entry["gloss"]
            label = gloss_to_idx[gloss]

            for inst in entry["instances"]:
                vid_id = inst["video_id"]
                if vid_id not in available:
                    continue

                done += 1
                split = inst.get("split", "train")
                if split not in split_seqs:
                    split = "train"

                seq = extract_clip_landmarks(
                    available[vid_id],
                    inst["frame_start"],
                    inst["frame_end"],
                    seq_len,
                    extractor,
                )

                if seq is not None:
                    split_seqs[split].append(seq)
                    split_labs[split].append(label)
                    ok += 1

                if done % 25 == 0 or done == total:
                    print(
                        f"  [{done}/{total}] {ok} ok | "
                        f"{gloss} / {vid_id}"
                    )
    finally:
        extractor.close()

    print(f"\nExtraction: {ok}/{done} clips succeeded")

    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for s in ("train", "val", "test"):
        if split_seqs[s]:
            result[s] = (
                np.array(split_seqs[s], dtype=np.float32),
                np.array(split_labs[s], dtype=np.int64),
            )
            n = len(split_seqs[s])
            print(f"  {s}: {n} sequences  shape {result[s][0].shape}")
    return result


# ---------------------------------------------------------------------------
# 5. Save / load helpers
# ---------------------------------------------------------------------------

def save_dataset(
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    glosses: list[str],
) -> None:
    """Save processed splits as compressed .npz + glosses.json."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for name, (sequences, labels) in splits.items():
        path = PROCESSED_DIR / f"{name}.npz"
        np.savez_compressed(str(path), sequences=sequences, labels=labels)
        size_mb = path.stat().st_size / 1024 / 1024
        print(f"  Saved {path.name}: {sequences.shape}  ({size_mb:.1f} MB)")

    gloss_path = PROCESSED_DIR / "glosses.json"
    with open(gloss_path, "w", encoding="utf-8") as f:
        json.dump({str(i): g for i, g in enumerate(glosses)}, f, indent=2)
    print(f"  Saved {gloss_path.name}")


def load_dataset(
    split: str = "train",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load a processed split.

    Returns (sequences, labels, glosses).
    """
    data = np.load(str(PROCESSED_DIR / f"{split}.npz"))
    with open(PROCESSED_DIR / "glosses.json", encoding="utf-8") as f:
        gmap = json.load(f)
    glosses = [gmap[str(i)] for i in range(len(gmap))]
    return data["sequences"], data["labels"], glosses


# ---------------------------------------------------------------------------
# 6. Main entry point
# ---------------------------------------------------------------------------

def build(
    num_glosses: int = DEFAULT_NUM_GLOSSES,
    seq_len: int = DEFAULT_SEQ_LEN,
    download_only: bool = False,
    process_only: bool = False,
) -> None:
    """Full pipeline: manifest → download → extract → save."""
    print(f"=== WLASL{num_glosses} Dataset Builder ===")
    print(f"  Sequence length : {seq_len} frames")
    print(f"  Features / frame: {FRAME_FEATURES}")
    print(f"  Data directory  : {DATA_DIR}")
    print()

    # 1 — manifest
    download_manifest()
    entries, glosses = parse_manifest(num_glosses)

    # 2 — download
    if not process_only:
        available = download_videos(entries)
    else:
        available: dict[str, Path] = {}
        for entry in entries:
            for inst in entry["instances"]:
                p = _raw_video_path(inst)
                if p.exists() and p.stat().st_size > 1024:
                    available[inst["video_id"]] = p
        print(f"Found {len(available)} already-downloaded videos")

    if download_only:
        print("\n--download-only: stopping after download phase.")
        return

    # 3 — process
    splits = process_entries(entries, glosses, available, seq_len)

    # 4 — save
    if splits:
        print()
        save_dataset(splits, glosses)
    else:
        print("\nNo sequences produced — nothing to save.")

    print("\nDone!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WLASL dataset pipeline: download → extract → save"
    )
    parser.add_argument(
        "--num-glosses", type=int, default=DEFAULT_NUM_GLOSSES,
        help=f"Number of glosses to include (default {DEFAULT_NUM_GLOSSES})",
    )
    parser.add_argument(
        "--seq-len", type=int, default=DEFAULT_SEQ_LEN,
        help=f"Fixed sequence length in frames (default {DEFAULT_SEQ_LEN})",
    )
    parser.add_argument(
        "--download-only", action="store_true",
        help="Download videos only — skip landmark processing",
    )
    parser.add_argument(
        "--process-only", action="store_true",
        help="Process already-downloaded videos only — skip downloading",
    )

    args = parser.parse_args()
    build(
        num_glosses=args.num_glosses,
        seq_len=args.seq_len,
        download_only=args.download_only,
        process_only=args.process_only,
    )
