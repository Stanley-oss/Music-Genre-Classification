# preprocessing.py
# Clean GTZAN preprocessing script for MusicFlowNet.
# Run this before training. It creates artist-aware train/val/test splits and audio segments.

from __future__ import annotations

import csv
import math
import random
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
from scipy import ndimage, signal

# =========================
# Fixed paths (Windows)
# =========================
ROOT_DIR = Path("/home/stanley/dev/Music-Genre-Classification")
RAW_DIR = ROOT_DIR / "gtzan_dataset" / "genres_original"
OUT_DIR = ROOT_DIR / "Ablation" / "code" / "preprocessed"
MANIFEST_DIR = OUT_DIR / "manifests"
ARTIST_MAP_CANDIDATES = [
    ROOT_DIR / "gtzan_dataset" / "GTZAN_SONGTITLE_ARTIST.csv",
]

# =========================
# Global settings
# =========================
SAMPLE_RATE = 22050
AUDIO_EXTS = {".wav", ".au", ".mp3", ".ogg", ".flac"}
RANDOM_SEED = 3407
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Only keep these segment scales
SEGMENT_SECONDS = [1.0, 3.0, 10.0, 30.0]
HOP_SECONDS = {
    1.0: 0.5,
    3.0: 1.5,
    10.0: 5.0,
    30.0: 30.0,
}

# Conservative trimming: only front/back silence, and cap trim amount
TRIM_TOP_DB = 30
TRIM_FRAME_LENGTH = 2048
TRIM_HOP_LENGTH = 512
MAX_TRIM_PER_SIDE_SEC = 0.8
MIN_KEEP_DURATION_SEC = 25.0

# Safe cleaning only
HPF_CUTOFF_HZ = 20.0
HPF_ORDER = 2
ENABLE_LIGHT_DENOISE = True
NOISE_LOW_ENERGY_FRAME_RATIO = 0.12
NOISE_GATE_STRENGTH = 1.2
MASK_SMOOTH_FREQ = 5
MASK_SMOOTH_TIME = 7
MIN_MASK = 0.15
N_FFT = 2048
STFT_HOP = 512

# Run switches
BUILD_SPLITS = True
BUILD_FULL_BASE = True
BUILD_FULL_CLEAN = True
BUILD_SEGMENTS = True
OVERWRITE_EXISTING = False


# =========================
# Small helpers
# =========================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_artist_map_path() -> Optional[Path]:
    for p in ARTIST_MAP_CANDIDATES:
        if p.exists():
            return p
    return None


ARTIST_MAP_CSV = resolve_artist_map_path()


def load_artist_map() -> Dict[str, str]:
    """
    Optional artist map.

    Supported file names (checked in order):
    - artist_map.csv
    - GTZAN_SONGTITLE_ARTIST.csv
    - GTZAN_SONGTITLE_ARTIST(1).csv

    Supported key columns:
    - song_id
    - filename
    - stem
    - ref   (GTZAN_SONGTITLE_ARTIST common format)

    Supported artist columns:
    - artist
    - artistName
    """
    csv_path = ARTIST_MAP_CSV
    if csv_path is None:
        return {}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Artist map csv is empty: {csv_path}")

        fieldnames = {name.strip() for name in reader.fieldnames if name is not None}

        artist_col: Optional[str] = None
        for candidate in ["artist", "artistName"]:
            if candidate in fieldnames:
                artist_col = candidate
                break
        if artist_col is None:
            raise ValueError(
                f"{csv_path.name} must contain one of these artist columns: artist, artistName"
            )

        key_col: Optional[str] = None
        for candidate in ["song_id", "filename", "stem", "ref"]:
            if candidate in fieldnames:
                key_col = candidate
                break
        if key_col is None:
            raise ValueError(
                f"{csv_path.name} must contain one of these key columns: song_id, filename, stem, ref"
            )

        mapping: Dict[str, str] = {}
        for row in reader:
            key = (row.get(key_col) or "").strip()
            artist = (row.get(artist_col) or "").strip()
            if key and artist:
                mapping[key] = artist
        print(f"[artist-map] loaded {len(mapping)} rows from: {csv_path}")
        return mapping


def scan_audio_files(artist_map: Dict[str, str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for genre_dir in sorted(RAW_DIR.iterdir()):
        if not genre_dir.is_dir():
            continue
        genre = genre_dir.name
        for audio_path in sorted(genre_dir.iterdir()):
            if audio_path.suffix.lower() not in AUDIO_EXTS:
                continue
            song_id = f"{genre}__{audio_path.stem}"
            artist = (
                artist_map.get(song_id)
                or artist_map.get(audio_path.name)
                or artist_map.get(audio_path.stem)
                or "__unknown_artist__"
            )
            rows.append(
                {
                    "song_id": song_id,
                    "genre": genre,
                    "artist": artist,
                    "orig_path": str(audio_path),
                    "filename": audio_path.name,
                    "stem": audio_path.stem,
                }
            )
    return rows


def compute_split_targets(n: int) -> Dict[str, int]:
    n_train = int(round(n * TRAIN_RATIO))
    n_val = int(round(n * VAL_RATIO))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    n_test = n - n_train - n_val
    if n_test <= 0:
        n_test = 1
        if n_train > n_val:
            n_train -= 1
        else:
            n_val -= 1
    return {"train": n_train, "val": n_val, "test": n_test}


def choose_split_by_remaining(remaining: Dict[str, int], rng: random.Random) -> str:
    available = [k for k, v in remaining.items() if v > 0]
    if not available:
        return "train"
    max_left = max(remaining[k] for k in available)
    candidates = [k for k in available if remaining[k] == max_left]
    return rng.choice(candidates)


def make_artist_aware_stratified_split(rows: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Within each genre:
    - shuffle artists
    - shuffle songs under each artist
    - distribute songs one by one to train/val/test according to remaining quota

    This matches the user's request: do not let one singer's songs all fall into the same split.
    If artist metadata is unavailable, every song is treated as unknown and this degenerates to song-level shuffle.
    """
    rng = random.Random(RANDOM_SEED)
    grouped_by_genre: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped_by_genre[row["genre"]].append(row)

    split_map: Dict[str, str] = {}
    for genre, items in grouped_by_genre.items():
        targets = compute_split_targets(len(items))
        remaining = dict(targets)

        by_artist: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        for row in items:
            by_artist[row["artist"]].append(row)

        artist_groups = list(by_artist.items())
        rng.shuffle(artist_groups)
        for _, artist_items in artist_groups:
            rng.shuffle(artist_items)
            for row in artist_items:
                split = choose_split_by_remaining(remaining, rng)
                split_map[row["song_id"]] = split
                remaining[split] -= 1

    return split_map


# =========================
# Audio preprocessing
# =========================
def load_audio(audio_path: Path) -> np.ndarray:
    """
    Let librosa handle the backend selection.
    If the file is broken / unsupported, raise to caller.
    The caller will record and skip it.
    """
    y, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    return y.astype(np.float32)


def remove_dc(y: np.ndarray) -> np.ndarray:
    return (y - np.mean(y)).astype(np.float32)


def highpass_20hz(y: np.ndarray) -> np.ndarray:
    nyquist = SAMPLE_RATE * 0.5
    cutoff = HPF_CUTOFF_HZ / nyquist
    b, a = signal.butter(HPF_ORDER, cutoff, btype="highpass")
    return signal.filtfilt(b, a, y).astype(np.float32)


def trim_edges_only(y: np.ndarray) -> Tuple[np.ndarray, float, float, float, float]:
    original_duration = len(y) / SAMPLE_RATE
    _, idx = librosa.effects.trim(
        y,
        top_db=TRIM_TOP_DB,
        frame_length=TRIM_FRAME_LENGTH,
        hop_length=TRIM_HOP_LENGTH,
    )
    start_idx, end_idx = int(idx[0]), int(idx[1])
    start_trim = min(start_idx / SAMPLE_RATE, MAX_TRIM_PER_SIDE_SEC)
    end_trim = min((len(y) - end_idx) / SAMPLE_RATE, MAX_TRIM_PER_SIDE_SEC)

    s = int(round(start_trim * SAMPLE_RATE))
    e = len(y) - int(round(end_trim * SAMPLE_RATE))
    if e <= s:
        return y.copy(), original_duration, original_duration, 0.0, 0.0

    y2 = y[s:e]
    if len(y2) / SAMPLE_RATE < MIN_KEEP_DURATION_SEC:
        return y.copy(), original_duration, original_duration, 0.0, 0.0

    return y2.astype(np.float32), original_duration, len(y2) / SAMPLE_RATE, start_trim, end_trim


def light_spectral_gate(y: np.ndarray) -> np.ndarray:
    D = librosa.stft(y, n_fft=N_FFT, hop_length=STFT_HOP)
    mag = np.abs(D)
    phase = np.angle(D)

    frame_energy = np.mean(mag, axis=0)
    n_noise_frames = max(3, int(math.ceil(len(frame_energy) * NOISE_LOW_ENERGY_FRAME_RATIO)))
    low_idx = np.argsort(frame_energy)[:n_noise_frames]
    noise_profile = np.median(mag[:, low_idx], axis=1, keepdims=True)

    threshold = noise_profile * NOISE_GATE_STRENGTH
    soft_mask = np.clip((mag - threshold) / (mag + 1e-8), MIN_MASK, 1.0)
    soft_mask = ndimage.uniform_filter(
        soft_mask,
        size=(MASK_SMOOTH_FREQ, MASK_SMOOTH_TIME),
        mode="nearest",
    )
    cleaned_mag = mag * soft_mask

    D_clean = cleaned_mag * np.exp(1j * phase)
    y_clean = librosa.istft(D_clean, hop_length=STFT_HOP, length=len(y))
    return y_clean.astype(np.float32)


def preprocess_base(audio_path: Path) -> Tuple[np.ndarray, Dict[str, float]]:
    y = load_audio(audio_path)
    y = remove_dc(y)
    y = highpass_20hz(y)
    y, dur_before, dur_after, trim_start, trim_end = trim_edges_only(y)
    meta = {
        "duration_before_trim": round(dur_before, 6),
        "duration_after_trim": round(dur_after, 6),
        "trim_start_sec": round(trim_start, 6),
        "trim_end_sec": round(trim_end, 6),
    }
    return y, meta


# =========================
# Segmentation
# =========================
def cut_segments(y: np.ndarray, seg_sec: float, hop_sec: float) -> List[np.ndarray]:
    seg_len = int(round(seg_sec * SAMPLE_RATE))
    hop_len = int(round(hop_sec * SAMPLE_RATE))

    if seg_len <= 0 or hop_len <= 0:
        return []

    if len(y) < seg_len:
        pad = seg_len - len(y)
        y_pad = np.pad(y, (0, pad), mode="constant")
        return [y_pad.astype(np.float32)]

    if seg_sec == 30.0:
        y_fix = y[:seg_len] if len(y) >= seg_len else np.pad(y, (0, seg_len - len(y)))
        return [y_fix.astype(np.float32)]

    segments: List[np.ndarray] = []
    last_start = len(y) - seg_len
    starts = list(range(0, last_start + 1, hop_len))
    if starts and starts[-1] != last_start:
        starts.append(last_start)

    for start in starts:
        seg = y[start : start + seg_len]
        if len(seg) < seg_len:
            seg = np.pad(seg, (0, seg_len - len(seg)), mode="constant")
        segments.append(seg.astype(np.float32))
    return segments


# =========================
# Main pipeline
# =========================
def save_audio(path: Path, y: np.ndarray) -> None:
    ensure_dir(path.parent)
    if path.exists() and not OVERWRITE_EXISTING:
        return
    sf.write(str(path), y, SAMPLE_RATE)


def build_song_manifests(all_rows: List[Dict[str, str]], split_map: Dict[str, str]) -> None:
    rows = []
    for row in all_rows:
        row2 = dict(row)
        row2["split"] = split_map[row["song_id"]]
        rows.append(row2)

    fieldnames = ["song_id", "genre", "artist", "filename", "stem", "orig_path", "split"]
    write_csv(MANIFEST_DIR / "songs_all.csv", rows, fieldnames)
    for split in ["train", "val", "test"]:
        split_rows = [r for r in rows if r["split"] == split]
        write_csv(MANIFEST_DIR / f"{split}.csv", split_rows, fieldnames)


def build_bad_file_row(row: Dict[str, str], split: str, exc: Exception) -> Dict[str, str]:
    return {
        "song_id": row["song_id"],
        "genre": row["genre"],
        "artist": row["artist"],
        "split": split,
        "filename": row["filename"],
        "orig_path": row["orig_path"],
        "error_type": type(exc).__name__,
        "error_message": str(exc).strip().replace("\n", " "),
    }


def main() -> None:
    ensure_dir(OUT_DIR)
    ensure_dir(MANIFEST_DIR)

    artist_map = load_artist_map()
    all_rows = scan_audio_files(artist_map)
    if not all_rows:
        raise FileNotFoundError(f"No audio files found under: {RAW_DIR}")

    split_map = make_artist_aware_stratified_split(all_rows) if BUILD_SPLITS else {}
    build_song_manifests(all_rows, split_map)

    full_manifest_rows: List[Dict[str, object]] = []
    segment_manifest_rows: List[Dict[str, object]] = []
    bad_file_rows: List[Dict[str, str]] = []
    skipped_count = 0
    processed_count = 0

    for i, row in enumerate(all_rows, start=1):
        song_id = row["song_id"]
        genre = row["genre"]
        artist = row["artist"]
        split = split_map[song_id]
        audio_path = Path(row["orig_path"])
        stem = audio_path.stem

        try:
            # base version
            y_base, meta = preprocess_base(audio_path)
            out_base = OUT_DIR / "full_base" / genre / f"{stem}.wav"
            if BUILD_FULL_BASE:
                save_audio(out_base, y_base)

            full_manifest_rows.append(
                {
                    "song_id": song_id,
                    "genre": genre,
                    "artist": artist,
                    "split": split,
                    "version": "base",
                    "orig_path": str(audio_path),
                    "processed_path": str(out_base),
                    **meta,
                }
            )

            # clean_light version
            y_clean = None
            out_clean = OUT_DIR / "full_clean_light" / genre / f"{stem}.wav"
            if ENABLE_LIGHT_DENOISE and BUILD_FULL_CLEAN:
                y_clean = light_spectral_gate(y_base)
                save_audio(out_clean, y_clean)
                full_manifest_rows.append(
                    {
                        "song_id": song_id,
                        "genre": genre,
                        "artist": artist,
                        "split": split,
                        "version": "clean_light",
                        "orig_path": str(audio_path),
                        "processed_path": str(out_clean),
                        **meta,
                    }
                )

            if BUILD_SEGMENTS:
                versions = [("base", y_base)]
                if ENABLE_LIGHT_DENOISE and y_clean is not None:
                    versions.append(("clean_light", y_clean))

                for version_name, y_version in versions:
                    for seg_sec in SEGMENT_SECONDS:
                        hop_sec = HOP_SECONDS[seg_sec]
                        segments = cut_segments(y_version, seg_sec, hop_sec)
                        folder_version = "base" if version_name == "base" else "clean"
                        seg_dir = OUT_DIR / f"seg_{int(seg_sec)}s_{folder_version}" / split / genre

                        for idx, seg in enumerate(segments):
                            seg_name = f"{stem}__{int(seg_sec * 1000):05d}ms__{idx:04d}.wav"
                            seg_path = seg_dir / seg_name
                            save_audio(seg_path, seg)
                            segment_manifest_rows.append(
                                {
                                    "song_id": song_id,
                                    "genre": genre,
                                    "artist": artist,
                                    "split": split,
                                    "version": version_name,
                                    "segment_len_sec": seg_sec,
                                    "hop_len_sec": hop_sec,
                                    "segment_index": idx,
                                    "segment_path": str(seg_path),
                                }
                            )

            processed_count += 1
            if i % 50 == 0:
                print(f"[progress] processed {i}/{len(all_rows)} songs, skipped {skipped_count}")

        except Exception as exc:
            skipped_count += 1
            bad_file_rows.append(build_bad_file_row(row, split, exc))
            print(f"[skip] {audio_path} -> {type(exc).__name__}: {exc}")
            continue

    full_fieldnames = [
        "song_id",
        "genre",
        "artist",
        "split",
        "version",
        "orig_path",
        "processed_path",
        "duration_before_trim",
        "duration_after_trim",
        "trim_start_sec",
        "trim_end_sec",
    ]
    seg_fieldnames = [
        "song_id",
        "genre",
        "artist",
        "split",
        "version",
        "segment_len_sec",
        "hop_len_sec",
        "segment_index",
        "segment_path",
    ]
    bad_fieldnames = [
        "song_id",
        "genre",
        "artist",
        "split",
        "filename",
        "orig_path",
        "error_type",
        "error_message",
    ]

    write_csv(MANIFEST_DIR / "full_audio_manifest.csv", full_manifest_rows, full_fieldnames)
    write_csv(MANIFEST_DIR / "segment_manifest.csv", segment_manifest_rows, seg_fieldnames)
    write_csv(MANIFEST_DIR / "bad_files.csv", bad_file_rows, bad_fieldnames)

    print("Done.")
    print(f"Raw input     : {RAW_DIR}")
    print(f"Output        : {OUT_DIR}")
    print(
        f"Artist map    : {ARTIST_MAP_CSV if ARTIST_MAP_CSV is not None else 'not found, fallback to random song-level split'}"
    )
    print(f"Total songs   : {len(all_rows)}")
    print(f"Processed     : {processed_count}")
    print(f"Skipped       : {skipped_count}")
    print(f"Segments      : {len(segment_manifest_rows)}")
    print(f"Bad file log  : {MANIFEST_DIR / 'bad_files.csv'}")


if __name__ == "__main__":
    main()
