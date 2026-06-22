from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ============================================================
# Project paths
# ============================================================
ROOT_DIR = Path(__file__).resolve().parent
SEG_DIR = ROOT_DIR / "preprocessed" / "seg_3s_base"
MEL_DIR = ROOT_DIR / "mel_cache" / "lm3_base_v1"
OUT_DIR = ROOT_DIR / "emf_v1_out"
TRAIN_ROOT_DIR = ROOT_DIR / "emf_train_runs"
ABLA_ROOT_DIR = ROOT_DIR / "emf_ablation_runs"
DEFAULT_CKPT_DIR = ROOT_DIR / "emf_v1_out"

# Minimal sweep knobs. Default uses the CNN temporal backbone requested for this replacement run.
TEMPORAL_KIND = "cnn"  # choices: conformer, mlp, lstm, rnn, cnn, resnet, transformer
USE_DENOISING = True
CNN_WIDTH_MULT = 0.75
CNN_DEPTH = 1
RESNET_WIDTH_MULT = 1.25
RESNET_DEPTH = 1
MLP_HIDDEN_DIMS = [192, 192]
MLP_OUTPUT_DIM = 160
MLP_DROPOUT = 0.15
RNN_HIDDEN_SIZE = 96
RNN_NUM_LAYERS = 2
RNN_BIDIRECTIONAL = True
RNN_DROPOUT = 0.10
TRANSFORMER_NHEAD = 8
TRANSFORMER_NUM_LAYERS = 1
TRANSFORMER_DIM_FEEDFORWARD = 640
TRANSFORMER_DROPOUT = 0.15

# ============================================================
# Audio / log-mel settings
# ============================================================
SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
FMIN = 20.0
FMAX = 11025.0
TARGET_FRAMES = 130
TOP_DB_FLOOR = -80.0
TOP_DB_CEIL = 20.0

# ============================================================
# Training settings
# ============================================================
SEED = 3407
NUM_CLASSES = 10
BATCH_SIZE = 64
EPOCHS = 60
LEARNING_RATE = 8e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0  # Windows-safe default
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FORCE_CUDA = True  # GPU-only guard: fail fast if this environment cannot see CUDA.

# Loss weights. This is not module stacking; denoising is a representation regularizer.
LABEL_SMOOTHING = 0.08
DENOISE_WEIGHT = 0.12
DENOISED_CE_WEIGHT = 0.20

# Flow-style denoising settings
T_MIN = 0.10
T_MAX = 0.95
TIME_EMB_DIM = 64
EMB_DIM = 128

# SpecAugment: simple, interpretable regularization on time-frequency input.
SPEC_AUG_PROB = 0.75
FREQ_MASKS = 2
TIME_MASKS = 2
MAX_FREQ_MASK = 14
MAX_TIME_MASK = 14

# Optimization utilities
GRAD_CLIP_NORM = 3.0
EARLY_STOP_PATIENCE = 15
SAVE_EXPLANATION_EXAMPLES = 12

# Switches
BUILD_MEL_CACHE = True
OVERWRITE_MEL = False
TRAIN_MODEL = True
RUN_TEST = True
EXPORT_EXPLANATIONS = True


# ============================================================
# General utilities
# ============================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def validate_cuda_device() -> None:
    """
    Keep the original training protocol, but do not silently fall back to CPU.
    If CUDA is not visible, stop immediately so the run cannot accidentally
    become a slow CPU experiment.
    """
    if FORCE_CUDA and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this Python environment. "
            "Current torch build: "
            f"{torch.__version__}, torch.version.cuda={torch.version.cuda}. "
            "Please install a CUDA-enabled PyTorch wheel before running this script."
        )


def print_device_info(device: torch.device) -> None:
    print(f"[device] {device}")
    print(f"[torch] version={torch.__version__} cuda_build={torch.version.cuda}")
    if device.type == "cuda":
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        total_gb = props.total_memory / (1024 ** 3)
        print(f"[gpu] index={idx} name={props.name} memory={total_gb:.2f}GB")
    else:
        print("[gpu] CUDA not available; running on CPU")


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, obj: Dict[str, object]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_song_id(genre: str, wav_name: str) -> Tuple[str, int]:
    """
    v4 segment names look like:
    blues.00000__03000ms__0000.wav
    Returns:
    song_id = blues__blues.00000
    segment_index = 0
    """
    stem = Path(wav_name).stem
    parts = stem.split("__")
    base = parts[0]
    seg_idx = -1
    if len(parts) >= 3:
        try:
            seg_idx = int(parts[-1])
        except ValueError:
            seg_idx = -1
    return f"{genre}__{base}", seg_idx


# ============================================================
# Mel cache generation
# ============================================================
def list_wav_rows() -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    if not SEG_DIR.exists():
        raise FileNotFoundError(
            f"Segment directory not found: {SEG_DIR}\n"
            "Please run dl_audio_preprocess_v4.py first."
        )

    class_names = sorted([p.name for p in (SEG_DIR / "train").iterdir() if p.is_dir()])
    if not class_names:
        raise FileNotFoundError(f"No class folders found under: {SEG_DIR / 'train'}")
    label_to_id = {name: idx for idx, name in enumerate(class_names)}

    rows: List[Dict[str, object]] = []
    for split in ["train", "val", "test"]:
        split_dir = SEG_DIR / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split folder: {split_dir}")
        for genre in class_names:
            genre_dir = split_dir / genre
            if not genre_dir.exists():
                continue
            for wav_path in sorted(genre_dir.glob("*.wav")):
                song_id, seg_idx = parse_song_id(genre, wav_path.name)
                rel = wav_path.relative_to(SEG_DIR)
                mel_path = MEL_DIR / rel.with_suffix(".npy")
                rows.append(
                    {
                        "split": split,
                        "genre": genre,
                        "label": label_to_id[genre],
                        "song_id": song_id,
                        "segment_index": seg_idx,
                        "wav_path": str(wav_path),
                        "mel_path": str(mel_path),
                    }
                )
    return rows, label_to_id


def wav_to_logmel(wav_path: Path) -> np.ndarray:
    y, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
    if y.size == 0:
        raise ValueError(f"Empty audio: {wav_path}")

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )
    # Keep absolute dB reference instead of per-sample peak normalization.
    logmel = librosa.power_to_db(mel, ref=1.0)
    logmel = np.clip(logmel, TOP_DB_FLOOR, TOP_DB_CEIL).astype(np.float32)

    # Fixed time dimension for stable batching.
    if logmel.shape[1] < TARGET_FRAMES:
        pad = TARGET_FRAMES - logmel.shape[1]
        logmel = np.pad(logmel, ((0, 0), (0, pad)), mode="constant", constant_values=TOP_DB_FLOOR)
    elif logmel.shape[1] > TARGET_FRAMES:
        logmel = logmel[:, :TARGET_FRAMES]
    return logmel.astype(np.float32)


def build_mel_cache(rows: List[Dict[str, object]]) -> None:
    print(f"[mel] cache root: {MEL_DIR}")
    ensure_dir(MEL_DIR)
    bad_rows: List[Dict[str, object]] = []
    done = 0
    for i, row in enumerate(rows, start=1):
        wav_path = Path(str(row["wav_path"]))
        mel_path = Path(str(row["mel_path"]))
        if mel_path.exists() and not OVERWRITE_MEL:
            done += 1
            continue
        try:
            logmel = wav_to_logmel(wav_path)
            ensure_dir(mel_path.parent)
            np.save(str(mel_path), logmel)
            done += 1
        except Exception as exc:
            bad_rows.append(
                {
                    "wav_path": str(wav_path),
                    "mel_path": str(mel_path),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc).replace("\n", " "),
                }
            )
            print(f"[mel-skip] {wav_path} -> {type(exc).__name__}: {exc}")
        if i % 1000 == 0:
            print(f"[mel] {i}/{len(rows)} scanned, usable={done}, bad={len(bad_rows)}")

    if bad_rows:
        write_csv(
            OUT_DIR / "bad_mel_files.csv",
            bad_rows,
            ["wav_path", "mel_path", "error_type", "error_message"],
        )
    print(f"[mel] usable={done}, bad={len(bad_rows)}")


def compute_train_stats(rows: List[Dict[str, object]]) -> Tuple[float, float]:
    stats_path = MEL_DIR / "stats_train.json"
    if stats_path.exists() and not OVERWRITE_MEL:
        obj = load_json(stats_path)
        return float(obj["mean"]), float(obj["std"])

    total_sum = 0.0
    total_sumsq = 0.0
    total_count = 0
    for row in rows:
        if row["split"] != "train":
            continue
        mel_path = Path(str(row["mel_path"]))
        if not mel_path.exists():
            continue
        arr = np.load(str(mel_path)).astype(np.float64)
        total_sum += float(arr.sum())
        total_sumsq += float((arr * arr).sum())
        total_count += int(arr.size)

    if total_count == 0:
        raise RuntimeError("No train mel files found for stats computation.")

    mean = total_sum / total_count
    var = max(total_sumsq / total_count - mean * mean, 1e-8)
    std = math.sqrt(var)
    save_json(stats_path, {"mean": mean, "std": std})
    print(f"[stats] train mean={mean:.6f}, std={std:.6f}")
    return float(mean), float(std)


# ============================================================
# Dataset
# ============================================================
class MelGenreDataset(Dataset):
    def __init__(self, rows: List[Dict[str, object]], split: str, mean: float, std: float) -> None:
        self.rows = [r for r in rows if r["split"] == split and Path(str(r["mel_path"])).exists()]
        self.split = split
        self.mean = mean
        self.std = max(std, 1e-6)
        if not self.rows:
            raise RuntimeError(f"No samples for split={split}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        arr = np.load(str(row["mel_path"])).astype(np.float32)
        arr = (arr - self.mean) / self.std
        arr = np.clip(arr, -6.0, 6.0)
        x = torch.from_numpy(arr).unsqueeze(0)  # [1, F, T]
        y = int(row["label"])
        return {
            "x": x,
            "y": torch.tensor(y, dtype=torch.long),
            "song_id": str(row["song_id"]),
            "genre": str(row["genre"]),
            "mel_path": str(row["mel_path"]),
        }


# ============================================================
# Model blocks
# ============================================================
class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size, padding) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(channels, channels, kernel_size=3, padding=1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class TemporalConformerBlock(nn.Module):
    """
    A compact temporal refinement block for music.

    The CNN encoder first extracts local time-frequency patterns. After that,
    this block treats each time frame as a token and models longer rhythm /
    phrase context with self-attention plus a lightweight temporal convolution.
    This is not module stacking for its own sake: it explicitly separates
    local timbre extraction from temporal context modeling.
    """

    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.15) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop_attn = nn.Dropout(dropout)

        self.norm_conv = nn.LayerNorm(channels)
        self.pw1 = nn.Conv1d(channels, channels * 2, kernel_size=1)
        self.dw = nn.Conv1d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
        )
        self.bn = nn.BatchNorm1d(channels)
        self.pw2 = nn.Conv1d(channels, channels, kernel_size=1)
        self.drop_conv = nn.Dropout(dropout)

        self.norm_ffn = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
            nn.Dropout(dropout),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: [B, T, C]
        h = self.norm_attn(seq)
        h, _ = self.attn(h, h, h, need_weights=False)
        seq = seq + self.drop_attn(h)

        h = self.norm_conv(seq).transpose(1, 2)  # [B, C, T]
        h = F.glu(self.pw1(h), dim=1)
        h = self.dw(h)
        h = F.gelu(self.bn(h))
        h = self.pw2(h).transpose(1, 2)
        seq = seq + self.drop_conv(h)

        seq = seq + self.ffn(self.norm_ffn(seq))
        return seq


class TemporalMLPBlock(nn.Module):
    """
    Minimal temporal ablation block.

    It keeps the same input/output shape as TemporalConformerBlock, but removes
    self-attention and temporal convolution. This is used to test whether the
    original temporal context module is actually contributing beyond a simple
    per-token nonlinear projection.
    """

    def __init__(self, channels: int, dropout: float = 0.15) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
            nn.Dropout(dropout),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: [B, T, C]
        return seq + self.mlp(self.norm(seq))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = TRANSFORMER_DROPOUT, max_len: int = 512) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1), :])


class MLPTimeBackbone(nn.Module):
    """
    MLP temporal backbone from emf_fast_ablation.py, using the selected
    hidden=[192, 192], dropout=0.15 setting.

    It consumes the 96-channel feature map after the two CNN branches and
    returns [B, T, 160] tokens for the existing attention pooling, classifier,
    and denoising branch.
    """

    def __init__(
        self,
        emb_dim: int = EMB_DIM,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = MLP_DROPOUT,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        dims = list(hidden_dims or MLP_HIDDEN_DIMS)
        if len(dims) != 2:
            raise ValueError("MLPTimeBackbone expects two hidden dims, matching emf_fast_ablation.py.")
        self.trunk_channels = MLP_OUTPUT_DIM
        self.mlp = nn.Sequential(
            nn.LayerNorm(96),
            nn.Linear(96, dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dims[0], dims[1]),
            nn.GELU(),
            nn.Linear(dims[1], self.trunk_channels),
        )

    def trunk(self, h: torch.Tensor) -> torch.Tensor:
        return self.mlp(h.mean(dim=2).transpose(1, 2))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.trunk(h)


class LSTMTimeBackbone(nn.Module):
    """
    LSTM replacement for the temporal context block.

    It adapts the teammate-provided backbone to this codebase: the input is the
    96-channel time-frequency feature map right after the two CNN branches, and
    the output is still a [B, T, 160] token sequence so the existing attention
    pooling, classifier, and denoising branch remain unchanged.
    """

    def __init__(self, emb_dim: int = EMB_DIM) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.lstm = nn.LSTM(
            96,
            80,
            batch_first=True,
            bidirectional=True,
            num_layers=1,
        )

    def trunk(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, 96, F, T] -> [B, T, 96] -> [B, T, 160]
        out, _ = self.lstm(h.mean(dim=2).transpose(1, 2))
        return out

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.trunk(h)


class RNNTimeBackbone(nn.Module):
    """
    RNN temporal backbone from emf_fast_ablation.py, using the selected
    hidden=96, layers=2, dropout=0.10 setting.
    """

    def __init__(
        self,
        emb_dim: int = EMB_DIM,
        hidden_size: int = RNN_HIDDEN_SIZE,
        num_layers: int = RNN_NUM_LAYERS,
        bidirectional: bool = RNN_BIDIRECTIONAL,
        dropout: float = RNN_DROPOUT,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.trunk_channels = hidden_size * (2 if bidirectional else 1)
        self.rnn = nn.RNN(
            input_size=96,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def trunk(self, h: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(h.mean(dim=2).transpose(1, 2))
        return out

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.trunk(h)


class TransformerTimeBackbone(nn.Module):
    """
    Transformer temporal backbone with d_model=160, nhead=8, layers=1,
    dim_feedforward=640, dropout=0.15.
    """

    def __init__(
        self,
        emb_dim: int = EMB_DIM,
        nhead: int = TRANSFORMER_NHEAD,
        num_layers: int = TRANSFORMER_NUM_LAYERS,
        dim_feedforward: int = TRANSFORMER_DIM_FEEDFORWARD,
        dropout: float = TRANSFORMER_DROPOUT,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.trunk_channels = 160
        self.proj = nn.Linear(96, 160)
        self.pos_encoder = PositionalEncoding(d_model=160, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=160,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def trunk(self, h: torch.Tensor) -> torch.Tensor:
        seq = self.proj(h.mean(dim=2).transpose(1, 2))
        return self.encoder(self.pos_encoder(seq))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.trunk(h)


class BasicCNNBackbone(nn.Module):
    """
    CNN replacement backbone for the temporal context block.

    It follows the teammate-provided design and consumes the 96-channel feature
    map after the frequency/time CNN branches. The output remains a feature map
    with 160 channels, which is then converted into time tokens for the existing
    attention pooling, classifier, and denoising branch.
    """

    def __init__(self, emb_dim: int = EMB_DIM, width_mult: float = CNN_WIDTH_MULT, depth: int = CNN_DEPTH) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        depth = max(1, int(depth))
        channels = [max(16, int(c * width_mult)) for c in (64, 96, 160)]
        self.trunk_channels = channels[-1]

        blocks: List[nn.Module] = [ConvBNAct(96, channels[0], 3, 1)]
        blocks.extend(ConvBNAct(channels[0], channels[0], 3, 1) for _ in range(depth - 1))
        blocks.append(nn.MaxPool2d(2))
        blocks.append(ConvBNAct(channels[0], channels[1], 3, 1))
        blocks.extend(ConvBNAct(channels[1], channels[1], 3, 1) for _ in range(depth - 1))
        blocks.append(nn.MaxPool2d(2))
        blocks.append(ConvBNAct(channels[1], channels[2], 3, 1))
        blocks.extend(ResidualConvBlock(channels[2]) for _ in range(depth))
        self.body = nn.Sequential(*blocks)

    def trunk(self, h: torch.Tensor) -> torch.Tensor:
        return self.body(h)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.trunk(h)


class ResNetBackbone(nn.Module):
    """
    ResNet-style replacement backbone for the temporal context block.

    It consumes the same 96-channel feature map as the basic CNN backbone, but
    uses residual blocks at each stage. The downstream attention pooling,
    classifier, and embedding denoising branch remain unchanged.
    """

    def __init__(self, emb_dim: int = EMB_DIM, width_mult: float = RESNET_WIDTH_MULT, depth: int = RESNET_DEPTH) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        depth = max(1, int(depth))
        channels = [max(16, int(c * width_mult)) for c in (64, 96, 160)]
        self.trunk_channels = channels[-1]

        blocks: List[nn.Module] = [ConvBNAct(96, channels[0], 3, 1)]
        blocks.extend(ResidualConvBlock(channels[0]) for _ in range(depth))
        blocks.append(nn.MaxPool2d(2))
        blocks.append(ConvBNAct(channels[0], channels[1], 3, 1))
        blocks.extend(ResidualConvBlock(channels[1]) for _ in range(depth))
        blocks.append(nn.MaxPool2d(2))
        blocks.append(ConvBNAct(channels[1], channels[2], 3, 1))
        blocks.extend(ResidualConvBlock(channels[2]) for _ in range(depth))
        self.body = nn.Sequential(*blocks)

    def trunk(self, h: torch.Tensor) -> torch.Tensor:
        return self.body(h)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.trunk(h)


class TimeAttentionPool(nn.Module):
    """
    Interpretable attention over time tokens.
    The exported attention curve answers: which time region contributed most?
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.score = nn.Linear(channels, 1)

    def forward(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # seq: [B, T, C]
        logits = self.score(seq).squeeze(-1)  # [B, T]
        attn = torch.softmax(logits, dim=-1)
        emb = torch.sum(seq * attn.unsqueeze(-1), dim=1)  # [B, C]
        return emb, attn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B, 1], values in [0, 1]
        half = self.dim // 2
        device = t.device
        freqs = torch.exp(
            torch.linspace(math.log(1.0), math.log(1000.0), half, device=device)
        ).view(1, -1)
        ang = t * freqs
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class EMFv1(nn.Module):
    """
    Embedded Music Flow v1.

    Design logic:
    - Use log-mel as a time-frequency representation.
    - Use two music-motivated convolution paths:
      1) frequency-tall kernels for timbre / spectral texture;
      2) time-wide kernels for rhythm / temporal pattern.
    - Use time-attention pooling to keep an interpretable time-importance curve.
    - Add one compact temporal Conformer block after CNN extraction, so the
      model can capture longer rhythm / phrase context without becoming a
      large generic Transformer.
    - Add flow-style clean-embedding prediction as a denoising regularizer.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        emb_dim: int = EMB_DIM,
        temporal_kind: str = "conformer",
        use_denoising: bool = True,
    ) -> None:
        super().__init__()
        self.temporal_kind = temporal_kind
        self.use_denoising = use_denoising
        self.temporal_channels = 160
        self.stem = nn.Sequential(
            ConvBNAct(1, 32, kernel_size=3, padding=1),
            ResidualConvBlock(32),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        self.freq_branch = ConvBNAct(32, 48, kernel_size=(9, 3), padding=(4, 1))
        self.time_branch = ConvBNAct(32, 48, kernel_size=(3, 9), padding=(1, 4))
        self.mix = nn.Sequential(
            ConvBNAct(96, 96, kernel_size=1, padding=0),
            ResidualConvBlock(96),
            nn.MaxPool2d(kernel_size=(2, 2)),
            ConvBNAct(96, 128, kernel_size=3, padding=1),
            ResidualConvBlock(128),
            nn.MaxPool2d(kernel_size=(2, 2)),
            ConvBNAct(128, 160, kernel_size=3, padding=1),
            ResidualConvBlock(160),
        )
        if temporal_kind == "conformer":
            self.temporal_refiner = TemporalConformerBlock(channels=160, num_heads=4, dropout=0.15)
        elif temporal_kind == "mlp":
            self.temporal_refiner = MLPTimeBackbone(emb_dim=emb_dim, hidden_dims=MLP_HIDDEN_DIMS, dropout=MLP_DROPOUT)
            self.temporal_channels = self.temporal_refiner.trunk_channels
        elif temporal_kind == "lstm":
            self.temporal_refiner = LSTMTimeBackbone(emb_dim=emb_dim)
        elif temporal_kind == "rnn":
            self.temporal_refiner = RNNTimeBackbone(
                emb_dim=emb_dim,
                hidden_size=RNN_HIDDEN_SIZE,
                num_layers=RNN_NUM_LAYERS,
                bidirectional=RNN_BIDIRECTIONAL,
                dropout=RNN_DROPOUT,
            )
            self.temporal_channels = self.temporal_refiner.trunk_channels
        elif temporal_kind == "transformer":
            self.temporal_refiner = TransformerTimeBackbone(
                emb_dim=emb_dim,
                nhead=TRANSFORMER_NHEAD,
                num_layers=TRANSFORMER_NUM_LAYERS,
                dim_feedforward=TRANSFORMER_DIM_FEEDFORWARD,
                dropout=TRANSFORMER_DROPOUT,
            )
            self.temporal_channels = self.temporal_refiner.trunk_channels
        elif temporal_kind == "cnn":
            self.temporal_refiner = BasicCNNBackbone(emb_dim=emb_dim, width_mult=CNN_WIDTH_MULT, depth=CNN_DEPTH)
            self.temporal_channels = self.temporal_refiner.trunk_channels
        elif temporal_kind == "resnet":
            self.temporal_refiner = ResNetBackbone(emb_dim=emb_dim, width_mult=RESNET_WIDTH_MULT, depth=RESNET_DEPTH)
            self.temporal_channels = self.temporal_refiner.trunk_channels
        else:
            raise ValueError(f"Unknown temporal_kind: {temporal_kind}")
        self.pool = TimeAttentionPool(channels=self.temporal_channels)
        self.to_emb = nn.Sequential(
            nn.LayerNorm(self.temporal_channels),
            nn.Linear(self.temporal_channels, emb_dim),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.LayerNorm(emb_dim),
        )

        self.time_emb = SinusoidalTimeEmbedding(TIME_EMB_DIM)
        self.denoiser = nn.Sequential(
            nn.Linear(emb_dim + TIME_EMB_DIM, 256),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, emb_dim),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Dropout(0.25),
            nn.Linear(emb_dim, num_classes),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        h_freq = self.freq_branch(h)
        h_time = self.time_branch(h)
        h = torch.cat([h_freq, h_time], dim=1)
        if self.temporal_kind in {"mlp", "lstm", "rnn", "transformer"}:
            fmap = h
            seq = self.temporal_refiner(h)
        elif self.temporal_kind in {"cnn", "resnet"}:
            fmap = self.temporal_refiner(h)
            seq = fmap.mean(dim=2).transpose(1, 2)
        else:
            fmap = self.mix(h)
            # Compress frequency dimension into time tokens, then refine temporal context.
            seq = fmap.mean(dim=2).transpose(1, 2)  # [B, T, C]
            seq = self.temporal_refiner(seq)
        pooled, attn = self.pool(seq)
        emb = self.to_emb(pooled)
        return emb, attn, fmap

    def denoise(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = self.time_emb(t)
        return self.denoiser(torch.cat([z, te], dim=-1))

    def classify(self, emb: torch.Tensor) -> torch.Tensor:
        return self.classifier(emb)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb, attn, fmap = self.encode(x)
        logits = self.classify(emb)
        return logits, emb, attn


# ============================================================
# SpecAugment
# ============================================================
def apply_specaugment(x: torch.Tensor) -> torch.Tensor:
    if random.random() > SPEC_AUG_PROB:
        return x
    x_aug = x.clone()
    b, c, f, t = x_aug.shape
    for bi in range(b):
        for _ in range(FREQ_MASKS):
            width = random.randint(0, MAX_FREQ_MASK)
            if width <= 0 or width >= f:
                continue
            start = random.randint(0, f - width)
            x_aug[bi, :, start : start + width, :] = 0.0
        for _ in range(TIME_MASKS):
            width = random.randint(0, MAX_TIME_MASK)
            if width <= 0 or width >= t:
                continue
            start = random.randint(0, t - width)
            x_aug[bi, :, :, start : start + width] = 0.0
    return x_aug


def sample_t(batch_size: int, device: torch.device) -> torch.Tensor:
    # Logit-normal-like sampling without external dependencies.
    u = torch.randn(batch_size, 1, device=device) * 0.8 - 0.6
    t = torch.sigmoid(u)
    return torch.clamp(t, T_MIN, T_MAX)


# ============================================================
# Metrics and evaluation
# ============================================================
def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, float]:
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        conf[int(t), int(p)] += 1
    acc = float(np.trace(conf) / max(conf.sum(), 1))
    recalls = []
    precisions = []
    f1s = []
    for k in range(num_classes):
        tp = conf[k, k]
        fn = conf[k, :].sum() - tp
        fp = conf[:, k].sum() - tp
        rec = tp / max(tp + fn, 1)
        prec = tp / max(tp + fp, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        recalls.append(rec)
        precisions.append(prec)
        f1s.append(f1)
    return {
        "acc": acc,
        "balanced_acc": float(np.mean(recalls)),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
    }


@torch.no_grad()
def collect_logits(model: EMFv1, loader: DataLoader, device: torch.device):
    model.eval()
    logits_all: List[np.ndarray] = []
    y_all: List[int] = []
    song_ids: List[str] = []
    genres: List[str] = []
    mel_paths: List[str] = []
    attn_peaks: List[float] = []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        logits, _, attn = model(x)
        logits_all.append(logits.detach().cpu().numpy())
        y_all.extend(y.detach().cpu().numpy().tolist())
        song_ids.extend(list(batch["song_id"]))
        genres.extend(list(batch["genre"]))
        mel_paths.extend(list(batch["mel_path"]))
        peak = torch.argmax(attn, dim=-1).detach().cpu().numpy()
        attn_peaks.extend(peak.astype(float).tolist())
    return (
        np.concatenate(logits_all, axis=0),
        np.asarray(y_all, dtype=np.int64),
        song_ids,
        genres,
        mel_paths,
        np.asarray(attn_peaks, dtype=np.float32),
    )


def softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / max(temperature, 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)


def nll_from_logits(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    probs = softmax_np(logits, temperature)
    idx = np.arange(len(labels))
    return float(-np.log(np.maximum(probs[idx, labels], 1e-12)).mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    best_t = 1.0
    best_nll = float("inf")
    for t in np.linspace(0.50, 5.00, 91):
        nll = nll_from_logits(logits, labels, float(t))
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)
    return best_t


def evaluate_from_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    song_ids: List[str],
    temperature: float,
    num_classes: int,
) -> Dict[str, object]:
    probs = softmax_np(logits, temperature)
    seg_pred = probs.argmax(axis=1)
    seg_metrics = classification_metrics(labels, seg_pred, num_classes)

    song_prob_sum: Dict[str, np.ndarray] = {}
    song_count: Dict[str, int] = defaultdict(int)
    song_label: Dict[str, int] = {}
    for p, y, sid in zip(probs, labels, song_ids):
        if sid not in song_prob_sum:
            song_prob_sum[sid] = np.zeros(num_classes, dtype=np.float64)
            song_label[sid] = int(y)
        song_prob_sum[sid] += p
        song_count[sid] += 1

    song_true = []
    song_pred = []
    for sid, psum in song_prob_sum.items():
        avg_p = psum / max(song_count[sid], 1)
        song_true.append(song_label[sid])
        song_pred.append(int(avg_p.argmax()))
    song_metrics = classification_metrics(
        np.asarray(song_true, dtype=np.int64), np.asarray(song_pred, dtype=np.int64), num_classes
    )

    return {
        "segment": seg_metrics,
        "song": song_metrics,
        "num_segments": int(len(labels)),
        "num_songs": int(len(song_true)),
        "temperature": float(temperature),
    }


def save_prediction_csv(
    path: Path,
    logits: np.ndarray,
    labels: np.ndarray,
    song_ids: List[str],
    genres: List[str],
    mel_paths: List[str],
    attn_peaks: np.ndarray,
    id_to_label: Dict[int, str],
    temperature: float,
) -> None:
    probs = softmax_np(logits, temperature)
    pred = probs.argmax(axis=1)
    rows: List[Dict[str, object]] = []
    for i in range(len(labels)):
        top3 = probs[i].argsort()[-3:][::-1]
        rows.append(
            {
                "song_id": song_ids[i],
                "true_label": id_to_label[int(labels[i])],
                "pred_label": id_to_label[int(pred[i])],
                "correct": int(pred[i] == labels[i]),
                "confidence": float(probs[i, pred[i]]),
                "top1_prob": float(probs[i, top3[0]]),
                "top2_label": id_to_label[int(top3[1])],
                "top2_prob": float(probs[i, top3[1]]),
                "top3_label": id_to_label[int(top3[2])],
                "top3_prob": float(probs[i, top3[2]]),
                "attn_peak_index": float(attn_peaks[i]),
                "genre_folder": genres[i],
                "mel_path": mel_paths[i],
            }
        )
    write_csv(
        path,
        rows,
        [
            "song_id",
            "true_label",
            "pred_label",
            "correct",
            "confidence",
            "top1_prob",
            "top2_label",
            "top2_prob",
            "top3_label",
            "top3_prob",
            "attn_peak_index",
            "genre_folder",
            "mel_path",
        ],
    )


# ============================================================
# Training
# ============================================================
def train_one_epoch(
    model: EMFv1,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    ce_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    mse_fn = nn.MSELoss()
    total_loss = 0.0
    total_ce = 0.0
    total_mse = 0.0
    total_den_ce = 0.0
    total_n = 0
    correct = 0

    for batch_idx, batch in enumerate(loader, start=1):
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        x = apply_specaugment(x)

        optimizer.zero_grad(set_to_none=True)
        logits, emb, _ = model(x)
        ce_clean = ce_fn(logits, y)

        if getattr(model, "use_denoising", True):
            t = sample_t(emb.shape[0], device)
            target = emb.detach()
            eps = torch.randn_like(target)
            z = t * target + (1.0 - t) * eps
            emb_hat = model.denoise(z, t)
            mse = mse_fn(emb_hat, target)
            logits_den = model.classify(emb_hat)
            ce_den = ce_fn(logits_den, y)
            loss = ce_clean + DENOISE_WEIGHT * mse + DENOISED_CE_WEIGHT * ce_den
        else:
            mse = emb.new_tensor(0.0)
            ce_den = emb.new_tensor(0.0)
            loss = ce_clean
        loss.backward()
        if GRAD_CLIP_NORM > 0:
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        bs = y.size(0)
        total_n += bs
        total_loss += float(loss.item()) * bs
        total_ce += float(ce_clean.item()) * bs
        total_mse += float(mse.item()) * bs
        total_den_ce += float(ce_den.item()) * bs
        correct += int((logits.argmax(dim=1) == y).sum().item())

        if batch_idx % 80 == 0:
            print(
                f"[train] epoch={epoch:03d} step={batch_idx:04d}/{len(loader)} "
                f"loss={total_loss/total_n:.4f} acc={correct/total_n:.4f}"
            )

    return {
        "loss": total_loss / max(total_n, 1),
        "ce": total_ce / max(total_n, 1),
        "mse": total_mse / max(total_n, 1),
        "den_ce": total_den_ce / max(total_n, 1),
        "acc": correct / max(total_n, 1),
    }


def make_loader(dataset: Dataset, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        drop_last=False,
    )


def train_model(rows: List[Dict[str, object]], label_to_id: Dict[str, int], mean: float, std: float) -> None:
    ensure_dir(OUT_DIR)
    device = torch.device(DEVICE)
    id_to_label = {v: k for k, v in label_to_id.items()}
    save_json(OUT_DIR / "label_to_id.json", label_to_id)

    train_ds = MelGenreDataset(rows, "train", mean, std)
    val_ds = MelGenreDataset(rows, "val", mean, std)
    test_ds = MelGenreDataset(rows, "test", mean, std)

    train_loader = make_loader(train_ds, shuffle=True)
    val_loader = make_loader(val_ds, shuffle=False)
    test_loader = make_loader(test_ds, shuffle=False)

    model = EMFv1(num_classes=len(label_to_id), temporal_kind=TEMPORAL_KIND, use_denoising=USE_DENOISING).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_song_acc = -1.0
    bad_epochs = 0
    history: List[Dict[str, object]] = []
    best_path = OUT_DIR / "best_emf_v1.pt"

    print_device_info(device)
    print(f"[run] seed={SEED} temporal={TEMPORAL_KIND} denoise={USE_DENOISING}")
    print(f"[out] {OUT_DIR}")
    print(f"[data] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    print(f"[classes] {label_to_id}")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_stats = train_one_epoch(model, train_loader, optimizer, device, epoch)
        scheduler.step()

        val_logits, val_y, val_sids, _, _, _ = collect_logits(model, val_loader, device)
        val_temp = fit_temperature(val_logits, val_y)
        val_eval = evaluate_from_logits(val_logits, val_y, val_sids, val_temp, len(label_to_id))
        val_song_acc = float(val_eval["song"]["acc"])
        val_seg_acc = float(val_eval["segment"]["acc"])

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train_stats,
            "val": val_eval,
            "seconds": round(time.time() - t0, 2),
        }
        history.append(row)
        save_json(OUT_DIR / "history.json", {"history": history})

        print(
            f"[epoch {epoch:03d}] train_loss={train_stats['loss']:.4f} "
            f"train_acc={train_stats['acc']:.4f} "
            f"val_seg_acc={val_seg_acc:.4f} val_song_acc={val_song_acc:.4f} "
            f"T={val_temp:.2f} time={row['seconds']}s"
        )

        if val_song_acc > best_song_acc:
            best_song_acc = val_song_acc
            bad_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "label_to_id": label_to_id,
                    "mean": mean,
                    "std": std,
                    "val_temperature": val_temp,
                    "epoch": epoch,
                    "val_eval": val_eval,
                    "config": get_config_dict(),
                },
                str(best_path),
            )
            print(f"[save] best model -> {best_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= EARLY_STOP_PATIENCE:
                print(f"[early-stop] no val song acc improvement for {EARLY_STOP_PATIENCE} epochs")
                break

    if RUN_TEST:
        print("[test] loading best checkpoint")
        ckpt = torch.load(str(best_path), map_location=device)
        model.load_state_dict(ckpt["model"])
        val_temperature = float(ckpt.get("val_temperature", 1.0))

        test_logits, test_y, test_sids, test_genres, test_paths, test_attn = collect_logits(
            model, test_loader, device
        )
        test_eval = evaluate_from_logits(
            test_logits, test_y, test_sids, val_temperature, len(label_to_id)
        )
        save_json(OUT_DIR / "test_metrics.json", test_eval)
        save_prediction_csv(
            OUT_DIR / "test_predictions_v1.csv",
            test_logits,
            test_y,
            test_sids,
            test_genres,
            test_paths,
            test_attn,
            id_to_label,
            val_temperature,
        )
        print("[test] metrics:")
        print(json.dumps(test_eval, indent=2, ensure_ascii=False))

        if EXPORT_EXPLANATIONS:
            export_explanations(model, test_ds, device, id_to_label, mean, std)


# ============================================================
# Explanation export
# ============================================================
@torch.no_grad()
def time_occlusion_importance(
    model: EMFv1,
    x: torch.Tensor,
    pred_class: int,
    device: torch.device,
    window: int = 12,
    stride: int = 6,
) -> np.ndarray:
    """
    Model-agnostic time occlusion explanation.
    A larger value means masking that time region reduces the predicted class
    probability more, so the model relied on that region more.
    """
    model.eval()
    x = x.to(device)
    logits, _, _ = model(x)
    base_prob = torch.softmax(logits, dim=1)[0, pred_class].item()
    _, _, _, total_t = x.shape
    scores = np.zeros(total_t, dtype=np.float32)
    counts = np.zeros(total_t, dtype=np.float32)

    for start in range(0, total_t, stride):
        end = min(total_t, start + window)
        if end <= start:
            continue
        x_mask = x.clone()
        # 0 is the standardized train mean after normalization.
        x_mask[:, :, :, start:end] = 0.0
        logits_m, _, _ = model(x_mask)
        prob_m = torch.softmax(logits_m, dim=1)[0, pred_class].item()
        drop = max(base_prob - prob_m, 0.0)
        scores[start:end] += drop
        counts[start:end] += 1.0
        if end == total_t:
            break

    scores = scores / np.maximum(counts, 1.0)
    if scores.max() > 1e-8:
        scores = scores / scores.max()
    return scores


def export_explanations(
    model: EMFv1,
    dataset: MelGenreDataset,
    device: torch.device,
    id_to_label: Dict[int, str],
    mean: float,
    std: float,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[explain] matplotlib not available, skip explanations: {exc}")
        return

    out_dir = OUT_DIR / "explain"
    ensure_dir(out_dir)
    model.eval()

    rng = random.Random(SEED)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    indices = indices[: min(SAVE_EXPLANATION_EXAMPLES, len(indices))]

    rows: List[Dict[str, object]] = []
    for count, idx in enumerate(indices):
        item = dataset[idx]
        x = item["x"].unsqueeze(0).to(device)
        y = int(item["y"].item())
        with torch.no_grad():
            logits, _, attn = model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred = int(np.argmax(probs))
            attn_np = attn[0].cpu().numpy()

        mel_norm = item["x"].squeeze(0).numpy()
        mel_db = mel_norm * std + mean
        attn_up = np.interp(
            np.linspace(0, len(attn_np) - 1, TARGET_FRAMES),
            np.arange(len(attn_np)),
            attn_np,
        )
        attn_up = attn_up / max(attn_up.max(), 1e-8)
        occ = time_occlusion_importance(model, x, pred, device)

        title = f"true={id_to_label[y]} pred={id_to_label[pred]} conf={probs[pred]:.2f}"
        fig = plt.figure(figsize=(8, 5.2))
        ax1 = fig.add_subplot(3, 1, 1)
        ax1.imshow(mel_db, aspect="auto", origin="lower")
        ax1.set_title(title)
        ax1.set_ylabel("mel bins")
        ax2 = fig.add_subplot(3, 1, 2)
        ax2.plot(attn_up)
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("attention")
        ax3 = fig.add_subplot(3, 1, 3)
        ax3.plot(occ)
        ax3.set_ylim(0, 1.05)
        ax3.set_ylabel("occlusion")
        ax3.set_xlabel("mel frames")
        fig.tight_layout()
        fig_path = out_dir / f"explain_{count:02d}.png"
        fig.savefig(str(fig_path), dpi=140)
        plt.close(fig)

        rows.append(
            {
                "figure": str(fig_path),
                "mel_path": item["mel_path"],
                "song_id": item["song_id"],
                "true_label": id_to_label[y],
                "pred_label": id_to_label[pred],
                "confidence": float(probs[pred]),
            }
        )

    write_csv(
        out_dir / "explain_index.csv",
        rows,
        ["figure", "mel_path", "song_id", "true_label", "pred_label", "confidence"],
    )
    print(f"[explain] saved to: {out_dir}")


# ============================================================
# Config snapshot
# ============================================================
def get_config_dict() -> Dict[str, object]:
    return {
        "SEED": SEED,
        "ROOT_DIR": str(ROOT_DIR),
        "SEG_DIR": str(SEG_DIR),
        "MEL_DIR": str(MEL_DIR),
        "OUT_DIR": str(OUT_DIR),
        "SAMPLE_RATE": SAMPLE_RATE,
        "N_FFT": N_FFT,
        "HOP_LENGTH": HOP_LENGTH,
        "N_MELS": N_MELS,
        "TARGET_FRAMES": TARGET_FRAMES,
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "LEARNING_RATE": LEARNING_RATE,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "LABEL_SMOOTHING": LABEL_SMOOTHING,
        "DENOISE_WEIGHT": DENOISE_WEIGHT,
        "DENOISED_CE_WEIGHT": DENOISED_CE_WEIGHT,
        "TEMPORAL_REFINER": TEMPORAL_KIND,
        "USE_DENOISING": USE_DENOISING,
        "CNN_WIDTH_MULT": CNN_WIDTH_MULT,
        "CNN_DEPTH": CNN_DEPTH,
        "RESNET_WIDTH_MULT": RESNET_WIDTH_MULT,
        "RESNET_DEPTH": RESNET_DEPTH,
        "MLP_HIDDEN_DIMS": MLP_HIDDEN_DIMS,
        "MLP_OUTPUT_DIM": MLP_OUTPUT_DIM,
        "MLP_DROPOUT": MLP_DROPOUT,
        "RNN_HIDDEN_SIZE": RNN_HIDDEN_SIZE,
        "RNN_NUM_LAYERS": RNN_NUM_LAYERS,
        "RNN_BIDIRECTIONAL": RNN_BIDIRECTIONAL,
        "RNN_DROPOUT": RNN_DROPOUT,
        "TRANSFORMER_NHEAD": TRANSFORMER_NHEAD,
        "TRANSFORMER_NUM_LAYERS": TRANSFORMER_NUM_LAYERS,
        "TRANSFORMER_DIM_FEEDFORWARD": TRANSFORMER_DIM_FEEDFORWARD,
        "TRANSFORMER_DROPOUT": TRANSFORMER_DROPOUT,
        "EXPLANATIONS": "time_attention + time_occlusion",
        "SPEC_AUG_PROB": SPEC_AUG_PROB,
        "DEVICE": DEVICE,
        "FORCE_CUDA": FORCE_CUDA,
    }


# ============================================================
# Main / seed + architecture sweep
# ============================================================
def str_to_bool(value: str) -> bool:
    v = str(value).strip().lower()
    if v in {"true", "1", "yes", "y", "on"}:
        return True
    if v in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def parse_int_list(text_value: str) -> List[int]:
    return [int(x.strip()) for x in text_value.split(",") if x.strip()]


def parse_name_list(text_value: str) -> List[str]:
    return [x.strip().lower() for x in text_value.split(",") if x.strip()]


def structure_to_knobs(name: str) -> Tuple[str, bool, str]:
    """
    Supported structure names:
    - full / conformer: original v1, TemporalConformerBlock + denoising
    - mlp / rnn / lstm / cnn / resnet / transformer: replace temporal backbone, keep denoising
    - nodn / no_denoise: original temporal block, turn off denoising branch
    - mlp_nodn: MLP temporal ablation + no denoising
    """
    key = name.strip().lower()
    if key in {"full", "base", "conformer", "con_dn"}:
        return "conformer", True, "full"
    if key in {"mlp", "abl_mlp", "mlp_dn"}:
        return "mlp", True, "mlp"
    if key in {"rnn", "rnn_time"}:
        return "rnn", True, "rnn"
    if key in {"lstm", "lstm_time"}:
        return "lstm", True, "lstm"
    if key in {"cnn", "basic_cnn"}:
        return "cnn", True, "cnn"
    if key in {"resnet"}:
        return "resnet", True, "resnet"
    if key in {"transformer", "transformer_time"}:
        return "transformer", True, "transformer"
    if key in {"nodn", "no_denoise", "con_nodn", "no_dn"}:
        return "conformer", False, "nodn"
    if key in {"mlp_nodn", "mlp_no_denoise", "mlp_no_dn"}:
        return "mlp", False, "mlp_nodn"
    raise ValueError(f"Unknown structure name: {name}")


def read_test_summary(out_dir: Path, seed: int, structure: str, temporal_kind: str, use_denoising: bool) -> Dict[str, object]:
    metrics_path = out_dir / "test_metrics.json"
    row: Dict[str, object] = {
        "seed": seed,
        "structure": structure,
        "temporal_kind": temporal_kind,
        "use_denoising": use_denoising,
        "out_dir": str(out_dir),
    }
    if not metrics_path.exists():
        row.update(
            {
                "segment_acc": "",
                "segment_macro_f1": "",
                "song_acc": "",
                "song_macro_f1": "",
                "temperature": "",
            }
        )
        return row
    obj = load_json(metrics_path)
    row.update(
        {
            "segment_acc": float(obj["segment"]["acc"]),
            "segment_macro_f1": float(obj["segment"]["macro_f1"]),
            "song_acc": float(obj["song"]["acc"]),
            "song_macro_f1": float(obj["song"]["macro_f1"]),
            "temperature": float(obj.get("temperature", 1.0)),
        }
    )
    return row


def run_experiment_once(
    seed: int,
    structure: str,
    temporal_kind: str,
    use_denoising: bool,
    out_dir: Path,
    build_mel_cache_flag: bool,
    export_explanations_flag: bool,
) -> Dict[str, object]:
    global SEED, OUT_DIR, BUILD_MEL_CACHE, EXPORT_EXPLANATIONS, TEMPORAL_KIND, USE_DENOISING

    SEED = int(seed)
    OUT_DIR = out_dir
    BUILD_MEL_CACHE = bool(build_mel_cache_flag)
    EXPORT_EXPLANATIONS = bool(export_explanations_flag)
    TEMPORAL_KIND = temporal_kind
    USE_DENOISING = bool(use_denoising)

    print("\n" + "=" * 72)
    print(f"[start] structure={structure} seed={SEED}")
    print(f"[knobs] temporal={TEMPORAL_KIND} denoise={USE_DENOISING}")
    print(f"[paths] SEG_DIR={SEG_DIR}")
    print(f"[paths] MEL_DIR={MEL_DIR}")
    print(f"[paths] OUT_DIR={OUT_DIR}")
    print("=" * 72)

    set_seed(SEED)
    validate_cuda_device()
    ensure_dir(OUT_DIR)
    save_json(OUT_DIR / "config.json", get_config_dict())

    rows, label_to_id = list_wav_rows()
    write_csv(
        OUT_DIR / "mel_items.csv",
        rows,
        ["split", "genre", "label", "song_id", "segment_index", "wav_path", "mel_path"],
    )

    if BUILD_MEL_CACHE:
        build_mel_cache(rows)

    mean, std = compute_train_stats(rows)

    if TRAIN_MODEL:
        train_model(rows, label_to_id, mean, std)
    else:
        print("TRAIN_MODEL=False, only built mel cache.")

    return read_test_summary(OUT_DIR, SEED, structure, TEMPORAL_KIND, USE_DENOISING)



def generate_random_seed() -> int:
    return random.SystemRandom().randint(1, 999999)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the final MusicFlowNet model only. "
            "This script keeps the original v1 model and training protocol, "
            "but uses a random seed by default so different runs do not silently repeat."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Training seed. Use -1 for a random seed. The actual seed is printed and saved.",
    )
    parser.add_argument(
        "--run_root",
        type=str,
        default=str(TRAIN_ROOT_DIR),
        help="Root folder for training outputs.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help="Optional run folder name. Default: <temporal>_<epochs>ep_s<seed>.",
    )
    parser.add_argument(
        "--temporal",
        type=str,
        default=TEMPORAL_KIND,
        choices=["conformer", "mlp", "lstm", "rnn", "cnn", "resnet", "transformer"],
        help="Temporal backbone to train. Default is cnn for the direct VSCode run.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Number of training epochs. Default follows EPOCHS=60 unless manually overridden.",
    )
    parser.add_argument(
        "--build_mel",
        type=str_to_bool,
        default=True,
        help="Build or check log-mel cache before training. Existing cache files are skipped.",
    )
    parser.add_argument(
        "--export_explain",
        type=str_to_bool,
        default=False,
        help="Export attention/occlusion explanation examples after test.",
    )
    parser.add_argument(
        "--force_cuda",
        type=str_to_bool,
        default=True,
        help="Fail fast if CUDA is not visible.",
    )
    parser.add_argument(
        "--copy_to_default_ckpt",
        type=str_to_bool,
        default=True,
        help=(
            "Copy the best checkpoint to the local emf_v1_out\\best_emf_v1.pt "
            "so predict_audio.py can use it by default."
        ),
    )
    return parser


def main() -> None:
    global FORCE_CUDA, EPOCHS

    args = build_arg_parser().parse_args()
    FORCE_CUDA = bool(args.force_cuda)
    EPOCHS = int(args.epochs)

    seed = generate_random_seed() if int(args.seed) < 0 else int(args.seed)
    run_root = Path(args.run_root)
    temporal_kind = args.temporal.strip().lower()
    run_name = args.run_name.strip() or f"{temporal_kind}_{EPOCHS}ep_s{seed}"
    out_dir = run_root / run_name

    row = run_experiment_once(
        seed=seed,
        structure=temporal_kind,
        temporal_kind=temporal_kind,
        use_denoising=True,
        out_dir=out_dir,
        build_mel_cache_flag=args.build_mel,
        export_explanations_flag=args.export_explain,
    )

    summary_path = run_root / "train_summary.csv"
    existing_rows: List[Dict[str, object]] = []
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8-sig", newline="") as f:
            existing_rows.extend(list(csv.DictReader(f)))
    existing_rows.append(row)
    write_csv(
        summary_path,
        existing_rows,
        [
            "seed",
            "structure",
            "temporal_kind",
            "use_denoising",
            "segment_acc",
            "segment_macro_f1",
            "song_acc",
            "song_macro_f1",
            "temperature",
            "out_dir",
        ],
    )

    best_ckpt = out_dir / "best_emf_v1.pt"
    if args.copy_to_default_ckpt and best_ckpt.exists():
        ensure_dir(DEFAULT_CKPT_DIR)
        default_ckpt = DEFAULT_CKPT_DIR / "best_emf_v1.pt"
        shutil.copy2(str(best_ckpt), str(default_ckpt))
        with (DEFAULT_CKPT_DIR / "latest_checkpoint_path.txt").open("w", encoding="utf-8") as f:
            f.write(str(best_ckpt))
        print(f"[copy] latest checkpoint copied to: {default_ckpt}")

    print(f"[summary] {summary_path}")
    print(f"[seed] {seed}")
    print(f"[checkpoint] {best_ckpt}")
    print(row)


if __name__ == "__main__":
    main()
