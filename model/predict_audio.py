# predict_audio.py
# Inference script for MusicFlowNet.
# It loads a trained checkpoint, cuts an input audio file into clips, predicts each clip,
# and averages clip probabilities for song-level prediction.

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal

# ============================================================
# Project paths
# ============================================================
ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_CKPT = ROOT_DIR / "emf_v1_out" / "best_emf_v1.pt"
DEFAULT_OUT_DIR = ROOT_DIR / "infer_out"

# ============================================================
# Audio / log-mel settings, kept consistent with training
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

SEGMENT_SEC = 3.0
HOP_SEC = 1.5
HPF_CUTOFF_HZ = 20.0
HPF_ORDER = 2
TRIM_TOP_DB = 30
TRIM_FRAME_LENGTH = 2048
TRIM_HOP_LENGTH = 512
MAX_TRIM_PER_SIDE_SEC = 0.8

NUM_CLASSES = 10
TIME_EMB_DIM = 64
EMB_DIM = 128
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
# Utilities
# ============================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


# ============================================================
# Model definition: must match emf_train_v1_updated.py
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
        self.dw = nn.Conv1d(channels, channels, kernel_size=7, padding=3, groups=channels)
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
        h = self.norm_attn(seq)
        h, _ = self.attn(h, h, h, need_weights=False)
        seq = seq + self.drop_attn(h)

        h = self.norm_conv(seq).transpose(1, 2)
        h = F.glu(self.pw1(h), dim=1)
        h = self.dw(h)
        h = F.gelu(self.bn(h))
        h = self.pw2(h).transpose(1, 2)
        seq = seq + self.drop_conv(h)

        seq = seq + self.ffn(self.norm_ffn(seq))
        return seq


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
            raise ValueError("MLPTimeBackbone expects two hidden dims.")
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

    This mirrors the training code: it consumes the 96-channel feature map from
    the two CNN branches and returns [B, T, 160] tokens for the existing pooling
    and classifier.
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
        out, _ = self.lstm(h.mean(dim=2).transpose(1, 2))
        return out

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.trunk(h)


class RNNTimeBackbone(nn.Module):
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

    It consumes the 96-channel feature map after the frequency/time CNN
    branches and returns a 160-channel feature map. The rest of the inference
    path is kept identical to training: average frequency into time tokens,
    attention-pool them, then classify the embedding.
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

    It consumes the same 96-channel feature map as BasicCNNBackbone, but uses
    residual blocks at each stage. This must match the training-time ResNet
    variant so ResNet checkpoints can be loaded directly for inference.
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
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.score = nn.Linear(channels, 1)

    def forward(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.score(seq).squeeze(-1)
        attn = torch.softmax(logits, dim=-1)
        emb = torch.sum(seq * attn.unsqueeze(-1), dim=1)
        return emb, attn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = t.device
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), half, device=device)).view(1, -1)
        ang = t * freqs
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class EMFv1(nn.Module):
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        emb_dim: int = EMB_DIM,
        temporal_kind: str = "conformer",
        cnn_width_mult: float = CNN_WIDTH_MULT,
        cnn_depth: int = CNN_DEPTH,
        resnet_width_mult: float = RESNET_WIDTH_MULT,
        resnet_depth: int = RESNET_DEPTH,
    ) -> None:
        super().__init__()
        self.temporal_kind = temporal_kind
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
            self.temporal_refiner = MLPTimeBackbone(emb_dim=emb_dim)
            self.temporal_channels = self.temporal_refiner.trunk_channels
        elif temporal_kind == "lstm":
            self.temporal_refiner = LSTMTimeBackbone(emb_dim=emb_dim)
        elif temporal_kind == "rnn":
            self.temporal_refiner = RNNTimeBackbone(emb_dim=emb_dim)
            self.temporal_channels = self.temporal_refiner.trunk_channels
        elif temporal_kind == "transformer":
            self.temporal_refiner = TransformerTimeBackbone(emb_dim=emb_dim)
            self.temporal_channels = self.temporal_refiner.trunk_channels
        elif temporal_kind == "cnn":
            self.temporal_refiner = BasicCNNBackbone(emb_dim=emb_dim, width_mult=cnn_width_mult, depth=cnn_depth)
            self.temporal_channels = self.temporal_refiner.trunk_channels
        elif temporal_kind == "resnet":
            self.temporal_refiner = ResNetBackbone(emb_dim=emb_dim, width_mult=resnet_width_mult, depth=resnet_depth)
            self.temporal_channels = self.temporal_refiner.trunk_channels
        else:
            raise ValueError(f"Unknown temporal_kind for inference: {temporal_kind}")
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
            seq = fmap.mean(dim=2).transpose(1, 2)
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
# Audio preprocessing for inference
# ============================================================
def remove_dc(y: np.ndarray) -> np.ndarray:
    return (y - np.mean(y)).astype(np.float32)


def highpass_20hz(y: np.ndarray) -> np.ndarray:
    if len(y) < 64:
        return y.astype(np.float32)
    nyquist = SAMPLE_RATE * 0.5
    cutoff = HPF_CUTOFF_HZ / nyquist
    b, a = signal.butter(HPF_ORDER, cutoff, btype="highpass")
    return signal.filtfilt(b, a, y).astype(np.float32)


def trim_edges_only(y: np.ndarray) -> np.ndarray:
    if len(y) == 0:
        return y
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
        return y.astype(np.float32)
    return y[s:e].astype(np.float32)


def load_and_preprocess_audio(audio_path: Path) -> np.ndarray:
    y, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    if y.size == 0:
        raise ValueError(f"Empty audio: {audio_path}")
    y = y.astype(np.float32)
    y = remove_dc(y)
    y = highpass_20hz(y)
    y = trim_edges_only(y)
    return y.astype(np.float32)


def cut_audio_segments(y: np.ndarray, segment_sec: float = SEGMENT_SEC, hop_sec: float = HOP_SEC) -> List[Tuple[int, float, float, np.ndarray]]:
    seg_len = int(round(segment_sec * SAMPLE_RATE))
    hop_len = int(round(hop_sec * SAMPLE_RATE))
    if len(y) < seg_len:
        y_pad = np.pad(y, (0, seg_len - len(y)), mode="constant")
        return [(0, 0.0, segment_sec, y_pad.astype(np.float32))]

    segments: List[Tuple[int, float, float, np.ndarray]] = []
    last_start = len(y) - seg_len
    starts = list(range(0, last_start + 1, hop_len))
    if starts and starts[-1] != last_start:
        starts.append(last_start)

    for idx, start in enumerate(starts):
        end = start + seg_len
        seg = y[start:end]
        if len(seg) < seg_len:
            seg = np.pad(seg, (0, seg_len - len(seg)), mode="constant")
        segments.append((idx, start / SAMPLE_RATE, end / SAMPLE_RATE, seg.astype(np.float32)))
    return segments


def segment_to_logmel(seg: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=seg,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )
    logmel = librosa.power_to_db(mel, ref=1.0)
    logmel = np.clip(logmel, TOP_DB_FLOOR, TOP_DB_CEIL).astype(np.float32)
    if logmel.shape[1] < TARGET_FRAMES:
        pad = TARGET_FRAMES - logmel.shape[1]
        logmel = np.pad(logmel, ((0, 0), (0, pad)), mode="constant", constant_values=TOP_DB_FLOOR)
    elif logmel.shape[1] > TARGET_FRAMES:
        logmel = logmel[:, :TARGET_FRAMES]
    return logmel.astype(np.float32)


def softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / max(float(temperature), 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)


def infer_temporal_kind(ckpt: Dict[str, object]) -> str:
    cfg = ckpt.get("config", {})
    if not isinstance(cfg, dict):
        return "conformer"
    raw = str(cfg.get("TEMPORAL_REFINER", cfg.get("temporal_kind", "conformer"))).lower()
    if "transformer" in raw:
        return "transformer"
    if "lstm" in raw:
        return "lstm"
    if "rnn" in raw:
        return "rnn"
    if "cnn" in raw:
        return "cnn"
    if "resnet" in raw:
        return "resnet"
    if "mlp" in raw:
        return "mlp"
    return "conformer"


def load_checkpoint(ckpt_path: Path, device: torch.device):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=device)
    label_to_id = ckpt["label_to_id"]
    id_to_label = {int(v): str(k) for k, v in label_to_id.items()}
    temporal_kind = infer_temporal_kind(ckpt)
    cfg = ckpt.get("config", {})
    if isinstance(cfg, dict):
        cnn_width_mult = float(cfg.get("CNN_WIDTH_MULT", CNN_WIDTH_MULT))
        cnn_depth = int(cfg.get("CNN_DEPTH", CNN_DEPTH))
        resnet_width_mult = float(cfg.get("RESNET_WIDTH_MULT", cfg.get("CNN_WIDTH_MULT", RESNET_WIDTH_MULT)))
        resnet_depth = int(cfg.get("RESNET_DEPTH", cfg.get("CNN_DEPTH", RESNET_DEPTH)))
    else:
        cnn_width_mult = CNN_WIDTH_MULT
        cnn_depth = CNN_DEPTH
        resnet_width_mult = RESNET_WIDTH_MULT
        resnet_depth = RESNET_DEPTH
    model = EMFv1(
        num_classes=len(label_to_id),
        temporal_kind=temporal_kind,
        cnn_width_mult=cnn_width_mult,
        cnn_depth=cnn_depth,
        resnet_width_mult=resnet_width_mult,
        resnet_depth=resnet_depth,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    mean = float(ckpt.get("mean", 0.0))
    std = float(ckpt.get("std", 1.0))
    temperature = float(ckpt.get("val_temperature", 1.0))
    return model, id_to_label, mean, std, temperature, ckpt


@torch.no_grad()
def predict_audio(audio_path: Path, ckpt_path: Path, out_dir: Path, topk: int = 3, batch_size: int = 64, save_csv: bool = True) -> Dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, id_to_label, mean, std, temperature, ckpt = load_checkpoint(ckpt_path, device)

    y = load_and_preprocess_audio(audio_path)
    segments = cut_audio_segments(y)
    if not segments:
        raise RuntimeError("No valid segments generated.")

    all_logits: List[np.ndarray] = []
    all_attn: List[np.ndarray] = []
    rows: List[Dict[str, object]] = []

    batch_mels: List[np.ndarray] = []
    batch_meta: List[Tuple[int, float, float]] = []

    def flush_batch() -> None:
        nonlocal batch_mels, batch_meta, all_logits, all_attn, rows
        if not batch_mels:
            return
        arr = np.stack(batch_mels, axis=0).astype(np.float32)
        arr = (arr - mean) / max(std, 1e-6)
        arr = np.clip(arr, -6.0, 6.0)
        x = torch.from_numpy(arr).unsqueeze(1).to(device)
        logits, _, attn = model(x)
        logits_np = logits.detach().cpu().numpy()
        attn_np = attn.detach().cpu().numpy()
        probs_np = softmax_np(logits_np, temperature)
        all_logits.append(logits_np)
        all_attn.append(attn_np)
        for j, (seg_idx, start_sec, end_sec) in enumerate(batch_meta):
            p = probs_np[j]
            order = p.argsort()[::-1][:topk]
            row = {
                "segment_index": int(seg_idx),
                "start_sec": round(float(start_sec), 3),
                "end_sec": round(float(end_sec), 3),
                "pred_label": id_to_label[int(order[0])],
                "confidence": float(p[order[0]]),
                "attn_peak_index": int(np.argmax(attn_np[j])),
            }
            for rank, cls_id in enumerate(order, start=1):
                row[f"top{rank}_label"] = id_to_label[int(cls_id)]
                row[f"top{rank}_prob"] = float(p[cls_id])
            rows.append(row)
        batch_mels = []
        batch_meta = []

    for seg_idx, start_sec, end_sec, seg in segments:
        mel = segment_to_logmel(seg)
        batch_mels.append(mel)
        batch_meta.append((seg_idx, start_sec, end_sec))
        if len(batch_mels) >= batch_size:
            flush_batch()
    flush_batch()

    logits_all = np.concatenate(all_logits, axis=0)
    probs_all = softmax_np(logits_all, temperature)
    song_probs = probs_all.mean(axis=0)
    order = song_probs.argsort()[::-1][:topk]

    summary = {
        "audio_path": str(audio_path),
        "checkpoint": str(ckpt_path),
        "device": str(device),
        "sample_rate": SAMPLE_RATE,
        "duration_after_preprocess_sec": round(float(len(y) / SAMPLE_RATE), 3),
        "num_segments": int(len(segments)),
        "segment_sec": SEGMENT_SEC,
        "hop_sec": HOP_SEC,
        "temperature": float(temperature),
        "top_predictions": [
            {"rank": int(i + 1), "label": id_to_label[int(cls_id)], "prob": float(song_probs[cls_id])}
            for i, cls_id in enumerate(order)
        ],
    }

    if save_csv:
        ensure_dir(out_dir)
        stem = audio_path.stem
        csv_path = out_dir / f"{stem}_segment_predictions.csv"
        fields = ["segment_index", "start_sec", "end_sec", "pred_label", "confidence", "attn_peak_index"]
        for rank in range(1, topk + 1):
            fields.extend([f"top{rank}_label", f"top{rank}_prob"])
        write_csv(csv_path, rows, fields)
        save_json(out_dir / f"{stem}_summary.json", summary)
        summary["segment_csv"] = str(csv_path)
        summary["summary_json"] = str(out_dir / f"{stem}_summary.json")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="MusicFlowNet / EMF genre inference for one audio file.")
    parser.add_argument("--audio", type=str, required=True, help="Path to an audio file, e.g. wav/mp3/flac.")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT), help="Path to best_emf_v1.pt.")
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output directory for CSV/JSON.")
    parser.add_argument("--topk", type=int, default=3, help="Number of top predictions to show.")
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size over 3s segments.")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.out_dir)

    summary = predict_audio(
        audio_path=audio_path,
        ckpt_path=ckpt_path,
        out_dir=out_dir,
        topk=args.topk,
        batch_size=args.batch_size,
        save_csv=True,
    )

    print("\n===== MusicFlowNet Prediction =====")
    print(f"Audio     : {summary['audio_path']}")
    print(f"Segments  : {summary['num_segments']} x {summary['segment_sec']}s, hop={summary['hop_sec']}s")
    print(f"Duration  : {summary['duration_after_preprocess_sec']}s after preprocessing")
    print(f"Device    : {summary['device']}")
    print("Top predictions:")
    for item in summary["top_predictions"]:
        print(f"  Top-{item['rank']}: {item['label']:10s}  prob={item['prob']:.4f}")
    print(f"Segment CSV : {summary.get('segment_csv')}")
    print(f"Summary JSON: {summary.get('summary_json')}")


if __name__ == "__main__":
    main()
