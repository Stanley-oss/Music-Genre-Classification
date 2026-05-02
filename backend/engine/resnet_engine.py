import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from typing import List

from .base import InferenceEngine

# 将 backend/ 加入路径以导入 infer_resnet
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .infer_resnet import AudioResNet, load_model, audio_to_mel_patch, GENRES


class ResNetEngine(InferenceEngine):
    def __init__(
        self,
        model_path: str = "./checkpoints_resnet18/best_resnet18_gtzan.pth",
        device: str = "auto",
        sr: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        frames: int = 96,
        dropout: float = 0.2,
    ):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.frames = frames
        self._patch_samples = n_fft + hop_length * (frames - 1)
        self._genres = GENRES

        # 设备自动选择
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = load_model(
            model_path=model_path,
            device=self.device,
            num_classes=len(GENRES),
            dropout=dropout,
        )
        self.model.eval()

    @property
    def genres(self) -> List[str]:
        return self._genres

    @property
    def sample_rate(self) -> int:
        return self.sr

    @property
    def patch_samples(self) -> int:
        return self._patch_samples

    def predict(self, audio_patch: np.ndarray) -> np.ndarray:
        # 长度对齐
        if len(audio_patch) < self.patch_samples:
            audio_patch = np.pad(
                audio_patch, (0, self.patch_samples - len(audio_patch)), mode="constant"
            )
        elif len(audio_patch) > self.patch_samples:
            audio_patch = audio_patch[: self.patch_samples]

        # 复用 infer_resnet 的 DSP 管线
        mel_db = audio_to_mel_patch(
            audio_patch,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            frames=self.frames,
        )

        # [1, 1, n_mels, frames] → GPU/CPU
        tensor = (
            torch.tensor(mel_db, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)

        return probs.cpu().numpy().flatten()