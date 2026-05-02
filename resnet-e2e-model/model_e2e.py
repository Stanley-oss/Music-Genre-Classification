import warnings
warnings.filterwarnings("ignore", message="stft with return_complex=False is deprecated")

import torch
import torch.nn as nn
import torchvision.models as models
import torchaudio


GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]


class WaveformToMel(nn.Module):
    """
    可导出的波形 -> Mel 频谱图层，所有操作均为基础 PyTorch ops，
    ONNX Opset 17 可完整覆盖（STFT + MatMul + ReduceMean 等）。
    
    输入:  [B, L]  单声道波形，L = samples_per_patch
    输出:  [B, 1, n_mels, frames]
    """
    def __init__(
        self,
        sr=22050,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        frames=96,
    ):
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.frames = frames
        
        # 与原先 dataset.py 保持一致
        self.samples_per_patch = n_fft + hop_length * (frames - 1)
        
        # Hann window（导出 ONNX 时作为常量 initializer 嵌入）
        window = torch.hann_window(n_fft, periodic=True)
        self.register_buffer("window", window)
        
        # Mel 滤波器组 —— 参数严格对齐 librosa 默认行为
        # librosa.feature.melspectrogram 默认: f_min=0, f_max=sr/2, htk=False, norm=None
        fbanks = torchaudio.functional.melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=0.0,
            f_max=float(sr / 2.0),
            n_mels=n_mels,
            sample_rate=sr,
            norm=None,            # 无 slaney 归一化，对齐 librosa 默认
            mel_scale="slaney",     # 对应 librosa htk=False
        )  # shape: [n_freqs, n_mels]
        self.register_buffer("mel_filterbank", fbanks)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, samples_per_patch]
        
        # 1. STFT → ONNX STFT (opset 17)
        #    return_complex=False 保证输出实数张量 [B, n_freqs, n_frames, 2]
        stft_out = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=False,               # 与原 dataset 对齐
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=False,       # 必须显式 False 才能触发 ONNX STFT 算子映射
        )
        
        # 2. Power spectrogram
        power = stft_out[..., 0] ** 2 + stft_out[..., 1] ** 2   # [B, n_freqs, n_frames]
        
        # 3. Mel filterbank via MatMul
        power_t = power.transpose(1, 2)                          # [B, n_frames, n_freqs]
        mel_power = torch.matmul(power_t, self.mel_filterbank)     # [B, n_frames, n_mels]
        mel_power = mel_power.transpose(1, 2)                      # [B, n_mels, n_frames]
        
        # 4. power_to_db(ref=np.max) —— 对齐 librosa
        ref_value = mel_power.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
        ref_value = ref_value.clamp(min=1e-10)
        mel_power_clipped = mel_power.clamp(min=1e-10)
        mel_db = 10.0 * torch.log10(mel_power_clipped / ref_value)
        
        # 5. Z-score normalization (per sample, 有偏 std 对齐 numpy)
        mean = mel_db.mean(dim=(1, 2), keepdim=True)
        std = mel_db.std(dim=(1, 2), keepdim=True, unbiased=False)
        mel_db = (mel_db - mean) / (std + 1e-6)
        
        # 6. Add channel dim: [B, 1, n_mels, n_frames]
        return mel_db.unsqueeze(1)


class AudioResNetE2E(nn.Module):
    """
    端到端音频分类模型:
      输入: [B, samples_per_patch] 波形
      输出: [B, num_classes] Logits（训练时用）
    """
    def __init__(
        self,
        num_classes=10,
        pretrained=False,
        dropout=0.0,
        sr=22050,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        frames=96,
    ):
        super().__init__()
        
        self.waveform_to_mel = WaveformToMel(
            sr=sr, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, frames=frames,
        )
        
        # ResNet-18 backbone
        if pretrained:
            try:
                weights = models.ResNet18_Weights.IMAGENET1K_V1
                self.resnet = models.resnet18(weights=weights)
            except Exception:
                self.resnet = models.resnet18(pretrained=True)
        else:
            try:
                self.resnet = models.resnet18(weights=None)
            except Exception:
                self.resnet = models.resnet18(pretrained=False)
        
        # 单通道 Mel 输入
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        num_ftrs = self.resnet.fc.in_features
        if dropout > 0:
            self.resnet.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(num_ftrs, num_classes),
            )
        else:
            self.resnet.fc = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mel = self.waveform_to_mel(x)   # [B, 1, n_mels, frames]
        logits = self.resnet(mel)       # [B, num_classes]
        return logits