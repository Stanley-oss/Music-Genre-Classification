import os
import argparse

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]


class AudioResNet(nn.Module):
    """
    ResNet-18 for GTZAN music genre classification.

    输入:[B, 1, 128, 96]

    输出:
        logits: [B, 10]

    注意:
        推理时再使用 softmax。
        模型内部不使用 sigmoid / softmax。
    """

    def __init__(self, num_classes=10, dropout=0.0):
        super().__init__()

        try:
            self.resnet = models.resnet18(weights=None)
        except Exception:
            self.resnet = models.resnet18(pretrained=False)

        self.resnet.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        num_ftrs = self.resnet.fc.in_features

        if dropout > 0:
            self.resnet.fc = nn.Sequential(
                nn.Dropout(p=dropout), nn.Linear(num_ftrs, num_classes)
            )
        else:
            self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)


def get_device(device_name=None):
    """
    自动选择设备。
    """

    if device_name is not None:
        return torch.device(device_name)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_model(model_path, device, num_classes=10, dropout=0.0):
    """
    加载训练好的 AudioResNet 模型。

    兼容:
        1. torch.save(model.state_dict(), path)
        2. torch.save({"model_state_dict": model.state_dict(), ...}, path)
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = AudioResNet(num_classes=num_classes, dropout=dropout).to(device)

    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    return model


def audio_to_mel_patch(
    y_patch, sr=22050, n_fft=2048, hop_length=512, n_mels=128, frames=96
):
    """
    将一个音频切片转换为 Mel dB 谱。

    输出:
        mel_db: [n_mels, frames]

    必须和训练阶段 Dataset 保持一致：
        center=False
        power=2.0
        Z-score normalization
    """

    mel = librosa.feature.melspectrogram(
        y=y_patch,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        center=False,
        power=2.0,
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

    # 保险处理，保证帧数为 frames
    if mel_db.shape[1] < frames:
        pad_width = frames - mel_db.shape[1]
        mel_db = np.pad(
            mel_db,
            pad_width=((0, 0), (0, pad_width)),
            mode="constant",  # 修复 1：调整缩进错误
        )
    elif mel_db.shape[1] > frames:
        mel_db = mel_db[:, :frames]

    return mel_db


def split_audio_to_patches(
    y,
    sr=22050,
    n_fft=2048,
    hop_length=512,
    n_mels=128,
    frames=96,
    overlap=0.0,
    include_last=True,
):
    """
    将整首音频切分成多个固定长度片段，并转换为 Mel 频谱。

    参数:
        overlap:
            片段重叠率。
            0.0 表示不重叠。
            0.5 表示 50% 重叠。

        include_last:
            是否包含最后不足一个 patch 的音频。
            如果 True，会对最后一段进行 zero padding。
            如果 False，会丢弃最后不足 patch 的部分。

    返回:
        patches: list of np.ndarray，每个 shape 为[n_mels, frames]
    """

    if overlap < 0 or overlap >= 1:
        raise ValueError("overlap must be in[0, 1).")

    # 注意：
    # 这里必须和训练阶段 Dataset 的 samples_per_patch持一致。
    # 如果训练时用了:
    # samples_per_patch = n_fft + hop_length * (frames - 1)
    # center=False
    # 那么推理也必须这样。
    samples_per_patch = n_fft + hop_length * (frames - 1)

    step = int(samples_per_patch * (1.0 - overlap))
    step = max(step, 1)

    patches = []

    audio_len = len(y)

    if audio_len == 0:
        return patches

    if audio_len < samples_per_patch:
        if include_last:
            y_patch = np.pad(
                y, pad_width=(0, samples_per_patch - audio_len), mode="constant"
            )

            mel_db = audio_to_mel_patch(
                y_patch,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                frames=frames,
            )

            patches.append(mel_db)

        return patches

    start = 0

    while start + samples_per_patch <= audio_len:
        y_patch = y[start : start + samples_per_patch]

        mel_db = audio_to_mel_patch(
            y_patch,
            sr=sr,  # 修复 2：原代码此处打字错误为 srsr
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            frames=frames,
        )

        patches.append(mel_db)

        start += step

    # 处理最后不足一个 patch 的尾部
    if include_last and start < audio_len:
        y_tail = y[start:]

        if len(y_tail) > 0:
            y_patch = np.pad(
                y_tail, pad_width=(0, samples_per_patch - len(y_tail)), mode="constant"
            )

            mel_db = audio_to_mel_patch(
                y_patch,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                frames=frames,
            )

            patches.append(mel_db)

    return patches


@torch.no_grad()
def predict_full_song(
    audio_path,
    model_path,
    sr=22050,
    n_fft=2048,
    hop_length=512,
    n_mels=128,
    frames=96,
    num_classes=10,
    dropout=0.0,
    batch_size=64,
    overlap=0.0,
    include_last=True,
    device_name=None,  # 修复 3：原参数名为 device，与下方变量名和 main 中的传参冲突
):
    """
    对整首歌曲进行流派预测。

    返回:
        predicted_genre: str
        confidence: float
        mean_probs: np.ndarray, shape = [10]
        patch_probs: np.ndarray, shape = [num_patches, 10]
    """

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    device = get_device(device_name)
    print(f"[INFO] Using device: {device}")

    model = load_model(
        model_path=model_path, device=device, num_classes=num_classes, dropout=dropout
    )

    print(f"[INFO] Loading audio: {audio_path}")

    y, _ = librosa.load(audio_path, sr=sr, mono=True)

    patches = split_audio_to_patches(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        frames=frames,
        overlap=overlap,
        include_last=include_last,
    )

    if len(patches) == 0:
        raise RuntimeError("音频过短或无法分割，推理失败。")  # 修复 4：补全丢失的字"割"

    print(f"[INFO] Number of patches: {len(patches)}")

    patch_array = np.array(patches, dtype=np.float32)

    # [N, 128, 96] -> [N, 1, 128, 96]
    tensor_batch = torch.from_numpy(patch_array).unsqueeze(1)

    all_probs = []

    for start in range(0, tensor_batch.size(0), batch_size):
        batch = tensor_batch[start : start + batch_size].to(device)

        logits = model(batch)
        probs = F.softmax(logits, dim=1)

        all_probs.append(probs.cpu())

    patch_probs = torch.cat(all_probs, dim=0)

    # 对所有片段概率求平均，得到整首歌的概率
    mean_probs = patch_probs.mean(dim=0)

    predicted_idx = torch.argmax(mean_probs).item()
    confidence = mean_probs[predicted_idx].item()
    predicted_genre = GENRES[predicted_idx]

    return (predicted_genre, confidence, mean_probs.numpy(), patch_probs.numpy())


def print_prediction_result(
    predicted_genre,
    confidence,
    mean_probs,
    top_k=5,  # 修复 5：恢复缩进
):
    """
    打印预测结果。
    """

    print("\n========== Prediction Result ==========")
    print(f"Final predicted genre: {predicted_genre.upper()}")
    print(f"Confidence: {confidence * 100:.2f}%")

    print(f"\nTop-{top_k} probabilities:")

    sorted_indices = np.argsort(mean_probs)[::-1]

    for rank, idx in enumerate(sorted_indices[:top_k], start=1):
        genre = GENRES[idx]
        prob = mean_probs[idx]
        print(f"{rank:02d}. {genre:10s}: {prob * 100:.2f}%")

    print("\nAll probabilities:")

    for genre, prob in zip(GENRES, mean_probs):
        print(f"{genre:10s}: {prob * 100:.2f}%")

    print("=======================================\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference script for GTZAN ResNet-18 genre classifier"
    )

    parser.add_argument(
        "--audio_path",
        type=str,
        required=True,
        help="Path to input audio file, e.g. test_music.wav",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="./checkpoints_resnet18/best_resnet18_gtzan.pth",
        help="Path to trained model .pth file",
    )

    # 音频特征参数：必须和训练时一致
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--frames", type=int, default=96)

    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for patch inference"
    )

    parser.add_argument(
        "--overlap",
        type=float,
        default=0.0,
        help="Overlap ratio between patches. 0.0 means no overlap, 0.5 means 50 percent overlap.",
    )

    parser.add_argument(
        "--no_include_last",
        action="store_true",
        help="If set, drop the last incomplete patch instead of padding it.",
    )

    parser.add_argument(
        "--dropout",
        type=float,  # 修复 6：原代码丢失了 =float
        default=0.0,
        help="Dropout value used in model definition. Must match training model.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device, e.g. cuda, cpu, mps. Default: auto.",
    )

    parser.add_argument("--top_k", type=int, default=5)

    return parser.parse_args()


def main():
    args = parse_args()

    predicted_genre, confidence, mean_probs, patch_probs = predict_full_song(
        audio_path=args.audio_path,
        model_path=args.model_path,
        sr=args.sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        frames=args.frames,
        batch_size=args.batch_size,
        overlap=args.overlap,
        include_last=not args.no_include_last,
        dropout=args.dropout,
        device_name=args.device,
    )

    print_prediction_result(
        predicted_genre=predicted_genre,
        confidence=confidence,
        mean_probs=mean_probs,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
