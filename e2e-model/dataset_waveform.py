import os
import argparse
import random

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class GTZANWaveformDataset(Dataset):
    """
    GTZAN Dataset —— 返回原始波形 patch。
    Mel 提取已移至模型内部，保证训练/推理/前端三端完全一致。
    """
    def __init__(
        self,
        data_dir,
        split_txt_path,
        sr=22050,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        frames=96,
        is_train=True,
        check_exists=True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.split_txt_path = split_txt_path
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.frames = frames
        self.is_train = is_train
        
        self.samples_per_patch = self.n_fft + self.hop_length * (self.frames - 1)
        
        self.genres = [
            "blues", "classical", "country", "disco", "hiphop",
            "jazz", "metal", "pop", "reggae", "rock",
        ]
        self.genre_to_id = {genre: idx for idx, genre in enumerate(self.genres)}
        self.file_list = []
        self._load_split_file(check_exists=check_exists)
        
        if len(self.file_list) == 0:
            raise RuntimeError(
                f"No valid audio files found. data_dir={data_dir}, split={split_txt_path}"
            )

    def _load_split_file(self, check_exists=True):
        if not os.path.exists(self.split_txt_path):
            raise FileNotFoundError(f"Split file not found: {self.split_txt_path}")
        with open(self.split_txt_path, "r", encoding="utf-8") as f:
            for line in f:
                filename = line.strip()
                if not filename:
                    continue
                filename = filename.split()[0]
                genre = filename.split(".")[0]
                if genre not in self.genre_to_id:
                    raise ValueError(f"Unknown genre '{genre}'")
                file_path = os.path.join(self.data_dir, genre, filename)
                if check_exists and not os.path.exists(file_path):
                    raise FileNotFoundError(f"Audio file not found: {file_path}")
                self.file_list.append((file_path, genre))

    def __len__(self):
        return len(self.file_list)

    def _load_audio(self, file_path):
        try:
            y, _ = librosa.load(file_path, sr=self.sr, mono=True)
            return y
        except Exception as e:
            print(f"\n[WARNING] Corrupted audio, zero-padding: {file_path}")
            return np.zeros(self.samples_per_patch, dtype=np.float32)

    def _crop_or_pad(self, y):
        target_len = self.samples_per_patch
        if len(y) > target_len:
            if self.is_train:
                start = np.random.randint(0, len(y) - target_len + 1)
            else:
                start = (len(y) - target_len) // 2
            y_patch = y[start : start + target_len]
        else:
            pad_len = target_len - len(y)
            y_patch = np.pad(y, (0, pad_len), mode="constant")
        return y_patch

    def __getitem__(self, idx):
        file_path, genre = self.file_list[idx]
        label_id = self.genre_to_id[genre]
        y = self._load_audio(file_path)
        y_patch = self._crop_or_pad(y)
        waveform = torch.tensor(y_patch, dtype=torch.float32)
        label = torch.tensor(label_id, dtype=torch.long)
        return waveform, label


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloader(
    data_dir, split_txt_path, batch_size=16, num_workers=4,
    is_train=True, sr=22050, n_fft=2048, hop_length=512,
    n_mels=128, frames=96,
):
    dataset = GTZANWaveformDataset(
        data_dir=data_dir, split_txt_path=split_txt_path,
        sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, frames=frames, is_train=is_train,
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=is_train,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    return dataset, dataloader