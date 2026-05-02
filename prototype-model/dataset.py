import os
import argparse
import random

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class GTZANSturmDataset(Dataset):
    """
    GTZAN Dataset with Sturm split.

    输出:
        tensor_img: Tensor, shape = [1, n_mels, frames]
        label_id:   LongTensor, genre label id
    """

    def __init__(
        self,
        data_dir,
        split_txt_path,  # 修复 1：补齐缺失的逗号
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

        # 为了保证 librosa.feature.melspectrogram(..., center=False)
        # 输出正好是 frames 帧，需要音频长度满足:
        # frames = 1 + floor((len(y) - n_fft) / hop_length)
        # 所以:
        # len(y) = n_fft + hop_length * (frames - 1)
        self.samples_per_patch = self.n_fft + self.hop_length * (self.frames - 1)

        self.genres = [
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

        # 修复 2：补齐缺失的 self
        self.genre_to_id = {genre: idx for idx, genre in enumerate(self.genres)}

        self.file_list = []

        self._load_split_file(check_exists=check_exists)

        if len(self.file_list) == 0:
            raise RuntimeError(
                f"No valid audio files found. Please check:\n"
                f"data_dir = {self.data_dir}\n"
                f"split_txt_path = {self.split_txt_path}"
            )

    def _load_split_file(self, check_exists=True):
        """
        读取 Sturm split 文件。

        split txt 中通常每一行类似:
            blues.00000.wav
            rock.00015.wav

        GTZAN 的真实路径通常是:
            data_dir/blues/blues.00000.wav
            data_dir/rock/rock.00015.wav
        """

        if not os.path.exists(self.split_txt_path):
            raise FileNotFoundError(f"Split file not found: {self.split_txt_path}")

        with open(self.split_txt_path, "r", encoding="utf-8") as f:
            for line in f:
                filename = line.strip()

                if not filename:
                    continue

                # 防止 txt 中出现额外空格
                filename = filename.split()[0]

                # GTZAN 文件名格式: rock.00001.wav
                genre = filename.split(".")[0]

                if genre not in self.genre_to_id:
                    raise ValueError(
                        f"Unknown genre '{genre}' in file name '{filename}'. "
                        f"Expected one of {self.genres}"
                    )

                file_path = os.path.join(self.data_dir, genre, filename)

                if check_exists and not os.path.exists(file_path):
                    raise FileNotFoundError(f"Audio file not found: {file_path}")

                self.file_list.append((file_path, genre))

    def __len__(self):
        return len(self.file_list)

    def _load_audio(self, file_path):
        """
        加载音频 (带损坏文件容错处理)。
        """
        try:
            y, _ = librosa.load(file_path, sr=self.sr, mono=True)
            return y
        except Exception as e:
            print(f"\n[WARNING] 发现损坏音频，已自动跳过并补零: {file_path}")
            print(f"[ERROR INFO] {e}")
            # 如果音频损坏，返回一个完全静音的空白数组，长度等于你的 patch_size
            # 这样网络最多只会学到一个毫无特征的黑图，绝对不会导致程序崩溃
            return np.zeros(self.samples_per_patch, dtype=np.float32)

    def _crop_or_pad(self, y):
        """
        训练时随机裁剪；
        验证 / 测试时中心裁剪；
        如果音频长度不足，则末尾补零。
        """

        # 修复 3：补齐缺失的 .samples
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

    def _extract_mel_spectrogram(self, y_patch):
        """
        提取 Mel Spectrogram，并保证输出 shape 为[n_mels, frames]。
        """

        mel = librosa.feature.melspectrogram(
            y=y_patch,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            center=False,
            power=2.0,
        )

        mel_db = librosa.power_to_db(mel, ref=np.max)

        # 数值稳定处理
        mean = mel_db.mean()
        std = mel_db.std()

        mel_db = (mel_db - mean) / (std + 1e-6)

        # 理论上 center=False 且 samples_per_patch 设置正确时，
        # mel_db.shape[1] 应该正好等于 self.frames。
        # 这里再做一次保险处理。
        if mel_db.shape[1] < self.frames:
            pad_width = self.frames - mel_db.shape[1]
            mel_db = np.pad(mel_db, pad_width=((0, 0), (0, pad_width)), mode="constant")
        elif mel_db.shape[1] > self.frames:
            mel_db = mel_db[:, : self.frames]

        return mel_db

    def __getitem__(self, idx):
        file_path, genre = self.file_list[idx]

        label_id = self.genre_to_id[genre]

        y = self._load_audio(file_path)
        y_patch = self._crop_or_pad(y)
        mel_db = self._extract_mel_spectrogram(y_patch)

        # [n_mels, frames] ->[1, n_mels, frames]
        tensor_img = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)

        label = torch.tensor(label_id, dtype=torch.long)

        return tensor_img, label


# 修复 4：补齐缺失的 (seed
def set_seed(seed=42):
    """
    固定随机种子，保证实验可复现。
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloader(
    data_dir,
    split_txt_path,
    batch_size=16,
    num_workers=4,
    is_train=True,
    sr=22050,
    n_fft=2048,
    hop_length=512,
    n_mels=128,
    frames=96,
):
    """
    构建 DataLoader。
    """

    dataset = GTZANSturmDataset(
        data_dir=data_dir,
        split_txt_path=split_txt_path,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        frames=frames,
        is_train=is_train,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return dataset, dataloader


def main():
    parser = argparse.ArgumentParser(
        description="GTZAN Sturm Dataset preprocessing test script"
    )

    # 修复 5：补齐缺失的左括号 (
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="GTZAN genres root directory, e.g. /path/to/GTZAN/genres_original",
    )

    parser.add_argument(
        "--split_txt",
        type=str,
        required=True,
        help="Sturm split txt file path, e.g. /path/to/train_filtered.txt",
    )

    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "val", "test"]
    )

    parser.add_argument("--sr", type=int, default=22050)

    parser.add_argument("--n_fft", type=int, default=2048)

    parser.add_argument("--hop_length", type=int, default=512)

    parser.add_argument("--n_mels", type=int, default=128)

    parser.add_argument(
        "--frames",
        type=int,
        default=96,  # 修复 6：补齐缺失的 default关键字
    )

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    is_train = args.mode == "train"

    dataset, dataloader = build_dataloader(
        data_dir=args.data_dir,
        split_txt_path=args.split_txt,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_train=is_train,
        sr=args.sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        frames=args.frames,
    )

    print("Dataset loaded successfully.")
    print(f"Number of samples: {len(dataset)}")
    print(f"Mode: {args.mode}")
    print(f"Sample rate: {args.sr}")
    print(f"n_fft: {args.n_fft}")
    print(f"hop_length: {args.hop_length}")
    print(f"n_mels: {args.n_mels}")
    print(f"frames: {args.frames}")
    print(f"samples_per_patch: {dataset.samples_per_patch}")

    for batch_idx, (x, y) in enumerate(dataloader):
        print("=" * 60)
        print(f"Batch index: {batch_idx}")
        print(f"Input shape: {x.shape}")
        print(f"Label shape: {y.shape}")
        print(f"Labels: {y}")

        # 只测试一个 batch
        break


if __name__ == "__main__":
    main()
