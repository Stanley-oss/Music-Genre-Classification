import numpy as np
import librosa
import threading
from typing import Optional


class AudioRingBuffer:
    """
    线程安全环形缓冲区。
    无论输入是什么采样率，都会实时重采样到 target_sr。
    """

    def __init__(self, target_sr: int = 22050, target_samples: Optional[int] = None):
        self.target_sr = target_sr
        self.target_samples = target_samples
        self._buffer = np.zeros(self.target_samples, dtype=np.float32)
        self._lock = threading.Lock()

    def append(self, chunk: np.ndarray, chunk_sr: int):
        """线程安全写入。chunk 可为任意采样率、任意长度。"""
        if chunk.size == 0:
            return

        # 重采样到目标采样率
        if chunk_sr != self.target_sr:
            chunk = librosa.resample(
                chunk.astype(np.float32),
                orig_sr=chunk_sr,
                target_sr=self.target_sr,
            )

        # 强制单声道
        if chunk.ndim > 1:
            chunk = chunk.mean(axis=0)

        with self._lock:
            concatenated = np.concatenate([self._buffer, chunk])
            if len(concatenated) > self.target_samples:
                self._buffer = concatenated[-self.target_samples :]
            else:
                # 初始阶段前补零，保持长度恒定，方便模型推理
                self._buffer = np.pad(
                    concatenated,
                    (self.target_samples - len(concatenated), 0),
                    mode="constant",
                )

    def get_snapshot(self) -> np.ndarray:
        with self._lock:
            return self._buffer.copy()

    def clear(self):
        with self._lock:
            self._buffer.fill(0)