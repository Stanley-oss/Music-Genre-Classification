from abc import ABC, abstractmethod
import numpy as np
import sounddevice as sd
import librosa
import threading
import time
from typing import Callable, Optional


class AudioCapture(ABC):
    @abstractmethod
    def start(self, callback: Callable[[np.ndarray, int], None]):
        """
        callback(chunk: np.ndarray, sr: int)
        chunk 为单声道 float32 数组，sr 为其实际采样率。
        """

    @abstractmethod
    def stop(self):
        pass

    @property
    @abstractmethod
    def is_active(self) -> bool:
        pass


class MicrophoneCapture(AudioCapture):
    """基于 sounddevice 的麦克风实时捕获。自动适配设备原生采样率，避免内部失真。"""

    def __init__(
        self,
        target_sr: int = 22050,
        block_size: int = 1024,
        device: Optional[int] = None,
    ):
        self.target_sr = target_sr
        self.block_size = block_size
        self.device = device
        self._stream: Optional[sd.InputStream] = None
        self._is_active = False

    def start(self, callback: Callable[[np.ndarray, int], None]):
        self._is_active = True

        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"[Mic Status] {status}")
            chunk = indata[:, 0].copy()  # 取单声道
            # 使用设备原生采样率，由 RingBuffer 负责重采样
            sr = int(self._stream.samplerate)
            callback(chunk, sr)

        # 查询设备默认采样率（通常是 44100/48000），避免强制 22050 导致设备不兼容
        info = sd.query_devices(self.device, "input")
        native_sr = int(info["default_samplerate"])

        self._stream = sd.InputStream(
            samplerate=native_sr,
            blocksize=self.block_size,
            device=self.device,
            channels=1,
            dtype=np.float32,
            callback=audio_callback,
        )
        self._stream.start()

    def stop(self):
        self._is_active = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    @property
    def is_active(self) -> bool:
        return self._is_active


class FileCapture(AudioCapture):
    """文件流式播放：整块读取后按 chunk_duration 模拟实时流速推送。"""

    def __init__(
        self,
        file_path: str,
        target_sr: int = 22050,
        chunk_duration: float = 0.1,
    ):
        self.file_path = file_path
        self.target_sr = target_sr
        self.chunk_duration = chunk_duration
        self._thread: Optional[threading.Thread] = None
        self._is_active = False

    def start(self, callback: Callable[[np.ndarray, int], None]):
        self._is_active = True

        def run():
            try:
                # 一次性加载后切片（GTZAN 歌曲很短，此方案足够简洁）
                y, sr = librosa.load(self.file_path, sr=self.target_sr, mono=True)
                chunk_samples = int(self.target_sr * self.chunk_duration)

                for i in range(0, len(y), chunk_samples):
                    if not self._is_active:
                        break
                    chunk = y[i : i + chunk_samples]
                    callback(chunk, self.target_sr)
                    time.sleep(self.chunk_duration)
            except Exception as e:
                print(f"[FileCapture Error] {e}")
            finally:
                self._is_active = False

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self):
        self._is_active = False
        if self._thread:
            self._thread.join(timeout=1.0)

    @property
    def is_active(self) -> bool:
        return self._is_active