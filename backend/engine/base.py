from abc import ABC, abstractmethod
from typing import List
import numpy as np


class InferenceEngine(ABC):
    """所有后端模型的统一接口。后续加入 CRNN / Transformer 时只需实现此类。"""

    @abstractmethod
    def predict(self, audio_patch: np.ndarray) -> np.ndarray:
        """
        输入: 单声道音频 patch [patch_samples], float32, sr=sample_rate
        输出: 概率分布 [num_classes]
        """
        pass

    @property
    @abstractmethod
    def genres(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        pass

    @property
    @abstractmethod
    def patch_samples(self) -> int:
        """一个 Patch 需要的采样点数（如 50688）"""
        pass