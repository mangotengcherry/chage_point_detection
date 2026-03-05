"""
변경점 탐지기 추상 베이스 클래스 및 결과 데이터 클래스
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


@dataclass
class DetectionResult:
    """단일 BIN item에 대한 탐지 결과"""
    bin_index: int = -1
    is_detected: bool = False
    confidence: float = 0.0       # 0.0 ~ 1.0 (ROC curve용)
    detected_cp_index: int = -1   # 추정된 change point 위치 (-1: 미탐지)
    method_name: str = ""
    extra: dict = field(default_factory=dict)


class BaseDetector(ABC):
    """변경점 탐지기 추상 베이스 클래스"""

    name: str = "BaseDetector"

    @abstractmethod
    def detect(
        self,
        ref_data: np.ndarray,
        comp_data: np.ndarray,
        full_series: np.ndarray = None,
    ) -> DetectionResult:
        """
        단일 BIN item의 변경점을 탐지한다.

        Args:
            ref_data: ref period 데이터 (1D array)
            comp_data: comp period 데이터 (1D array)
            full_series: 전체 시계열 (optional, CUSUM/ruptures 등에서 사용)
        Returns:
            DetectionResult
        """
        pass

    def detect_all(self, dataset) -> list:
        """
        데이터셋의 모든 BIN item에 대해 탐지를 수행한다.
        다변량 방법(PCA, AE)은 이 메서드를 오버라이드한다.
        """
        from src.data_generation import SyntheticBINDataset

        results = []
        n_bins = dataset.data.shape[1]

        for i in range(n_bins):
            ref = dataset.data[:dataset.ref_end_index, i]
            comp = dataset.data[dataset.ref_end_index:, i]
            full = dataset.data[:, i]

            result = self.detect(ref, comp, full)
            result.bin_index = i
            result.method_name = self.name
            results.append(result)

        return results
