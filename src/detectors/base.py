"""
변경점 탐지기 추상 베이스 클래스 및 결과 데이터 클래스
Wafer 기반: ref_data (n_ref, n_features) vs comp_data (n_comp, n_features)
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np


@dataclass
class DetectionResult:
    """단일 Feature에 대한 탐지 결과"""
    feature_index: int = -1
    is_detected: bool = False
    confidence: float = 0.0       # 0.0 ~ 1.0 (ROC curve용)
    method_name: str = ""
    extra: dict = field(default_factory=dict)


class BaseDetector(ABC):
    """변경점 탐지기 추상 베이스 클래스 (Wafer 기반)"""

    name: str = "BaseDetector"

    @abstractmethod
    def detect_feature(
        self,
        ref_values: np.ndarray,
        comp_values: np.ndarray,
    ) -> DetectionResult:
        """
        단일 Feature의 Ref vs Comp 비교.

        Args:
            ref_values: ref group wafer 값 (1D array, length=n_ref)
            comp_values: comp group wafer 값 (1D array, length=n_comp)
        Returns:
            DetectionResult
        """
        pass

    def detect_all(self, dataset) -> list:
        """
        데이터셋의 모든 Feature에 대해 탐지를 수행한다.
        다변량 방법(PCA, AE)은 이 메서드를 오버라이드한다.
        """
        results = []
        n_features = dataset.ref_data.shape[1]

        for j in range(n_features):
            ref_vals = dataset.ref_data[:, j]
            comp_vals = dataset.comp_data[:, j]

            result = self.detect_feature(ref_vals, comp_vals)
            result.feature_index = j
            result.method_name = self.name
            results.append(result)

        return results
