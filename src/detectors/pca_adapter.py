"""
기존 PCAHotellingT2를 벤치마크 인터페이스에 맞게 래핑하는 어댑터
다변량 방법: detect_all()을 오버라이드하여 전체 Feature 행렬을 한번에 처리한다.
Wafer 기반: ref_data (n_ref, n_features) vs comp_data (n_comp, n_features)
"""
import numpy as np
from .base import BaseDetector, DetectionResult

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.pca_hotelling import PCAHotellingT2


class PCAHotellingAdapter(BaseDetector):
    """PCA + Hotelling T² 다변량 변경점 탐지기 (기존 코드 래핑)"""

    name = "PCA+Hotelling T²"

    def __init__(self, n_components: float = 0.95, alpha: float = 0.01,
                 contribution_threshold: float = 2.0):
        self.n_components = n_components
        self.alpha = alpha
        self.contribution_threshold = contribution_threshold

    def detect_feature(self, ref_values, comp_values) -> DetectionResult:
        return DetectionResult(confidence=0.0, is_detected=False)

    def detect_all(self, dataset) -> list:
        """다변량: 전체 Feature 행렬을 PCA+T²로 분석"""
        ref_matrix = dataset.ref_data    # (n_ref, n_features)
        comp_matrix = dataset.comp_data  # (n_comp, n_features)
        n_features = ref_matrix.shape[1]

        try:
            model = PCAHotellingT2(
                n_components=self.n_components,
                alpha=self.alpha,
            )
            model.fit(ref_matrix)
            result = model.analyze(comp_matrix)

            mean_contributions = np.mean(
                np.abs(result.feature_contributions), axis=0
            )

            # IQR 기반 robust threshold
            q75 = np.percentile(mean_contributions, 75)
            q25 = np.percentile(mean_contributions, 25)
            iqr = q75 - q25
            threshold = q75 + 1.5 * iqr

            results = []
            for i in range(n_features):
                contrib = mean_contributions[i]
                is_detected = contrib > threshold
                confidence = min(contrib / (threshold * 2), 1.0) if threshold > 0 else 0.0

                results.append(DetectionResult(
                    feature_index=i,
                    is_detected=is_detected,
                    confidence=confidence,
                    method_name=self.name,
                    extra={
                        "contribution": float(contrib),
                        "threshold": float(threshold),
                    },
                ))
            return results

        except Exception as e:
            return [
                DetectionResult(feature_index=i, method_name=self.name,
                                extra={"error": str(e)})
                for i in range(n_features)
            ]
