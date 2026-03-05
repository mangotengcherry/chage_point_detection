"""
기존 PCAHotellingT2를 벤치마크 인터페이스에 맞게 래핑하는 어댑터
다변량 방법: detect_all()을 오버라이드하여 전체 BIN 행렬을 한번에 처리한다.
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

    def detect(self, ref_data, comp_data, full_series=None) -> DetectionResult:
        # 단변량에서는 사용하지 않음 (detect_all에서 다변량으로 처리)
        return DetectionResult(confidence=0.0, is_detected=False)

    def detect_all(self, dataset) -> list:
        """다변량: 전체 BIN 행렬을 PCA+T²로 분석"""
        ref_matrix = dataset.data[:dataset.ref_end_index, :]  # (ref_len, n_bins)
        comp_matrix = dataset.data[dataset.ref_end_index:, :]  # (comp_len, n_bins)

        n_bins = dataset.data.shape[1]

        try:
            model = PCAHotellingT2(
                n_components=self.n_components,
                alpha=self.alpha,
            )
            model.fit(ref_matrix)
            result = model.analyze(comp_matrix)

            # feature_contributions: (n_comp_samples, n_bins)의 평균 기여도
            mean_contributions = np.mean(
                np.abs(result.feature_contributions), axis=0
            )

            # 기여도 기반 탐지: percentile 기반 threshold
            # contribution_threshold는 상위 N%를 탐지하는 percentile 값
            threshold = np.percentile(mean_contributions,
                                       100 - self.contribution_threshold * 100 / n_bins * 100)
            # fallback: mean + 1.5*IQR (robust 방법)
            q75 = np.percentile(mean_contributions, 75)
            q25 = np.percentile(mean_contributions, 25)
            iqr = q75 - q25
            threshold_iqr = q75 + 1.5 * iqr
            threshold = min(threshold, threshold_iqr)

            results = []
            for i in range(n_bins):
                contrib = mean_contributions[i]
                is_detected = contrib > threshold
                # confidence: 기여도를 threshold 대비 정규화
                confidence = min(contrib / (threshold * 2), 1.0) if threshold > 0 else 0.0

                results.append(DetectionResult(
                    bin_index=i,
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
            # PCA 실패 시 모든 BIN에 대해 미탐지 반환
            return [
                DetectionResult(bin_index=i, method_name=self.name,
                                extra={"error": str(e)})
                for i in range(n_bins)
            ]
