"""
CUSUM (Cumulative Sum) 기반 변경점 탐지기
Ref period의 평균/표준편차로 표준화 후 양방향 CUSUM을 적용한다.
"""
import numpy as np
from .base import BaseDetector, DetectionResult


class CUSUMDetector(BaseDetector):
    """Two-sided CUSUM 변경점 탐지기"""

    name = "CUSUM"

    def __init__(self, threshold: float = 5.0, drift: float = 0.5):
        self.threshold = threshold
        self.drift = drift

    def detect(self, ref_data, comp_data, full_series=None) -> DetectionResult:
        if full_series is None:
            full_series = np.concatenate([ref_data, comp_data])

        ref_mean = np.mean(ref_data)
        ref_std = np.std(ref_data)
        if ref_std < 1e-10:
            ref_std = 1e-10

        # 표준화
        z = (full_series - ref_mean) / ref_std

        # Two-sided CUSUM
        n = len(z)
        s_pos = np.zeros(n)
        s_neg = np.zeros(n)

        for i in range(1, n):
            s_pos[i] = max(0, s_pos[i - 1] + z[i] - self.drift)
            s_neg[i] = max(0, s_neg[i - 1] - z[i] - self.drift)

        # Comp period 내에서 threshold 초과하는 첫 번째 포인트 탐색
        ref_len = len(ref_data)
        detected_cp = -1
        max_cusum = 0.0

        for i in range(ref_len, n):
            cusum_val = max(s_pos[i], s_neg[i])
            if cusum_val > max_cusum:
                max_cusum = cusum_val
            if cusum_val > self.threshold and detected_cp == -1:
                detected_cp = i

        is_detected = detected_cp >= 0
        # confidence: threshold 대비 최대 CUSUM 비율 (1.0으로 클리핑)
        confidence = min(max_cusum / (self.threshold * 2), 1.0) if self.threshold > 0 else 0.0

        return DetectionResult(
            is_detected=is_detected,
            confidence=confidence,
            detected_cp_index=detected_cp,
            extra={
                "max_cusum_pos": float(np.max(s_pos[ref_len:])) if ref_len < n else 0.0,
                "max_cusum_neg": float(np.max(s_neg[ref_len:])) if ref_len < n else 0.0,
                "threshold": self.threshold,
            },
        )
