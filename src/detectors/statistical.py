"""
통계검정 기반 변경점 탐지기
Mann-Whitney U, KS Test, T-test, Welch's t-test
"""
import numpy as np
from scipy import stats
from .base import BaseDetector, DetectionResult


class MannWhitneyDetector(BaseDetector):
    """Mann-Whitney U 검정 (비모수, rank-based)"""

    name = "Mann-Whitney U"

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def detect(self, ref_data, comp_data, full_series=None) -> DetectionResult:
        # 상수 배열 체크
        if np.std(ref_data) == 0 and np.std(comp_data) == 0:
            return DetectionResult(confidence=0.0, is_detected=False)

        try:
            stat, p_value = stats.mannwhitneyu(
                ref_data, comp_data, alternative="two-sided"
            )
        except ValueError:
            return DetectionResult(confidence=0.0, is_detected=False)

        confidence = 1.0 - p_value
        return DetectionResult(
            is_detected=(p_value < self.alpha),
            confidence=min(max(confidence, 0.0), 1.0),
            extra={"statistic": float(stat), "p_value": float(p_value)},
        )


class KSTestDetector(BaseDetector):
    """Kolmogorov-Smirnov 2-sample 검정 (분포 형태 비교)"""

    name = "KS Test"

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def detect(self, ref_data, comp_data, full_series=None) -> DetectionResult:
        try:
            stat, p_value = stats.ks_2samp(ref_data, comp_data)
        except Exception:
            return DetectionResult(confidence=0.0, is_detected=False)

        confidence = 1.0 - p_value
        return DetectionResult(
            is_detected=(p_value < self.alpha),
            confidence=min(max(confidence, 0.0), 1.0),
            extra={"statistic": float(stat), "p_value": float(p_value)},
        )


class TTestDetector(BaseDetector):
    """독립 2-표본 t 검정 (모수적, equal variance)"""

    name = "T-test"

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def detect(self, ref_data, comp_data, full_series=None) -> DetectionResult:
        if np.std(ref_data) == 0 and np.std(comp_data) == 0:
            return DetectionResult(confidence=0.0, is_detected=False)

        try:
            stat, p_value = stats.ttest_ind(ref_data, comp_data, equal_var=True)
        except Exception:
            return DetectionResult(confidence=0.0, is_detected=False)

        confidence = 1.0 - p_value
        return DetectionResult(
            is_detected=(p_value < self.alpha),
            confidence=min(max(confidence, 0.0), 1.0),
            extra={"statistic": float(stat), "p_value": float(p_value)},
        )


class WelchTTestDetector(BaseDetector):
    """Welch's t 검정 (모수적, unequal variance)"""

    name = "Welch's t-test"

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def detect(self, ref_data, comp_data, full_series=None) -> DetectionResult:
        if np.std(ref_data) == 0 and np.std(comp_data) == 0:
            return DetectionResult(confidence=0.0, is_detected=False)

        try:
            stat, p_value = stats.ttest_ind(ref_data, comp_data, equal_var=False)
        except Exception:
            return DetectionResult(confidence=0.0, is_detected=False)

        confidence = 1.0 - p_value
        return DetectionResult(
            is_detected=(p_value < self.alpha),
            confidence=min(max(confidence, 0.0), 1.0),
            extra={"statistic": float(stat), "p_value": float(p_value)},
        )
