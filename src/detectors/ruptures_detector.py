"""
ruptures 라이브러리 기반 변경점 탐지기
Pelt, BinSeg, Window 3가지 방법을 래핑한다.
"""
import numpy as np
import ruptures as rpt
from .base import BaseDetector, DetectionResult


class RupturesPeltDetector(BaseDetector):
    """PELT (Pruned Exact Linear Time) 변경점 탐지기"""

    name = "PELT"

    def __init__(self, model: str = "rbf", pen: float = 3.0):
        self.model = model
        self.pen = pen

    def detect(self, ref_data, comp_data, full_series=None) -> DetectionResult:
        if full_series is None:
            full_series = np.concatenate([ref_data, comp_data])

        ref_len = len(ref_data)
        signal = full_series.reshape(-1, 1)

        try:
            algo = rpt.Pelt(model=self.model).fit(signal)
            bkps = algo.predict(pen=self.pen)
        except Exception:
            return DetectionResult(confidence=0.0, is_detected=False)

        # 마지막 breakpoint (= 시계열 길이)를 제외
        bkps = [b for b in bkps if b < len(full_series)]

        # comp period 내 change point가 있는지 확인
        comp_bkps = [b for b in bkps if b >= ref_len]
        is_detected = len(comp_bkps) > 0
        detected_cp = comp_bkps[0] if comp_bkps else -1

        # confidence: 검출된 change point 수 기반
        confidence = min(len(comp_bkps) / 3.0, 1.0) if comp_bkps else 0.0

        return DetectionResult(
            is_detected=is_detected,
            confidence=confidence,
            detected_cp_index=detected_cp,
            extra={"all_breakpoints": bkps, "comp_breakpoints": comp_bkps},
        )


class RupturesBinsegDetector(BaseDetector):
    """Binary Segmentation 변경점 탐지기"""

    name = "BinSeg"

    def __init__(self, model: str = "rbf", n_bkps: int = 2):
        self.model = model
        self.n_bkps = n_bkps

    def detect(self, ref_data, comp_data, full_series=None) -> DetectionResult:
        if full_series is None:
            full_series = np.concatenate([ref_data, comp_data])

        ref_len = len(ref_data)
        signal = full_series.reshape(-1, 1)

        try:
            algo = rpt.Binseg(model=self.model).fit(signal)
            bkps = algo.predict(n_bkps=self.n_bkps)
        except Exception:
            return DetectionResult(confidence=0.0, is_detected=False)

        bkps = [b for b in bkps if b < len(full_series)]
        comp_bkps = [b for b in bkps if b >= ref_len]
        is_detected = len(comp_bkps) > 0
        detected_cp = comp_bkps[0] if comp_bkps else -1

        confidence = min(len(comp_bkps) / 2.0, 1.0) if comp_bkps else 0.0

        return DetectionResult(
            is_detected=is_detected,
            confidence=confidence,
            detected_cp_index=detected_cp,
            extra={"all_breakpoints": bkps, "comp_breakpoints": comp_bkps},
        )


class RupturesWindowDetector(BaseDetector):
    """Window-based 변경점 탐지기"""

    name = "Window"

    def __init__(self, model: str = "rbf", width: int = 30, n_bkps: int = 2):
        self.model = model
        self.width = width
        self.n_bkps = n_bkps

    def detect(self, ref_data, comp_data, full_series=None) -> DetectionResult:
        if full_series is None:
            full_series = np.concatenate([ref_data, comp_data])

        ref_len = len(ref_data)
        signal = full_series.reshape(-1, 1)

        try:
            algo = rpt.Window(model=self.model, width=self.width).fit(signal)
            bkps = algo.predict(n_bkps=self.n_bkps)
        except Exception:
            return DetectionResult(confidence=0.0, is_detected=False)

        bkps = [b for b in bkps if b < len(full_series)]
        comp_bkps = [b for b in bkps if b >= ref_len]
        is_detected = len(comp_bkps) > 0
        detected_cp = comp_bkps[0] if comp_bkps else -1

        confidence = min(len(comp_bkps) / 2.0, 1.0) if comp_bkps else 0.0

        return DetectionResult(
            is_detected=is_detected,
            confidence=confidence,
            detected_cp_index=detected_cp,
            extra={"all_breakpoints": bkps, "comp_breakpoints": comp_bkps},
        )
