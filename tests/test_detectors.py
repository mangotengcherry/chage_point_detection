"""탐지기 테스트"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from src.detectors.statistical import (
    MannWhitneyDetector, KSTestDetector, TTestDetector, WelchTTestDetector
)
from src.detectors.cusum import CUSUMDetector
from src.detectors.base import DetectionResult


@pytest.fixture
def normal_data():
    """정상 데이터 (ref == comp 분포)"""
    rng = np.random.RandomState(42)
    ref = rng.normal(0, 1, 150)
    comp = rng.normal(0, 1, 150)
    return ref, comp


@pytest.fixture
def shifted_data():
    """변경된 데이터 (comp에 mean shift)"""
    rng = np.random.RandomState(42)
    ref = rng.normal(0, 1, 150)
    comp = rng.normal(3, 1, 150)  # 3-sigma shift
    return ref, comp


class TestStatisticalDetectors:
    @pytest.mark.parametrize("DetectorClass", [
        MannWhitneyDetector, KSTestDetector, TTestDetector, WelchTTestDetector
    ])
    def test_result_type(self, DetectorClass, normal_data):
        """결과가 DetectionResult 타입인지"""
        detector = DetectorClass()
        ref, comp = normal_data
        result = detector.detect(ref, comp)
        assert isinstance(result, DetectionResult)

    @pytest.mark.parametrize("DetectorClass", [
        MannWhitneyDetector, KSTestDetector, TTestDetector, WelchTTestDetector
    ])
    def test_detect_clear_shift(self, DetectorClass, shifted_data):
        """명확한 mean shift를 탐지하는지"""
        detector = DetectorClass(alpha=0.05)
        ref, comp = shifted_data
        result = detector.detect(ref, comp)
        assert result.is_detected, f"{DetectorClass.__name__}가 3-sigma shift를 놓침"

    @pytest.mark.parametrize("DetectorClass", [
        MannWhitneyDetector, KSTestDetector, TTestDetector, WelchTTestDetector
    ])
    def test_confidence_range(self, DetectorClass, shifted_data):
        """confidence가 0~1 범위인지"""
        detector = DetectorClass()
        ref, comp = shifted_data
        result = detector.detect(ref, comp)
        assert 0.0 <= result.confidence <= 1.0


class TestCUSUM:
    def test_detect_shift(self, shifted_data):
        ref, comp = shifted_data
        full = np.concatenate([ref, comp])
        detector = CUSUMDetector(threshold=5.0, drift=0.5)
        result = detector.detect(ref, comp, full)
        assert result.is_detected

    def test_change_point_location(self, shifted_data):
        ref, comp = shifted_data
        full = np.concatenate([ref, comp])
        detector = CUSUMDetector(threshold=5.0, drift=0.5)
        result = detector.detect(ref, comp, full)
        # change point는 ref period 이후에 있어야 함
        if result.detected_cp_index >= 0:
            assert result.detected_cp_index >= len(ref) - 10  # 약간의 여유

    def test_constant_data(self):
        """상수 데이터에서 false positive 없어야 함"""
        ref = np.zeros(150)
        comp = np.zeros(150)
        full = np.concatenate([ref, comp])
        detector = CUSUMDetector(threshold=5.0)
        result = detector.detect(ref, comp, full)
        assert not result.is_detected
