"""탐지기 테스트 (Wafer 기반)"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from src.detectors.statistical import (
    MannWhitneyDetector, KSTestDetector, TTestDetector, WelchTTestDetector
)
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
        result = detector.detect_feature(ref, comp)
        assert isinstance(result, DetectionResult)

    @pytest.mark.parametrize("DetectorClass", [
        MannWhitneyDetector, KSTestDetector, TTestDetector, WelchTTestDetector
    ])
    def test_detect_clear_shift(self, DetectorClass, shifted_data):
        """명확한 mean shift를 탐지하는지"""
        detector = DetectorClass(alpha=0.05)
        ref, comp = shifted_data
        result = detector.detect_feature(ref, comp)
        assert result.is_detected, f"{DetectorClass.__name__}가 3-sigma shift를 놓침"

    @pytest.mark.parametrize("DetectorClass", [
        MannWhitneyDetector, KSTestDetector, TTestDetector, WelchTTestDetector
    ])
    def test_confidence_range(self, DetectorClass, shifted_data):
        """confidence가 0~1 범위인지"""
        detector = DetectorClass()
        ref, comp = shifted_data
        result = detector.detect_feature(ref, comp)
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.parametrize("DetectorClass", [
        MannWhitneyDetector, KSTestDetector, TTestDetector, WelchTTestDetector
    ])
    def test_no_false_positive_normal(self, DetectorClass, normal_data):
        """동일 분포에서 false positive가 적어야 함"""
        detector = DetectorClass(alpha=0.01)
        ref, comp = normal_data
        result = detector.detect_feature(ref, comp)
        # 동일 분포에서도 우연히 유의할 수 있으나 confidence는 낮아야 함
        assert result.confidence < 0.99

    def test_constant_data(self):
        """상수 데이터에서 false positive 없어야 함"""
        ref = np.zeros(150)
        comp = np.zeros(150)
        for DetectorClass in [MannWhitneyDetector, TTestDetector, WelchTTestDetector]:
            detector = DetectorClass()
            result = detector.detect_feature(ref, comp)
            assert not result.is_detected
