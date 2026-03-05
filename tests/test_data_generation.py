"""합성 BIN 데이터 생성기 테스트 (Wafer 기반)"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from src.data_generation import BINDataGenerator, SyntheticBINDataset


@pytest.fixture
def dataset():
    gen = BINDataGenerator(
        n_ref=1000, n_comp=100, n_features=500,
        n_anomaly_per_type_per_difficulty=10, random_state=42
    )
    return gen.generate()


class TestBINDataGenerator:
    def test_output_shape(self, dataset):
        assert dataset.ref_data.shape == (1000, 500)
        assert dataset.comp_data.shape == (100, 500)
        assert len(dataset.feature_names) == 500
        assert len(dataset.labels) == 500

    def test_anomaly_count(self, dataset):
        """총 150개 anomaly Feature (5유형 x 3난이도 x 10개)"""
        assert dataset.labels.sum() == 150
        assert (dataset.labels == 0).sum() == 350

    def test_anomaly_type_distribution(self, dataset):
        """각 유형별 30개씩"""
        from collections import Counter
        type_counts = Counter(dataset.anomaly_types.values())
        for atype in ["sporadic_spikes", "level_shift", "gradual_trend",
                       "complex_trend", "sudden_jump"]:
            assert type_counts[atype] == 30

    def test_difficulty_distribution(self, dataset):
        """각 난이도별 50개씩"""
        from collections import Counter
        diff_counts = Counter(dataset.difficulty_levels.values())
        assert diff_counts["easy"] == 50
        assert diff_counts["medium"] == 50
        assert diff_counts["hard"] == 50

    def test_non_negative(self, dataset):
        """BIN 값은 항상 0 이상"""
        assert np.all(dataset.ref_data >= 0)
        assert np.all(dataset.comp_data >= 0)

    def test_reproducibility(self):
        """동일 seed -> 동일 데이터"""
        gen1 = BINDataGenerator(random_state=42)
        gen2 = BINDataGenerator(random_state=42)
        ds1 = gen1.generate()
        ds2 = gen2.generate()
        np.testing.assert_array_equal(ds1.ref_data, ds2.ref_data)
        np.testing.assert_array_equal(ds1.comp_data, ds2.comp_data)
        np.testing.assert_array_equal(ds1.labels, ds2.labels)

    def test_feature_names(self, dataset):
        assert dataset.feature_names[0] == "BIN130"
        assert dataset.feature_names[-1] == "BIN629"
