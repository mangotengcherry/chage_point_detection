"""합성 BIN 데이터 생성기 테스트"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from src.data_generation import BINDataGenerator, SyntheticBINDataset


@pytest.fixture
def dataset():
    gen = BINDataGenerator(n_timepoints=300, n_bins=470, random_state=42)
    return gen.generate()


class TestBINDataGenerator:
    def test_output_shape(self, dataset):
        assert dataset.data.shape == (300, 470)
        assert len(dataset.bin_names) == 470
        assert len(dataset.labels) == 470

    def test_anomaly_count(self, dataset):
        """총 150개 anomaly BIN (5유형 × 3난이도 × 10개)"""
        assert dataset.labels.sum() == 150
        assert (dataset.labels == 0).sum() == 320

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

    def test_ref_end_index(self, dataset):
        assert dataset.ref_end_index == 150

    def test_non_negative(self, dataset):
        """BIN 값은 항상 0 이상"""
        assert np.all(dataset.data >= 0)

    def test_reproducibility(self):
        """동일 seed → 동일 데이터"""
        gen1 = BINDataGenerator(random_state=42)
        gen2 = BINDataGenerator(random_state=42)
        ds1 = gen1.generate()
        ds2 = gen2.generate()
        np.testing.assert_array_equal(ds1.data, ds2.data)
        np.testing.assert_array_equal(ds1.labels, ds2.labels)

    def test_change_points_in_comp_period(self, dataset):
        """모든 change point가 comp period 내에 있는지"""
        for idx, cp in dataset.change_points.items():
            assert cp >= dataset.ref_end_index, \
                f"BIN {idx}의 change point({cp})가 ref period 내에 있음"

    def test_bin_names(self, dataset):
        assert dataset.bin_names[0] == "BIN130"
        assert dataset.bin_names[-1] == "BIN599"
