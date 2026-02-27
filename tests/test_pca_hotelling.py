"""PCA + Hotelling T² 모델 단위 테스트"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pca_hotelling import PCAHotellingT2
from src.preprocessing import DataPreprocessor


@pytest.fixture
def normal_data():
    """정상 분포 테스트 데이터"""
    rng = np.random.RandomState(42)
    n_samples, n_features = 100, 500
    X = rng.randn(n_samples, n_features)
    return X


@pytest.fixture
def shifted_data(normal_data):
    """일부 feature에 평균 이동을 준 데이터"""
    rng = np.random.RandomState(99)
    X = rng.randn(50, normal_data.shape[1])
    # feature 0~9에 평균 이동
    X[:, :10] += 5.0
    return X, list(range(10))


class TestPCAHotellingT2:
    """PCA + Hotelling T² 모델 테스트"""

    def test_fit_basic(self, normal_data):
        model = PCAHotellingT2(n_components=0.95)
        model.fit(normal_data)
        info = model.get_model_info()
        assert info["fitted"] is True
        assert info["n_features"] == 500
        assert info["n_ref_samples"] == 100
        assert info["total_explained_variance"] >= 0.95

    def test_analyze_normal(self, normal_data):
        """정상 데이터 → 유의차 비율 낮아야 함"""
        rng = np.random.RandomState(123)
        X_comp = rng.randn(50, normal_data.shape[1])

        model = PCAHotellingT2(n_components=0.95, alpha=0.01)
        model.fit(normal_data)
        result = model.analyze(X_comp)

        # alpha=0.01이므로 유의차 비율이 10% 미만이어야 정상
        sig_ratio = np.mean(result.is_significant_t2)
        assert sig_ratio < 0.15, f"정상 데이터의 유의차 비율이 너무 높음: {sig_ratio:.3f}"

    def test_analyze_shifted(self, normal_data, shifted_data):
        """변경 데이터 → 유의차 비율 높아야 함"""
        X_shifted, _ = shifted_data

        # 순수 랜덤 데이터에서는 주성분 수를 적게 설정해야 UCL이 합리적
        model = PCAHotellingT2(n_components=20, alpha=0.01)
        model.fit(normal_data)
        result = model.analyze(X_shifted)

        # T² 또는 SPE 중 하나라도 유의차 wafer 비율이 50% 이상이어야 함
        sig_ratio_t2 = np.mean(result.is_significant_t2)
        sig_ratio_spe = np.mean(result.is_significant_spe)
        sig_ratio = max(sig_ratio_t2, sig_ratio_spe)
        assert sig_ratio > 0.5, (
            f"변경 데이터의 유의차 비율이 너무 낮음: "
            f"T²={sig_ratio_t2:.3f}, SPE={sig_ratio_spe:.3f}"
        )

    def test_feature_contribution(self, normal_data, shifted_data):
        """변경된 feature가 기여도 상위에 위치해야 함"""
        X_shifted, true_changed = shifted_data

        model = PCAHotellingT2(n_components=0.95, alpha=0.01)
        model.fit(normal_data)
        result = model.analyze(X_shifted)

        # 상위 20개 중 실제 변경 feature가 포함되어야 함
        top_20 = set(result.significant_features[:20].tolist())
        true_set = set(true_changed)
        overlap = len(top_20 & true_set)

        assert overlap >= 5, (
            f"상위 20개 중 실제 변경 feature가 {overlap}개뿐 "
            f"(기대: 5개 이상, 실제 변경: {true_changed})"
        )

    def test_t2_ucl_positive(self, normal_data):
        model = PCAHotellingT2(n_components=10, alpha=0.01)
        model.fit(normal_data)
        result = model.analyze(normal_data[:10])
        assert result.t2_ucl > 0
        assert result.spe_ucl > 0

    def test_result_shapes(self, normal_data):
        model = PCAHotellingT2(n_components=10)
        model.fit(normal_data)

        X_comp = normal_data[:30]
        result = model.analyze(X_comp)

        assert result.t2_values.shape == (30,)
        assert result.spe_values.shape == (30,)
        assert result.t2_contributions.shape == (30, 500)
        assert result.spe_contributions.shape == (30, 500)
        assert result.feature_importance.shape == (500,)

    def test_not_fitted_error(self):
        model = PCAHotellingT2()
        with pytest.raises(RuntimeError, match="fit"):
            model.analyze(np.random.randn(10, 50))


class TestDataPreprocessor:
    """데이터 전처리기 테스트"""

    def test_feature_classification(self):
        rng = np.random.RandomState(42)
        n = 200
        data = np.column_stack([
            rng.binomial(1, 0.9, n),         # binary
            rng.poisson(2, n),                # discrete
            rng.normal(100, 10, n),           # continuous normal
            rng.lognormal(3, 1, n),           # continuous skewed
        ])
        names = ["bin", "disc", "normal", "skewed"]

        pp = DataPreprocessor()
        types = pp.analyze_features(data, names)

        assert types[0].dtype == "binary"
        assert types[1].dtype == "discrete"
        assert types[2].dtype in ("continuous_normal", "continuous_skewed")
        assert types[3].dtype == "continuous_skewed"

    def test_continuous_extraction(self):
        rng = np.random.RandomState(42)
        n = 100
        data = np.column_stack([
            rng.binomial(1, 0.9, n),
            rng.normal(100, 10, n),
            rng.normal(200, 20, n),
        ])

        pp = DataPreprocessor()
        pp.analyze_features(data, ["bin", "cont1", "cont2"])

        cont = pp.get_continuous_features(data)
        assert cont.shape[1] == 2  # binary 제외

        disc = pp.get_discrete_features(data)
        assert disc.shape[1] == 1  # binary만


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
