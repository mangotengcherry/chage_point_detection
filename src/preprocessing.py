"""
데이터 전처리 모듈

EDS Test 데이터의 다양한 분포 유형(Binary, Discrete, Continuous, Skewed)을
식별하고 유형별로 적절한 변환을 적용한다.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy import stats


@dataclass
class FeatureTypeInfo:
    """Feature 유형 분류 결과"""

    name: str
    dtype: str  # "binary", "discrete", "continuous_normal", "continuous_skewed"
    unique_ratio: float
    skewness: float
    recommended_transform: str


class DataPreprocessor:
    """
    반도체 EDS 데이터 전처리기

    Feature 유형을 자동 분류하고, PCA+T² 분석에 적합한 형태로 변환한다.
    Binary/Discrete feature는 별도 통계검정 대상으로 분리한다.

    Parameters
    ----------
    unique_ratio_threshold : float, default=0.05
        고유값 비율이 이 값 미만이면 discrete로 분류
    skewness_threshold : float, default=1.0
        |skewness|가 이 값 초과이면 skewed로 분류
    """

    def __init__(
        self,
        unique_ratio_threshold: float = 0.05,
        skewness_threshold: float = 1.0,
    ):
        self.unique_ratio_threshold = unique_ratio_threshold
        self.skewness_threshold = skewness_threshold
        self._feature_types: list[FeatureTypeInfo] = []
        self._continuous_mask: np.ndarray | None = None

    def analyze_features(
        self,
        data: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> list[FeatureTypeInfo]:
        """
        Feature 유형 자동 분류

        Returns
        -------
        list[FeatureTypeInfo]
            각 feature의 유형 정보
        """
        n_samples, n_features = data.shape
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(n_features)]

        self._feature_types = []
        continuous_flags = []

        for j in range(n_features):
            col = data[:, j]
            valid = col[~np.isnan(col)]

            if len(valid) == 0:
                info = FeatureTypeInfo(
                    name=feature_names[j],
                    dtype="constant",
                    unique_ratio=0.0,
                    skewness=0.0,
                    recommended_transform="exclude",
                )
                continuous_flags.append(False)
            else:
                n_unique = len(np.unique(valid))
                unique_ratio = n_unique / len(valid)
                skew = float(stats.skew(valid)) if len(valid) > 2 else 0.0

                if n_unique <= 2:
                    dtype = "binary"
                    transform = "chi_squared_test"
                    continuous_flags.append(False)
                elif unique_ratio < self.unique_ratio_threshold:
                    dtype = "discrete"
                    transform = "chi_squared_test"
                    continuous_flags.append(False)
                elif abs(skew) > self.skewness_threshold:
                    dtype = "continuous_skewed"
                    transform = "log_or_boxcox"
                    continuous_flags.append(True)
                else:
                    dtype = "continuous_normal"
                    transform = "standard_scaling"
                    continuous_flags.append(True)

                info = FeatureTypeInfo(
                    name=feature_names[j],
                    dtype=dtype,
                    unique_ratio=unique_ratio,
                    skewness=skew,
                    recommended_transform=transform,
                )

            self._feature_types.append(info)

        self._continuous_mask = np.array(continuous_flags)
        return self._feature_types

    def get_continuous_features(self, data: np.ndarray) -> np.ndarray:
        """PCA+T² 분석 대상인 continuous feature만 추출"""
        if self._continuous_mask is None:
            raise RuntimeError("analyze_features()를 먼저 호출하세요.")
        return data[:, self._continuous_mask]

    def get_discrete_features(self, data: np.ndarray) -> np.ndarray:
        """별도 통계검정 대상인 discrete/binary feature 추출"""
        if self._continuous_mask is None:
            raise RuntimeError("analyze_features()를 먼저 호출하세요.")
        return data[:, ~self._continuous_mask]

    def get_continuous_feature_names(self) -> list[str]:
        """Continuous feature 이름 목록"""
        return [
            ft.name
            for ft, mask in zip(self._feature_types, self._continuous_mask)
            if mask
        ]

    def get_discrete_feature_names(self) -> list[str]:
        """Discrete/Binary feature 이름 목록"""
        return [
            ft.name
            for ft, mask in zip(self._feature_types, self._continuous_mask)
            if not mask
        ]

    def transform_skewed(self, data: np.ndarray) -> np.ndarray:
        """Skewed feature에 log 변환 적용"""
        result = data.copy()
        for j in range(data.shape[1]):
            if self._continuous_mask[j] and self._feature_types[j].dtype == "continuous_skewed":
                col = data[:, j]
                min_val = np.nanmin(col)
                if min_val <= 0:
                    result[:, j] = np.log1p(col - min_val + 1)
                else:
                    result[:, j] = np.log1p(col)
        return result

    def summary(self) -> dict:
        """Feature 유형 분포 요약"""
        if not self._feature_types:
            return {}

        type_counts = {}
        for ft in self._feature_types:
            type_counts[ft.dtype] = type_counts.get(ft.dtype, 0) + 1

        return {
            "total_features": len(self._feature_types),
            "type_distribution": type_counts,
            "continuous_count": int(np.sum(self._continuous_mask)),
            "discrete_count": int(np.sum(~self._continuous_mask)),
        }
