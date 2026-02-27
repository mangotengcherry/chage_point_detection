"""
PCA + Hotelling T² 기반 변경점 분석 모듈

반도체 EDS 데이터의 Ref Group vs Comp Group 간 유의차를 검출하고,
어떤 Feature(Test Item)가 변경점에 기여했는지 식별한다.

References:
    - Hotelling, H. (1947). Multivariate Quality Control.
    - Kourti, T. & MacGregor, J.F. (1995). Process Analysis, Monitoring
      and Diagnosis Using Multivariate Projection Methods.
    - Wise, B.M. & Gallagher, N.B. (1996). The Process Chemometrics
      Approach to Process Monitoring and Fault Detection.
"""

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HotellingResult:
    """Hotelling T² 분석 결과를 담는 데이터 클래스"""

    # T² 통계량 (wafer별)
    t2_values: np.ndarray
    # SPE (Q) 통계량 (wafer별)
    spe_values: np.ndarray
    # T² 임계값 (UCL)
    t2_ucl: float
    # SPE 임계값 (UCL)
    spe_ucl: float
    # Feature별 T² 기여도 (wafer × feature)
    t2_contributions: np.ndarray
    # Feature별 SPE 기여도 (wafer × feature)
    spe_contributions: np.ndarray
    # 유의차 판정 (wafer별)
    is_significant_t2: np.ndarray
    is_significant_spe: np.ndarray
    # Feature별 평균 기여도 (유의차 wafer 대상)
    feature_importance: np.ndarray = field(default_factory=lambda: np.array([]))
    # 유의차 feature 인덱스 (기여도 순 정렬)
    significant_features: np.ndarray = field(default_factory=lambda: np.array([]))


class PCAHotellingT2:
    """
    PCA + Hotelling T² 기반 변경점 분석 모델

    Ref Group 데이터로 PCA 모델을 학습하고,
    Comp Group 데이터에 대해 T²/SPE 통계량을 산출하여
    유의차 여부 및 변경 Feature를 식별한다.

    Parameters
    ----------
    n_components : int or float, default=0.95
        PCA 주성분 수. int이면 고정 개수, float이면 설명 분산 비율.
    alpha : float, default=0.01
        유의수준 (UCL 산출용). 0.01 = 99% 신뢰수준.
    """

    def __init__(
        self,
        n_components: int | float = 0.95,
        alpha: float = 0.01,
    ):
        self.n_components = n_components
        self.alpha = alpha

        self._pca: Optional[PCA] = None
        self._scaler: Optional[StandardScaler] = None
        self._ref_scores: Optional[np.ndarray] = None
        self._ref_t2: Optional[np.ndarray] = None
        self._ref_spe: Optional[np.ndarray] = None
        self._n_samples: int = 0
        self._n_features: int = 0
        self._n_components_fitted: int = 0
        self._is_fitted: bool = False

    def fit(self, X_ref: np.ndarray) -> "PCAHotellingT2":
        """
        Ref Group 데이터로 PCA 모델 학습

        Parameters
        ----------
        X_ref : np.ndarray, shape (n_samples, n_features)
            Reference Group 데이터 (정상 조건)
        """
        self._n_samples, self._n_features = X_ref.shape

        # StandardScaler로 정규화
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_ref)

        # PCA 학습
        self._pca = PCA(n_components=self.n_components)
        self._ref_scores = self._pca.fit_transform(X_scaled)
        self._n_components_fitted = self._pca.n_components_

        # Ref group의 T², SPE 계산 (UCL 산출용)
        self._ref_t2 = self._compute_t2(self._ref_scores)
        X_reconstructed = self._pca.inverse_transform(self._ref_scores)
        self._ref_spe = np.sum((X_scaled - X_reconstructed) ** 2, axis=1)

        self._is_fitted = True
        return self

    def analyze(
        self,
        X_comp: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> HotellingResult:
        """
        Comp Group 데이터에 대해 변경점 분석 수행

        Parameters
        ----------
        X_comp : np.ndarray, shape (n_samples, n_features)
            Compare Group 데이터 (평가 조건)
        feature_names : list[str], optional
            Feature 이름 목록

        Returns
        -------
        HotellingResult
            분석 결과
        """
        if not self._is_fitted:
            raise RuntimeError("fit()을 먼저 호출하세요.")

        # Ref와 동일한 scaler로 정규화
        X_scaled = self._scaler.transform(X_comp)

        # PCA 투영
        scores = self._pca.transform(X_scaled)

        # T² 통계량
        t2_values = self._compute_t2(scores)

        # SPE 통계량
        X_reconstructed = self._pca.inverse_transform(scores)
        residuals = X_scaled - X_reconstructed
        spe_values = np.sum(residuals**2, axis=1)

        # UCL 산출
        t2_ucl = self._compute_t2_ucl()
        spe_ucl = self._compute_spe_ucl()

        # 유의차 판정
        is_sig_t2 = t2_values > t2_ucl
        is_sig_spe = spe_values > spe_ucl

        # Feature별 기여도 산출
        t2_contrib = self._compute_t2_contributions(X_scaled)
        spe_contrib = residuals**2

        # 유의차 wafer들의 평균 기여도로 Feature 중요도 산출
        is_significant = is_sig_t2 | is_sig_spe
        if np.any(is_significant):
            # T²와 SPE 기여도를 합산하여 종합 기여도 산출
            combined_contrib = t2_contrib + spe_contrib
            feature_importance = np.mean(
                combined_contrib[is_significant], axis=0
            )
        else:
            # 유의차 wafer가 없으면 전체 평균 사용
            combined_contrib = t2_contrib + spe_contrib
            feature_importance = np.mean(combined_contrib, axis=0)

        # 기여도 순으로 정렬
        sorted_idx = np.argsort(feature_importance)[::-1]

        return HotellingResult(
            t2_values=t2_values,
            spe_values=spe_values,
            t2_ucl=t2_ucl,
            spe_ucl=spe_ucl,
            t2_contributions=t2_contrib,
            spe_contributions=spe_contrib,
            is_significant_t2=is_sig_t2,
            is_significant_spe=is_sig_spe,
            feature_importance=feature_importance,
            significant_features=sorted_idx,
        )

    def get_model_info(self) -> dict:
        """모델 정보 반환"""
        if not self._is_fitted:
            return {"fitted": False}

        return {
            "fitted": True,
            "n_components": self._n_components_fitted,
            "n_features": self._n_features,
            "n_ref_samples": self._n_samples,
            "explained_variance_ratio": self._pca.explained_variance_ratio_.tolist(),
            "total_explained_variance": float(
                np.sum(self._pca.explained_variance_ratio_)
            ),
            "alpha": self.alpha,
        }

    # ---- Private Methods ----

    def _compute_t2(self, scores: np.ndarray) -> np.ndarray:
        """T² 통계량 산출: T² = Σ(t_i² / λ_i)"""
        eigenvalues = self._pca.explained_variance_
        return np.sum(scores**2 / eigenvalues, axis=1)

    def _compute_t2_ucl(self) -> float:
        """T² 임계값 (F-분포 기반)"""
        N = self._n_samples
        k = self._n_components_fitted
        f_crit = stats.f.ppf(1 - self.alpha, k, N - k)
        return k * (N**2 - 1) / (N * (N - k)) * f_crit

    def _compute_spe_ucl(self) -> float:
        """SPE 임계값 (Chi-squared 근사 또는 Ref 경험적 분위수)"""
        # Box (1954) 근사법: SPE ~ g * chi2(h)
        # g = variance(SPE) / (2 * mean(SPE))
        # h = 2 * mean(SPE)^2 / variance(SPE)
        ref_spe = self._ref_spe
        mu = np.mean(ref_spe)
        var = np.var(ref_spe)

        if var < 1e-12:
            return mu * 3  # fallback

        g = var / (2 * mu)
        h = 2 * mu**2 / var
        return g * stats.chi2.ppf(1 - self.alpha, h)

    def _compute_t2_contributions(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Feature별 T² 기여도 산출 (Contribution Plot)

        Miller et al. (1998) 방식:
        cont_j = x_j * Σ_i (p_ij * t_i / λ_i)
        """
        scores = self._pca.transform(X_scaled)
        loadings = self._pca.components_.T  # (n_features, n_components)
        eigenvalues = self._pca.explained_variance_

        # (n_samples, n_components) / (n_components,) -> weighted scores
        weighted_scores = scores / eigenvalues  # (n_samples, n_components)

        # (n_samples, n_components) @ (n_components, n_features) -> (n_samples, n_features)
        contributions = X_scaled * (weighted_scores @ loadings.T)

        # 기여도는 양수로 (절대값)
        return np.abs(contributions)
