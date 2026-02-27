"""
변경점 분석 시각화 모듈

PCA + Hotelling T² 분석 결과를 다양한 차트로 시각화한다.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from typing import Optional

from .pca_hotelling import HotellingResult

# 한글 폰트 설정
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False


class ChangePointVisualizer:
    """변경점 분석 결과 시각화"""

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_t2_chart(
        self,
        result: HotellingResult,
        title: str = "Hotelling T² Control Chart",
        save_name: Optional[str] = None,
    ) -> plt.Figure:
        """T² 관리도"""
        fig, ax = plt.subplots(figsize=(12, 5))

        n = len(result.t2_values)
        x = np.arange(n)

        colors = [
            "red" if sig else "steelblue" for sig in result.is_significant_t2
        ]
        ax.bar(x, result.t2_values, color=colors, alpha=0.7, width=0.8)
        ax.axhline(
            y=result.t2_ucl, color="red", linestyle="--", linewidth=2, label=f"UCL = {result.t2_ucl:.2f}"
        )

        ax.set_xlabel("Wafer Index")
        ax.set_ylabel("T² Statistic")
        ax.set_title(title)
        ax.legend()

        sig_count = np.sum(result.is_significant_t2)
        ax.text(
            0.02, 0.95,
            f"Significant: {sig_count}/{n} wafers ({sig_count/n*100:.1f}%)",
            transform=ax.transAxes, fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        if save_name:
            fig.savefig(self.output_dir / save_name, dpi=150, bbox_inches="tight")
        return fig

    def plot_spe_chart(
        self,
        result: HotellingResult,
        title: str = "SPE (Q) Control Chart",
        save_name: Optional[str] = None,
    ) -> plt.Figure:
        """SPE 관리도"""
        fig, ax = plt.subplots(figsize=(12, 5))

        n = len(result.spe_values)
        x = np.arange(n)

        colors = [
            "red" if sig else "steelblue" for sig in result.is_significant_spe
        ]
        ax.bar(x, result.spe_values, color=colors, alpha=0.7, width=0.8)
        ax.axhline(
            y=result.spe_ucl, color="red", linestyle="--", linewidth=2, label=f"UCL = {result.spe_ucl:.2f}"
        )

        ax.set_xlabel("Wafer Index")
        ax.set_ylabel("SPE (Q) Statistic")
        ax.set_title(title)
        ax.legend()

        plt.tight_layout()
        if save_name:
            fig.savefig(self.output_dir / save_name, dpi=150, bbox_inches="tight")
        return fig

    def plot_contribution(
        self,
        result: HotellingResult,
        feature_names: Optional[list[str]] = None,
        top_k: int = 30,
        title: str = "Feature Contribution (Top-K)",
        save_name: Optional[str] = None,
    ) -> plt.Figure:
        """Feature 기여도 차트 (Contribution Plot)"""
        fig, ax = plt.subplots(figsize=(14, 6))

        importance = result.feature_importance
        sorted_idx = result.significant_features[:top_k]
        sorted_importance = importance[sorted_idx]

        if feature_names is not None:
            labels = [feature_names[i] for i in sorted_idx]
        else:
            labels = [f"F_{i}" for i in sorted_idx]

        colors = plt.cm.Reds(np.linspace(0.8, 0.3, top_k))
        ax.barh(range(top_k), sorted_importance[::-1], color=colors[::-1])
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(labels[::-1], fontsize=8)
        ax.set_xlabel("Contribution Score")
        ax.set_title(title)

        plt.tight_layout()
        if save_name:
            fig.savefig(self.output_dir / save_name, dpi=150, bbox_inches="tight")
        return fig

    def plot_error_rank_curve(
        self,
        result: HotellingResult,
        feature_names: Optional[list[str]] = None,
        title: str = "Reconstruction Error Rank Curve (PCA-based)",
        save_name: Optional[str] = None,
    ) -> plt.Figure:
        """
        Error Rank 곡선 - AE의 reconstruction error rank와 동일한 시각화

        ㄴ자 형태 → 좋은 분리 (소수의 high contribution + 다수의 low)
        선형/knee → 분리 부족
        """
        fig, ax = plt.subplots(figsize=(12, 5))

        importance = result.feature_importance
        sorted_importance = np.sort(importance)[::-1]
        n_features = len(sorted_importance)

        ax.plot(
            range(n_features), sorted_importance,
            color="steelblue", linewidth=1.5,
        )
        ax.fill_between(
            range(n_features), sorted_importance,
            alpha=0.2, color="steelblue",
        )

        # Elbow point 대신 통계적 threshold 표시
        mean_imp = np.mean(importance)
        std_imp = np.std(importance)
        threshold = mean_imp + 2 * std_imp

        n_above = np.sum(sorted_importance > threshold)
        ax.axhline(
            y=threshold, color="red", linestyle="--", linewidth=1.5,
            label=f"Threshold (mean+2*std) = {threshold:.4f}",
        )
        ax.axvline(
            x=n_above, color="orange", linestyle=":", linewidth=1.5,
            label=f"Significant features: {n_above}",
        )

        ax.set_xlabel("Feature Rank (by contribution)")
        ax.set_ylabel("Contribution Score")
        ax.set_title(title)
        ax.legend()

        # 형태 판정 텍스트
        ratio_top10 = np.sum(sorted_importance[:max(1, n_features // 10)]) / np.sum(sorted_importance)
        if ratio_top10 > 0.7:
            shape_text = "Shape: ㄴ (L-shape) - Good separation"
            shape_color = "green"
        elif ratio_top10 > 0.4:
            shape_text = "Shape: Knee - Moderate separation"
            shape_color = "orange"
        else:
            shape_text = "Shape: Linear - Poor separation"
            shape_color = "red"

        ax.text(
            0.98, 0.95, shape_text,
            transform=ax.transAxes, fontsize=11, fontweight="bold",
            ha="right", va="top", color=shape_color,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.tight_layout()
        if save_name:
            fig.savefig(self.output_dir / save_name, dpi=150, bbox_inches="tight")
        return fig

    def plot_dashboard(
        self,
        result: HotellingResult,
        feature_names: Optional[list[str]] = None,
        top_k: int = 20,
        title: str = "Change Point Analysis Dashboard",
        save_name: Optional[str] = None,
    ) -> plt.Figure:
        """4-panel 종합 대시보드"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(title, fontsize=16, fontweight="bold")

        # (1) T² Control Chart
        ax = axes[0, 0]
        n = len(result.t2_values)
        colors = ["red" if s else "steelblue" for s in result.is_significant_t2]
        ax.bar(range(n), result.t2_values, color=colors, alpha=0.7)
        ax.axhline(y=result.t2_ucl, color="red", linestyle="--", label=f"UCL={result.t2_ucl:.2f}")
        ax.set_title("Hotelling T² Control Chart")
        ax.set_xlabel("Wafer Index")
        ax.set_ylabel("T²")
        ax.legend(fontsize=8)

        # (2) SPE Control Chart
        ax = axes[0, 1]
        colors = ["red" if s else "steelblue" for s in result.is_significant_spe]
        ax.bar(range(n), result.spe_values, color=colors, alpha=0.7)
        ax.axhline(y=result.spe_ucl, color="red", linestyle="--", label=f"UCL={result.spe_ucl:.2f}")
        ax.set_title("SPE (Q) Control Chart")
        ax.set_xlabel("Wafer Index")
        ax.set_ylabel("SPE")
        ax.legend(fontsize=8)

        # (3) Error Rank Curve
        ax = axes[1, 0]
        importance = result.feature_importance
        sorted_imp = np.sort(importance)[::-1]
        ax.plot(range(len(sorted_imp)), sorted_imp, color="steelblue", linewidth=1.5)
        ax.fill_between(range(len(sorted_imp)), sorted_imp, alpha=0.2, color="steelblue")
        threshold = np.mean(importance) + 2 * np.std(importance)
        ax.axhline(y=threshold, color="red", linestyle="--", label=f"Threshold={threshold:.4f}")
        ax.set_title("Feature Contribution Rank Curve")
        ax.set_xlabel("Feature Rank")
        ax.set_ylabel("Contribution")
        ax.legend(fontsize=8)

        # (4) Top-K Contribution
        ax = axes[1, 1]
        sorted_idx = result.significant_features[:top_k]
        sorted_importance = importance[sorted_idx]
        if feature_names is not None:
            labels = [feature_names[i] for i in sorted_idx]
        else:
            labels = [f"F_{i}" for i in sorted_idx]
        colors_bar = plt.cm.Reds(np.linspace(0.8, 0.3, top_k))
        ax.barh(range(top_k), sorted_importance[::-1], color=colors_bar[::-1])
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(labels[::-1], fontsize=7)
        ax.set_title(f"Top-{top_k} Feature Contributions")
        ax.set_xlabel("Contribution Score")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save_name:
            fig.savefig(self.output_dir / save_name, dpi=150, bbox_inches="tight")
        return fig
