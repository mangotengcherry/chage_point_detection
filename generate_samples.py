"""
Anomaly 유형별 x 난이도별 샘플 시각화 생성
5 유형 x 3 난이도 = 15개 차트를 docs/sample_views/ 에 저장
"""
import sys
import os
import warnings
import numpy as np
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

from src.data_generation import BINDataGenerator, ANOMALY_TYPES
from pathlib import Path


def main():
    print("=" * 60)
    print("  Anomaly 유형별 x 난이도별 샘플 시각화 생성")
    print("=" * 60)

    # 데이터 생성
    generator = BINDataGenerator(
        n_ref=1000, n_comp=100, n_features=500,
        n_anomaly_per_type_per_difficulty=10, random_state=42,
    )
    dataset = generator.generate()

    output_dir = Path("docs/sample_views")
    output_dir.mkdir(parents=True, exist_ok=True)

    anomaly_types = ANOMALY_TYPES
    difficulties = ["easy", "medium", "hard"]

    for atype in anomaly_types:
        for diff in difficulties:
            # 해당 유형+난이도에 해당하는 feature index 찾기
            candidates = [
                idx for idx, t in dataset.anomaly_types.items()
                if t == atype and dataset.difficulty_levels.get(idx) == diff
            ]
            if not candidates:
                continue

            feat_idx = candidates[0]
            ref_vals = dataset.ref_data[:, feat_idx]
            comp_vals = dataset.comp_data[:, feat_idx]
            feat_name = dataset.feature_names[feat_idx]

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # 1) Scatter: Ref wafer values
            ax1 = axes[0]
            ax1.scatter(range(len(ref_vals)), ref_vals,
                        alpha=0.4, s=8, color="#1f77b4", label="Ref (1000 wafers)")
            ax1.scatter(range(len(comp_vals)), comp_vals,
                        alpha=0.6, s=15, color="#d62728", label="Comp (100 wafers)")
            ax1.set_xlabel("Wafer Index")
            ax1.set_ylabel("Feature Value")
            ax1.set_title(f"Scatter - {feat_name}")
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)

            # 2) 분포 비교 (Histogram)
            ax2 = axes[1]
            ref_nonzero = ref_vals[ref_vals > 0]
            comp_nonzero = comp_vals[comp_vals > 0]

            if len(ref_nonzero) > 5 or len(comp_nonzero) > 5:
                all_vals = np.concatenate([ref_nonzero, comp_nonzero]) if len(comp_nonzero) > 0 else ref_nonzero
                bins = np.linspace(0, np.percentile(all_vals, 99), 40)
                ax2.hist(ref_vals, bins=bins, alpha=0.5, color="#1f77b4",
                         label=f"Ref (mean={np.mean(ref_vals):.5f})", density=True)
                ax2.hist(comp_vals, bins=bins, alpha=0.5, color="#d62728",
                         label=f"Comp (mean={np.mean(comp_vals):.5f})", density=True)
            else:
                ax2.hist(ref_vals, bins=30, alpha=0.5, color="#1f77b4",
                         label=f"Ref (mean={np.mean(ref_vals):.5f})", density=True)
                ax2.hist(comp_vals, bins=30, alpha=0.5, color="#d62728",
                         label=f"Comp (mean={np.mean(comp_vals):.5f})", density=True)
            ax2.set_xlabel("Feature Value")
            ax2.set_ylabel("Density")
            ax2.set_title("Distribution Comparison")
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)

            # 3) Box plot 비교
            ax3 = axes[2]
            bp = ax3.boxplot(
                [ref_vals, comp_vals],
                labels=["Ref", "Comp"],
                patch_artist=True,
                widths=0.5,
            )
            bp["boxes"][0].set_facecolor("#1f77b4")
            bp["boxes"][0].set_alpha(0.5)
            bp["boxes"][1].set_facecolor("#d62728")
            bp["boxes"][1].set_alpha(0.5)
            ax3.set_ylabel("Feature Value")
            ax3.set_title("Box Plot")
            ax3.grid(True, alpha=0.3)

            # 통계 정보 추가
            from scipy import stats
            ks_stat, ks_pval = stats.ks_2samp(ref_vals, comp_vals)
            try:
                mw_stat, mw_pval = stats.mannwhitneyu(ref_vals, comp_vals, alternative="two-sided")
            except ValueError:
                mw_pval = 1.0

            stat_text = (f"KS stat={ks_stat:.4f}, p={ks_pval:.2e}\n"
                         f"MW-U p={mw_pval:.2e}\n"
                         f"Ref zero%={100*(ref_vals==0).mean():.1f}%\n"
                         f"Comp zero%={100*(comp_vals==0).mean():.1f}%")
            ax3.text(0.95, 0.95, stat_text, transform=ax3.transAxes,
                     fontsize=8, verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            diff_label = {"easy": "Easy", "medium": "Medium", "hard": "Hard"}[diff]
            fig.suptitle(
                f"{atype} / {diff_label} - {feat_name}",
                fontsize=14, fontweight="bold", y=1.02
            )
            fig.tight_layout()

            filename = f"{atype}_{diff}.png"
            fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  [OK] {filename}")

    # 전체 요약 매트릭스 차트 (5x3 grid)
    print("\n  요약 매트릭스 생성 중...")
    fig, axes = plt.subplots(5, 3, figsize=(20, 28))

    for row, atype in enumerate(anomaly_types):
        for col, diff in enumerate(difficulties):
            ax = axes[row][col]
            candidates = [
                idx for idx, t in dataset.anomaly_types.items()
                if t == atype and dataset.difficulty_levels.get(idx) == diff
            ]
            if not candidates:
                ax.set_visible(False)
                continue

            feat_idx = candidates[0]
            ref_vals = dataset.ref_data[:, feat_idx]
            comp_vals = dataset.comp_data[:, feat_idx]

            ax.scatter(range(len(ref_vals)), ref_vals,
                       alpha=0.3, s=4, color="#1f77b4", label="Ref")
            ax.scatter(range(len(comp_vals)), comp_vals,
                       alpha=0.5, s=8, color="#d62728", label="Comp")

            diff_label = {"easy": "Easy", "medium": "Medium", "hard": "Hard"}[diff]
            ax.set_title(f"{atype} / {diff_label}", fontsize=10, fontweight="bold")
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel("Value", fontsize=9)
            if row == 4:
                ax.set_xlabel("Wafer Index", fontsize=9)
            if row == 0 and col == 2:
                ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Anomaly 유형별 x 난이도별 샘플 요약 (5 x 3)", fontsize=16, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "summary_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] summary_matrix.png")

    print(f"\n[완료] 모든 샘플 차트가 {output_dir}/ 에 저장되었습니다.")
    print(f"  - 개별 차트: 15개 ({', '.join(anomaly_types)})")
    print(f"  - 요약 매트릭스: summary_matrix.png")


if __name__ == "__main__":
    main()
