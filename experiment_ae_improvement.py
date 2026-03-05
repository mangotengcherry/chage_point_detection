"""
AE Dual-Path Pipeline 성능 개선 실험

현재 성능: P=1.000, R=0.627, F1=0.770
목표: Precision 0.9+ 유지하면서 Recall 향상

실험 조건:
1. FDR alpha 조정 (0.05 -> 0.10, 0.15, 0.20)
2. 교집합 -> 합집합 전략 변경
3. AE 아키텍처 변경 (더 깊은/얕은 네트워크)
4. Hybrid: 통계검정 Primary + AE 보조 필터
5. Relaxed Intersection: AE OR Raw 중 하나라도 높은 confidence면 포함

* sudden_jump 유형은 평가에서 제외 (포커스 X)
"""
import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

from src.data_generation import BINDataGenerator
from src.dual_path_pipeline import DualPathPipeline
from src.detectors import MannWhitneyDetector, KSTestDetector, TTestDetector
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests


def compute_metrics(y_true, y_pred, exclude_types=None, anomaly_types=None):
    """Precision, Recall, F1 산출. exclude_types에 해당하는 anomaly는 평가에서 제외"""
    if exclude_types and anomaly_types:
        exclude_indices = set()
        for idx, atype in anomaly_types.items():
            if atype in exclude_types:
                exclude_indices.add(idx)

        mask = np.ones(len(y_true), dtype=bool)
        for idx in exclude_indices:
            mask[idx] = False
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def experiment_alpha_tuning(dataset):
    """실험 1: FDR alpha 조정"""
    print("\n" + "=" * 60)
    print("  실험 1: FDR Alpha 조정")
    print("=" * 60)

    alphas = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30]
    results = []

    for alpha in alphas:
        pipeline = DualPathPipeline(
            hidden_dims=[256, 128, 64], epochs=100, batch_size=64,
            lr=1e-3, alpha=alpha, fdr_method="fdr_bh",
            train_ratio=0.8, random_state=42,
        )
        t0 = time.time()
        result = pipeline.run(dataset.ref_data, dataset.comp_data, dataset.feature_names)
        elapsed = time.time() - t0

        y_pred = result.intersection.astype(int)

        # 전체 평가
        metrics_all = compute_metrics(dataset.labels, y_pred)
        # sudden_jump 제외 평가
        metrics_no_sj = compute_metrics(
            dataset.labels, y_pred,
            exclude_types={"sudden_jump"}, anomaly_types=dataset.anomaly_types
        )

        results.append({
            "alpha": alpha,
            "P (all)": metrics_all["precision"],
            "R (all)": metrics_all["recall"],
            "F1 (all)": metrics_all["f1"],
            "P (no SJ)": metrics_no_sj["precision"],
            "R (no SJ)": metrics_no_sj["recall"],
            "F1 (no SJ)": metrics_no_sj["f1"],
            "Detected": int(y_pred.sum()),
            "FP": metrics_all["fp"],
            "Time(s)": round(elapsed, 2),
        })
        print(f"    alpha={alpha:.2f}: P={metrics_all['precision']:.3f}, "
              f"R={metrics_all['recall']:.3f}, F1={metrics_all['f1']:.3f}, "
              f"FP={metrics_all['fp']}, Detected={int(y_pred.sum())}")

    return pd.DataFrame(results)


def experiment_union_strategy(dataset):
    """실험 2: 교집합 vs 합집합 vs Relaxed"""
    print("\n" + "=" * 60)
    print("  실험 2: 교집합 vs 합집합 vs Relaxed 전략")
    print("=" * 60)

    # 기본 파이프라인 실행
    pipeline = DualPathPipeline(
        hidden_dims=[256, 128, 64], epochs=100, batch_size=64,
        lr=1e-3, alpha=0.10, fdr_method="fdr_bh",
        train_ratio=0.8, random_state=42,
    )
    result = pipeline.run(dataset.ref_data, dataset.comp_data, dataset.feature_names)

    strategies = {}

    # 1) 교집합 (기존)
    strategies["Intersection (AE AND Raw)"] = result.intersection.astype(int)

    # 2) 합집합
    union = (result.ae_significant | result.raw_significant).astype(int)
    strategies["Union (AE OR Raw)"] = union

    # 3) Relaxed: Raw 유의 OR (AE 유의 AND 높은 KS)
    ks_threshold = np.percentile(result.raw_ks_statistics[result.raw_ks_statistics > 0], 50)
    relaxed = result.raw_significant | (result.ae_significant & (result.raw_ks_statistics > ks_threshold))
    strategies["Relaxed (Raw OR AE+highKS)"] = relaxed.astype(int)

    # 4) Raw만 (alpha=0.10)
    strategies["Raw Only (alpha=0.10)"] = result.raw_significant.astype(int)

    # 5) AE만
    strategies["AE Only"] = result.ae_significant.astype(int)

    rows = []
    for name, y_pred in strategies.items():
        m_all = compute_metrics(dataset.labels, y_pred)
        m_no_sj = compute_metrics(
            dataset.labels, y_pred,
            exclude_types={"sudden_jump"}, anomaly_types=dataset.anomaly_types
        )
        rows.append({
            "Strategy": name,
            "P (all)": m_all["precision"],
            "R (all)": m_all["recall"],
            "F1 (all)": m_all["f1"],
            "P (no SJ)": m_no_sj["precision"],
            "R (no SJ)": m_no_sj["recall"],
            "F1 (no SJ)": m_no_sj["f1"],
            "Detected": int(y_pred.sum()),
            "FP": m_all["fp"],
        })
        print(f"    {name}: P={m_all['precision']:.3f}, R={m_all['recall']:.3f}, "
              f"F1={m_all['f1']:.3f}, FP={m_all['fp']}")

    return pd.DataFrame(rows)


def experiment_architecture(dataset):
    """실험 3: AE 아키텍처 변경"""
    print("\n" + "=" * 60)
    print("  실험 3: AE 아키텍처 비교")
    print("=" * 60)

    architectures = {
        "Shallow [128, 64]": [128, 64],
        "Default [256, 128, 64]": [256, 128, 64],
        "Deep [512, 256, 128, 64]": [512, 256, 128, 64],
        "Wide [512, 256, 128]": [512, 256, 128],
        "Narrow [128, 32]": [128, 32],
    }

    results = []
    for name, dims in architectures.items():
        pipeline = DualPathPipeline(
            hidden_dims=dims, epochs=100, batch_size=64,
            lr=1e-3, alpha=0.10, fdr_method="fdr_bh",
            train_ratio=0.8, random_state=42,
        )
        t0 = time.time()
        result = pipeline.run(dataset.ref_data, dataset.comp_data, dataset.feature_names)
        elapsed = time.time() - t0

        y_pred = result.intersection.astype(int)
        m_all = compute_metrics(dataset.labels, y_pred)
        m_no_sj = compute_metrics(
            dataset.labels, y_pred,
            exclude_types={"sudden_jump"}, anomaly_types=dataset.anomaly_types
        )

        results.append({
            "Architecture": name,
            "P (all)": m_all["precision"],
            "R (all)": m_all["recall"],
            "F1 (all)": m_all["f1"],
            "P (no SJ)": m_no_sj["precision"],
            "R (no SJ)": m_no_sj["recall"],
            "F1 (no SJ)": m_no_sj["f1"],
            "FP": m_all["fp"],
            "Time(s)": round(elapsed, 2),
        })
        print(f"    {name}: P={m_all['precision']:.3f}, R={m_all['recall']:.3f}, "
              f"F1={m_all['f1']:.3f}, FP={m_all['fp']}, Time={elapsed:.2f}s")

    return pd.DataFrame(results)


def experiment_hybrid(dataset):
    """실험 4: Hybrid - 통계검정 Primary + AE 보조"""
    print("\n" + "=" * 60)
    print("  실험 4: Hybrid 전략 (통계검정 + AE Dual-Path)")
    print("=" * 60)

    # AE Dual-Path (alpha=0.10)
    pipeline = DualPathPipeline(
        hidden_dims=[256, 128, 64], epochs=100, batch_size=64,
        lr=1e-3, alpha=0.10, fdr_method="fdr_bh",
        train_ratio=0.8, random_state=42,
    )
    dp_result = pipeline.run(dataset.ref_data, dataset.comp_data, dataset.feature_names)

    # 통계검정 결과
    stat_detectors = {
        "T-test": TTestDetector(alpha=0.05),
        "KS Test": KSTestDetector(alpha=0.05),
        "Mann-Whitney U": MannWhitneyDetector(alpha=0.05),
    }

    stat_results = {}
    for name, det in stat_detectors.items():
        det_results = det.detect_all(dataset)
        stat_results[name] = np.array([r.is_detected for r in det_results], dtype=bool)

    # Hybrid 전략들
    strategies = {}

    # 1) T-test + AE Dual-Path 합집합
    strategies["T-test OR AE-DP"] = (stat_results["T-test"] | dp_result.intersection).astype(int)

    # 2) T-test + AE-DP 교집합 -> T-test only 보완
    # T-test 결과를 기본으로 하되, AE-DP가 발견한 것도 추가 (높은 confidence만)
    ttest_base = stat_results["T-test"].copy()
    ae_high_conf = dp_result.intersection & (dp_result.raw_ks_statistics > 0.5)
    strategies["T-test + AE(highConf)"] = (ttest_base | ae_high_conf).astype(int)

    # 3) 2개 이상 통계검정이 동의 + AE 보조
    stat_agreement = (stat_results["T-test"].astype(int) +
                      stat_results["KS Test"].astype(int) +
                      stat_results["Mann-Whitney U"].astype(int))
    strategies["2+ StatTests agree"] = (stat_agreement >= 2).astype(int)

    # 4) 2+ 통계검정 동의 OR AE-DP
    strategies["2+ StatTests OR AE-DP"] = ((stat_agreement >= 2) | dp_result.intersection).astype(int)

    # 5) 모든 통계검정 동의 (가장 보수적)
    strategies["All 3 StatTests agree"] = (stat_agreement == 3).astype(int)

    # 6) T-test 단독 (기준선)
    strategies["T-test only (baseline)"] = stat_results["T-test"].astype(int)

    # 7) AE Dual-Path 단독 (기준선)
    strategies["AE-DP only (baseline)"] = dp_result.intersection.astype(int)

    rows = []
    for name, y_pred in strategies.items():
        m_all = compute_metrics(dataset.labels, y_pred)
        m_no_sj = compute_metrics(
            dataset.labels, y_pred,
            exclude_types={"sudden_jump"}, anomaly_types=dataset.anomaly_types
        )
        rows.append({
            "Strategy": name,
            "P (all)": m_all["precision"],
            "R (all)": m_all["recall"],
            "F1 (all)": m_all["f1"],
            "P (no SJ)": m_no_sj["precision"],
            "R (no SJ)": m_no_sj["recall"],
            "F1 (no SJ)": m_no_sj["f1"],
            "Detected": int(y_pred.sum()),
            "FP": m_all["fp"],
        })
        print(f"    {name}: P={m_all['precision']:.3f}, R={m_all['recall']:.3f}, "
              f"F1={m_all['f1']:.3f}, FP={m_all['fp']}")

    return pd.DataFrame(rows)


def visualize_results(exp1_df, exp2_df, exp3_df, exp4_df, output_dir):
    """실험 결과 시각화"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 실험 1: Alpha 튜닝
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    ax.plot(exp1_df["alpha"], exp1_df["P (all)"], "o-", label="Precision", color="#1f77b4")
    ax.plot(exp1_df["alpha"], exp1_df["R (all)"], "s-", label="Recall", color="#ff7f0e")
    ax.plot(exp1_df["alpha"], exp1_df["F1 (all)"], "^-", label="F1", color="#2ca02c")
    ax.axhline(0.9, color="red", linestyle="--", alpha=0.5, label="P=0.9 target")
    ax.set_xlabel("FDR Alpha")
    ax.set_ylabel("Score")
    ax.set_title("전체 (sudden_jump 포함)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    ax2 = axes[1]
    ax2.plot(exp1_df["alpha"], exp1_df["P (no SJ)"], "o-", label="Precision", color="#1f77b4")
    ax2.plot(exp1_df["alpha"], exp1_df["R (no SJ)"], "s-", label="Recall", color="#ff7f0e")
    ax2.plot(exp1_df["alpha"], exp1_df["F1 (no SJ)"], "^-", label="F1", color="#2ca02c")
    ax2.axhline(0.9, color="red", linestyle="--", alpha=0.5, label="P=0.9 target")
    ax2.set_xlabel("FDR Alpha")
    ax2.set_ylabel("Score")
    ax2.set_title("Sudden Jump 제외")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    fig.suptitle("실험 1: FDR Alpha 튜닝", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "exp1_alpha_tuning.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 실험 4: Hybrid 전략 비교
    fig, ax = plt.subplots(figsize=(12, 7))
    strategies = exp4_df["Strategy"].tolist()
    x = np.arange(len(strategies))
    width = 0.25

    ax.bar(x - width, exp4_df["P (no SJ)"], width, label="Precision (no SJ)", color="#1f77b4")
    ax.bar(x, exp4_df["R (no SJ)"], width, label="Recall (no SJ)", color="#ff7f0e")
    ax.bar(x + width, exp4_df["F1 (no SJ)"], width, label="F1 (no SJ)", color="#2ca02c")
    ax.axhline(0.9, color="red", linestyle="--", alpha=0.5)
    ax.axhline(0.8, color="orange", linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title("실험 4: Hybrid 전략 비교 (Sudden Jump 제외)", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    for i, (p, r, f1) in enumerate(zip(
        exp4_df["P (no SJ)"], exp4_df["R (no SJ)"], exp4_df["F1 (no SJ)"]
    )):
        ax.text(i - width, p + 0.02, f"{p:.2f}", ha="center", fontsize=6)
        ax.text(i, r + 0.02, f"{r:.2f}", ha="center", fontsize=6)
        ax.text(i + width, f1 + 0.02, f"{f1:.2f}", ha="center", fontsize=6)

    fig.tight_layout()
    fig.savefig(output_dir / "exp4_hybrid_strategies.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 종합 비교: P vs R scatter (모든 실험)
    fig, ax = plt.subplots(figsize=(10, 8))

    # exp1
    ax.scatter(exp1_df["R (no SJ)"], exp1_df["P (no SJ)"],
               s=80, c="#1f77b4", marker="o", label="Exp1: Alpha tuning", zorder=3)
    for _, row in exp1_df.iterrows():
        ax.annotate(f"a={row['alpha']}", (row["R (no SJ)"], row["P (no SJ)"]),
                    fontsize=7, xytext=(5, 5), textcoords="offset points")

    # exp2
    ax.scatter(exp2_df["R (no SJ)"], exp2_df["P (no SJ)"],
               s=80, c="#ff7f0e", marker="s", label="Exp2: Strategy", zorder=3)
    for _, row in exp2_df.iterrows():
        ax.annotate(row["Strategy"][:15], (row["R (no SJ)"], row["P (no SJ)"]),
                    fontsize=6, xytext=(5, -10), textcoords="offset points")

    # exp4
    ax.scatter(exp4_df["R (no SJ)"], exp4_df["P (no SJ)"],
               s=80, c="#2ca02c", marker="^", label="Exp4: Hybrid", zorder=3)
    for _, row in exp4_df.iterrows():
        ax.annotate(row["Strategy"][:15], (row["R (no SJ)"], row["P (no SJ)"]),
                    fontsize=6, xytext=(5, -10), textcoords="offset points")

    ax.axhline(0.9, color="red", linestyle="--", alpha=0.3)
    ax.axhline(0.8, color="orange", linestyle="--", alpha=0.3)
    ax.axvline(0.8, color="orange", linestyle="--", alpha=0.3)
    ax.set_xlabel("Recall (no SJ)", fontsize=12)
    ax.set_ylabel("Precision (no SJ)", fontsize=12)
    ax.set_title("Precision vs Recall Trade-off (Sudden Jump 제외)", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(output_dir / "overall_pr_tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 60)
    print("  AE Dual-Path 성능 개선 실험")
    print("  (sudden_jump 제외 평가 포함)")
    print("=" * 60)

    # 데이터 생성
    print("\n[데이터 생성]")
    generator = BINDataGenerator(
        n_ref=1000, n_comp=100, n_features=500,
        n_anomaly_per_type_per_difficulty=10, random_state=42,
    )
    dataset = generator.generate()
    print(f"  Ref: {dataset.ref_data.shape}, Comp: {dataset.comp_data.shape}")
    print(f"  Anomaly: {dataset.labels.sum()}, Normal: {(dataset.labels == 0).sum()}")

    # 실험 실행
    exp1_df = experiment_alpha_tuning(dataset)
    exp2_df = experiment_union_strategy(dataset)
    exp3_df = experiment_architecture(dataset)
    exp4_df = experiment_hybrid(dataset)

    # 결과 저장
    output_dir = Path("docs/ae_improvement")
    output_dir.mkdir(parents=True, exist_ok=True)

    exp1_df.to_csv(output_dir / "exp1_alpha_tuning.csv", index=False)
    exp2_df.to_csv(output_dir / "exp2_strategy.csv", index=False)
    exp3_df.to_csv(output_dir / "exp3_architecture.csv", index=False)
    exp4_df.to_csv(output_dir / "exp4_hybrid.csv", index=False)

    # 시각화
    print("\n[시각화 생성]")
    visualize_results(exp1_df, exp2_df, exp3_df, exp4_df, output_dir)

    # 최종 요약
    print("\n" + "=" * 60)
    print("  최종 요약 및 권고")
    print("=" * 60)

    from tabulate import tabulate
    print("\n[실험 1: Alpha 튜닝]")
    print(tabulate(exp1_df, headers="keys", tablefmt="grid", floatfmt=".3f"))

    print("\n[실험 2: 전략 비교]")
    print(tabulate(exp2_df, headers="keys", tablefmt="grid", floatfmt=".3f"))

    print("\n[실험 3: 아키텍처]")
    print(tabulate(exp3_df, headers="keys", tablefmt="grid", floatfmt=".3f"))

    print("\n[실험 4: Hybrid]")
    print(tabulate(exp4_df, headers="keys", tablefmt="grid", floatfmt=".3f"))

    # Best 조합 찾기
    print("\n" + "=" * 60)
    print("  P >= 0.9 조건에서 F1 최고 조합")
    print("=" * 60)

    all_results = []
    for _, row in exp1_df.iterrows():
        all_results.append({"Exp": f"Exp1:a={row['alpha']}", **row.to_dict()})
    for _, row in exp2_df.iterrows():
        all_results.append({"Exp": f"Exp2:{row['Strategy']}", **row.to_dict()})
    for _, row in exp4_df.iterrows():
        all_results.append({"Exp": f"Exp4:{row['Strategy']}", **row.to_dict()})

    high_p = [r for r in all_results if r.get("P (no SJ)", 0) >= 0.9]
    if high_p:
        high_p.sort(key=lambda x: x.get("F1 (no SJ)", 0), reverse=True)
        print(f"\n  P(no SJ) >= 0.9 달성 조합 중 상위:")
        for i, r in enumerate(high_p[:5]):
            print(f"    {i+1}. {r['Exp']}: P={r.get('P (no SJ)',0):.3f}, "
                  f"R={r.get('R (no SJ)',0):.3f}, F1={r.get('F1 (no SJ)',0):.3f}")
    else:
        print("  P >= 0.9 달성 조합 없음. P >= 0.8 기준:")
        high_p = [r for r in all_results if r.get("P (no SJ)", 0) >= 0.8]
        high_p.sort(key=lambda x: x.get("F1 (no SJ)", 0), reverse=True)
        for i, r in enumerate(high_p[:5]):
            print(f"    {i+1}. {r['Exp']}: P={r.get('P (no SJ)',0):.3f}, "
                  f"R={r.get('R (no SJ)',0):.3f}, F1={r.get('F1 (no SJ)',0):.3f}")

    print(f"\n[완료] 결과가 {output_dir}/ 에 저장되었습니다.")


if __name__ == "__main__":
    main()
