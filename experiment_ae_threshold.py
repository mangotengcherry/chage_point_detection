"""
AE Threshold 전략 실험: Elbow Point vs Percentile 기반

현재 방식: IQR (Q75 + k*IQR) 기반 고정 threshold
실험 1: Elbow Point - 정렬된 error_ratio 곡선에서 급변점 자동 탐지
실험 2: Percentile - holdout error 분포 기반 percentile cutoff
실험 3: 종합 비교 - IQR vs Elbow vs Percentile vs Dual-Path

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
from src.dual_path_pipeline import DualPathPipeline, _FCAutoencoder
from pathlib import Path
from scipy import stats
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def compute_metrics(y_true, y_pred, exclude_types=None, anomaly_types=None):
    """Precision, Recall, F1 산출"""
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


def find_elbow_point(values_sorted):
    """
    Kneedle-style elbow point 탐지.
    정렬된 값 배열에서 곡률이 최대인 지점(급변점)을 찾는다.

    원리: 첫점-끝점 직선으로부터 각 점까지의 수직 거리가 최대인 점 = elbow
    """
    n = len(values_sorted)
    if n < 3:
        return n - 1

    # 정규화 (0~1 범위)
    x = np.linspace(0, 1, n)
    y_min, y_max = values_sorted.min(), values_sorted.max()
    if y_max - y_min < 1e-10:
        return n - 1
    y = (values_sorted - y_min) / (y_max - y_min)

    # 첫점-끝점 직선
    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-10:
        return n - 1
    line_unit = line_vec / line_len

    # 각 점에서 직선까지의 수직 거리
    distances = np.zeros(n)
    for i in range(n):
        point_vec = np.array([x[i], y[i]]) - p1
        # 직선에 투영
        proj_len = np.dot(point_vec, line_unit)
        proj = line_unit * proj_len
        perp = point_vec - proj
        distances[i] = np.linalg.norm(perp)

    return int(np.argmax(distances))


def train_ae_and_get_errors(ref_data, comp_data, hidden_dims=None, epochs=100,
                            batch_size=64, lr=1e-3, train_ratio=0.8, random_state=42):
    """AE를 학습하고 feature별 error를 반환"""
    hidden_dims = hidden_dims or [256, 128, 64]
    n_features = ref_data.shape[1]

    # 전처리
    ref_mean = np.mean(ref_data, axis=0, keepdims=True)
    ref_std = np.std(ref_data, axis=0, keepdims=True)
    ref_std[ref_std < 1e-10] = 1e-10
    ref_scaled = (ref_data - ref_mean) / ref_std
    comp_scaled = (comp_data - ref_mean) / ref_std

    # Train/Holdout 분리
    n_ref = ref_scaled.shape[0]
    n_train = int(n_ref * train_ratio)
    indices = np.arange(n_ref)
    rng = np.random.RandomState(random_state)
    rng.shuffle(indices)
    train_data = ref_scaled[indices[:n_train]]
    holdout_data = ref_scaled[indices[n_train:]]

    # AE 학습
    torch.manual_seed(random_state)
    device = torch.device("cpu")
    train_tensor = torch.FloatTensor(train_data).to(device)
    dataset_t = TensorDataset(train_tensor, train_tensor)
    loader = DataLoader(dataset_t, batch_size=batch_size, shuffle=True)

    model = _FCAutoencoder(n_features, hidden_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        for batch_x, _ in loader:
            optimizer.zero_grad()
            recon = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        holdout_recon = model(torch.FloatTensor(holdout_data)).numpy()
        comp_recon = model(torch.FloatTensor(comp_scaled)).numpy()

    # Per-sample, per-feature errors
    holdout_errors_full = (holdout_data - holdout_recon) ** 2  # (n_holdout, n_features)
    comp_errors_full = (comp_scaled - comp_recon) ** 2         # (n_comp, n_features)

    # Feature별 평균 error
    holdout_errors = np.mean(holdout_errors_full, axis=0)  # (n_features,)
    comp_errors = np.mean(comp_errors_full, axis=0)        # (n_features,)

    return {
        "holdout_errors": holdout_errors,
        "comp_errors": comp_errors,
        "holdout_errors_full": holdout_errors_full,
        "comp_errors_full": comp_errors_full,
        "error_ratio": comp_errors / (holdout_errors + 1e-10),
    }


def experiment_elbow(dataset, ae_results):
    """실험 1: Elbow Point 기반 threshold"""
    print("\n" + "=" * 60)
    print("  실험 1: Elbow Point Threshold")
    print("=" * 60)

    error_ratio = ae_results["error_ratio"]
    n_features = len(error_ratio)

    # error_ratio를 정렬
    sorted_indices = np.argsort(error_ratio)
    sorted_ratios = error_ratio[sorted_indices]

    # Elbow point 찾기
    elbow_idx = find_elbow_point(sorted_ratios)
    elbow_threshold = sorted_ratios[elbow_idx]

    print(f"    Elbow Point: index={elbow_idx}/{n_features}, threshold={elbow_threshold:.4f}")

    # Elbow 기준으로 검출
    y_pred_elbow = (error_ratio > elbow_threshold).astype(int)

    m_all = compute_metrics(dataset.labels, y_pred_elbow)
    m_no_sj = compute_metrics(
        dataset.labels, y_pred_elbow,
        exclude_types={"sudden_jump"}, anomaly_types=dataset.anomaly_types
    )

    print(f"    Elbow: P={m_all['precision']:.3f}, R={m_all['recall']:.3f}, "
          f"F1={m_all['f1']:.3f}, FP={m_all['fp']}, Detected={int(y_pred_elbow.sum())}")

    # Elbow 주변 변형 (elbow * factor)
    results = []
    factors = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]
    for factor in factors:
        thresh = elbow_threshold * factor
        y_pred = (error_ratio > thresh).astype(int)
        m_all = compute_metrics(dataset.labels, y_pred)
        m_no_sj = compute_metrics(
            dataset.labels, y_pred,
            exclude_types={"sudden_jump"}, anomaly_types=dataset.anomaly_types
        )

        results.append({
            "Factor": factor,
            "Threshold": round(thresh, 4),
            "P (all)": m_all["precision"],
            "R (all)": m_all["recall"],
            "F1 (all)": m_all["f1"],
            "P (no SJ)": m_no_sj["precision"],
            "R (no SJ)": m_no_sj["recall"],
            "F1 (no SJ)": m_no_sj["f1"],
            "Detected": int(y_pred.sum()),
            "FP": m_all["fp"],
        })
        print(f"    Factor={factor:.1f} (thresh={thresh:.4f}): "
              f"P={m_all['precision']:.3f}, R={m_all['recall']:.3f}, "
              f"F1={m_all['f1']:.3f}, FP={m_all['fp']}")

    return pd.DataFrame(results), sorted_ratios, elbow_idx, elbow_threshold


def experiment_percentile(dataset, ae_results):
    """실험 2: Percentile 기반 threshold"""
    print("\n" + "=" * 60)
    print("  실험 2: Percentile 기반 Threshold")
    print("=" * 60)

    error_ratio = ae_results["error_ratio"]
    holdout_errors = ae_results["holdout_errors"]
    comp_errors = ae_results["comp_errors"]

    results = []

    # 방법 A: error_ratio의 percentile을 threshold로 사용
    # "상위 X%를 anomaly로 판정"
    print("\n  [방법 A] Error Ratio Percentile Cutoff")
    percentiles = [70, 75, 80, 85, 90, 95, 97, 99]
    for pct in percentiles:
        thresh = np.percentile(error_ratio, pct)
        y_pred = (error_ratio > thresh).astype(int)
        m_all = compute_metrics(dataset.labels, y_pred)
        m_no_sj = compute_metrics(
            dataset.labels, y_pred,
            exclude_types={"sudden_jump"}, anomaly_types=dataset.anomaly_types
        )
        results.append({
            "Method": f"Ratio P{pct}",
            "Percentile": pct,
            "Threshold": round(thresh, 4),
            "P (all)": m_all["precision"],
            "R (all)": m_all["recall"],
            "F1 (all)": m_all["f1"],
            "P (no SJ)": m_no_sj["precision"],
            "R (no SJ)": m_no_sj["recall"],
            "F1 (no SJ)": m_no_sj["f1"],
            "Detected": int(y_pred.sum()),
            "FP": m_all["fp"],
        })
        print(f"    P{pct}: thresh={thresh:.4f}, P={m_all['precision']:.3f}, "
              f"R={m_all['recall']:.3f}, F1={m_all['f1']:.3f}, FP={m_all['fp']}")

    # 방법 B: Holdout error 분포 기반 - comp error가 holdout의 상위 percentile 초과
    # "설명력 기준": holdout에서의 정상 error 분포를 기준으로,
    # comp의 error가 정상 범위를 벗어나면 해당 feature는 AE가 "설명하지 못하는" 변화가 있음
    print("\n  [방법 B] Holdout Error 설명력 기준 (Comp error > Holdout Pxx)")
    holdout_errors_full = ae_results["holdout_errors_full"]  # (n_holdout, n_features)
    comp_errors_full = ae_results["comp_errors_full"]        # (n_comp, n_features)

    for pct in [90, 95, 97, 99]:
        y_pred = np.zeros(len(error_ratio), dtype=int)
        for j in range(len(error_ratio)):
            # holdout의 해당 feature error 분포에서 percentile 계산
            holdout_pct = np.percentile(holdout_errors_full[:, j], pct)
            # comp에서 이 percentile을 초과하는 비율
            comp_exceed_ratio = (comp_errors_full[:, j] > holdout_pct).mean()
            # holdout에서 기대되는 초과 비율보다 유의하게 높으면 검출
            expected_exceed = 1.0 - pct / 100.0
            if comp_exceed_ratio > expected_exceed * 2.0:  # 2배 이상이면 유의
                y_pred[j] = 1

        m_all = compute_metrics(dataset.labels, y_pred)
        m_no_sj = compute_metrics(
            dataset.labels, y_pred,
            exclude_types={"sudden_jump"}, anomaly_types=dataset.anomaly_types
        )
        results.append({
            "Method": f"Holdout P{pct} (x2)",
            "Percentile": pct,
            "Threshold": 0,
            "P (all)": m_all["precision"],
            "R (all)": m_all["recall"],
            "F1 (all)": m_all["f1"],
            "P (no SJ)": m_no_sj["precision"],
            "R (no SJ)": m_no_sj["recall"],
            "F1 (no SJ)": m_no_sj["f1"],
            "Detected": int(y_pred.sum()),
            "FP": m_all["fp"],
        })
        print(f"    Holdout P{pct} (x2): P={m_all['precision']:.3f}, "
              f"R={m_all['recall']:.3f}, F1={m_all['f1']:.3f}, FP={m_all['fp']}")

    # 방법 B 변형: 초과 비율 배수 조정 (x1.5, x3, x5)
    print("\n  [방법 B-2] 설명력 기준 초과 배수 변형 (P95 기준)")
    for mult in [1.5, 2.0, 3.0, 5.0]:
        y_pred = np.zeros(len(error_ratio), dtype=int)
        for j in range(len(error_ratio)):
            holdout_pct = np.percentile(holdout_errors_full[:, j], 95)
            comp_exceed_ratio = (comp_errors_full[:, j] > holdout_pct).mean()
            expected_exceed = 0.05  # 1 - 95/100
            if comp_exceed_ratio > expected_exceed * mult:
                y_pred[j] = 1

        m_all = compute_metrics(dataset.labels, y_pred)
        m_no_sj = compute_metrics(
            dataset.labels, y_pred,
            exclude_types={"sudden_jump"}, anomaly_types=dataset.anomaly_types
        )
        results.append({
            "Method": f"Holdout P95 (x{mult})",
            "Percentile": 95,
            "Threshold": mult,
            "P (all)": m_all["precision"],
            "R (all)": m_all["recall"],
            "F1 (all)": m_all["f1"],
            "P (no SJ)": m_no_sj["precision"],
            "R (no SJ)": m_no_sj["recall"],
            "F1 (no SJ)": m_no_sj["f1"],
            "Detected": int(y_pred.sum()),
            "FP": m_all["fp"],
        })
        print(f"    P95 x{mult}: P={m_all['precision']:.3f}, R={m_all['recall']:.3f}, "
              f"F1={m_all['f1']:.3f}, FP={m_all['fp']}")

    return pd.DataFrame(results)


def experiment_comprehensive(dataset, ae_results):
    """실험 3: IQR vs Elbow vs Percentile vs Dual-Path 종합 비교"""
    print("\n" + "=" * 60)
    print("  실험 3: 종합 Threshold 비교")
    print("=" * 60)

    error_ratio = ae_results["error_ratio"]
    holdout_errors_full = ae_results["holdout_errors_full"]
    comp_errors_full = ae_results["comp_errors_full"]
    n_features = len(error_ratio)

    results = []

    # 1) IQR 기반 (현재 방식, k=1.5, 2.0, 3.0)
    q75 = np.percentile(error_ratio, 75)
    q25 = np.percentile(error_ratio, 25)
    iqr = q75 - q25
    for k in [1.5, 2.0, 3.0]:
        thresh = q75 + k * iqr
        y_pred = (error_ratio > thresh).astype(int)
        m_all = compute_metrics(dataset.labels, y_pred)
        m_no_sj = compute_metrics(
            dataset.labels, y_pred,
            exclude_types={"sudden_jump"}, anomaly_types=dataset.anomaly_types
        )
        results.append({
            "Method": f"IQR (k={k})",
            "P (all)": m_all["precision"],
            "R (all)": m_all["recall"],
            "F1 (all)": m_all["f1"],
            "P (no SJ)": m_no_sj["precision"],
            "R (no SJ)": m_no_sj["recall"],
            "F1 (no SJ)": m_no_sj["f1"],
            "Detected": int(y_pred.sum()),
            "FP": m_all["fp"],
        })

    # 2) Elbow (factor=1.0)
    sorted_ratios = np.sort(error_ratio)
    elbow_idx = find_elbow_point(sorted_ratios)
    elbow_thresh = sorted_ratios[elbow_idx]
    y_pred = (error_ratio > elbow_thresh).astype(int)
    m_all = compute_metrics(dataset.labels, y_pred)
    m_no_sj = compute_metrics(
        dataset.labels, y_pred,
        exclude_types={"sudden_jump"}, anomaly_types=dataset.anomaly_types
    )
    results.append({
        "Method": f"Elbow Point",
        "P (all)": m_all["precision"],
        "R (all)": m_all["recall"],
        "F1 (all)": m_all["f1"],
        "P (no SJ)": m_no_sj["precision"],
        "R (no SJ)": m_no_sj["recall"],
        "F1 (no SJ)": m_no_sj["f1"],
        "Detected": int(y_pred.sum()),
        "FP": m_all["fp"],
    })

    # 3) Percentile (Ratio P85, P90)
    for pct in [85, 90]:
        thresh = np.percentile(error_ratio, pct)
        y_pred = (error_ratio > thresh).astype(int)
        m_all = compute_metrics(dataset.labels, y_pred)
        m_no_sj = compute_metrics(
            dataset.labels, y_pred,
            exclude_types={"sudden_jump"}, anomaly_types=dataset.anomaly_types
        )
        results.append({
            "Method": f"Ratio P{pct}",
            "P (all)": m_all["precision"],
            "R (all)": m_all["recall"],
            "F1 (all)": m_all["f1"],
            "P (no SJ)": m_no_sj["precision"],
            "R (no SJ)": m_no_sj["recall"],
            "F1 (no SJ)": m_no_sj["f1"],
            "Detected": int(y_pred.sum()),
            "FP": m_all["fp"],
        })

    # 4) Holdout 설명력 기준 (P95 x2, P95 x3)
    for mult in [2.0, 3.0]:
        y_pred = np.zeros(n_features, dtype=int)
        for j in range(n_features):
            holdout_pct = np.percentile(holdout_errors_full[:, j], 95)
            comp_exceed = (comp_errors_full[:, j] > holdout_pct).mean()
            if comp_exceed > 0.05 * mult:
                y_pred[j] = 1
        m_all = compute_metrics(dataset.labels, y_pred)
        m_no_sj = compute_metrics(
            dataset.labels, y_pred,
            exclude_types={"sudden_jump"}, anomaly_types=dataset.anomaly_types
        )
        results.append({
            "Method": f"Holdout P95 (x{mult})",
            "P (all)": m_all["precision"],
            "R (all)": m_all["recall"],
            "F1 (all)": m_all["f1"],
            "P (no SJ)": m_no_sj["precision"],
            "R (no SJ)": m_no_sj["recall"],
            "F1 (no SJ)": m_no_sj["f1"],
            "Detected": int(y_pred.sum()),
            "FP": m_all["fp"],
        })

    # 5) Dual-Path (alpha=0.05, 0.10)
    for alpha in [0.05, 0.10]:
        pipeline = DualPathPipeline(
            hidden_dims=[256, 128, 64], epochs=100, batch_size=64,
            lr=1e-3, alpha=alpha, fdr_method="fdr_bh",
            train_ratio=0.8, random_state=42,
        )
        result = pipeline.run(dataset.ref_data, dataset.comp_data, dataset.feature_names)
        y_pred = result.intersection.astype(int)
        m_all = compute_metrics(dataset.labels, y_pred)
        m_no_sj = compute_metrics(
            dataset.labels, y_pred,
            exclude_types={"sudden_jump"}, anomaly_types=dataset.anomaly_types
        )
        results.append({
            "Method": f"Dual-Path (a={alpha})",
            "P (all)": m_all["precision"],
            "R (all)": m_all["recall"],
            "F1 (all)": m_all["f1"],
            "P (no SJ)": m_no_sj["precision"],
            "R (no SJ)": m_no_sj["recall"],
            "F1 (no SJ)": m_no_sj["f1"],
            "Detected": int(y_pred.sum()),
            "FP": m_all["fp"],
        })

    df = pd.DataFrame(results)
    for _, row in df.iterrows():
        print(f"    {row['Method']:25s}: P={row['P (all)']:.3f}, R={row['R (all)']:.3f}, "
              f"F1={row['F1 (all)']:.3f}, FP={row['FP']}")

    return df


def visualize_threshold_results(exp1_df, exp2_df, exp3_df,
                                 sorted_ratios, elbow_idx, elbow_threshold,
                                 error_ratio, dataset, output_dir):
    """시각화 생성"""
    output_dir = Path(output_dir)

    # ── 차트 1: Elbow Point 시각화 ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1-1: 정렬된 error ratio 곡선 + elbow point
    ax = axes[0]
    ax.plot(sorted_ratios, color="#1f77b4", linewidth=1.5, label="Error Ratio (sorted)")
    ax.axvline(elbow_idx, color="red", linestyle="--", linewidth=2,
               label=f"Elbow Point (idx={elbow_idx})")
    ax.axhline(elbow_threshold, color="red", linestyle=":", alpha=0.7,
               label=f"Threshold={elbow_threshold:.4f}")
    ax.set_xlabel("Feature Index (sorted)")
    ax.set_ylabel("Error Ratio (comp/holdout)")
    ax.set_title("Elbow Point Detection")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 1-2: Elbow factor별 P/R/F1
    ax2 = axes[1]
    ax2.plot(exp1_df["Factor"], exp1_df["P (no SJ)"], "o-", label="Precision", color="#1f77b4")
    ax2.plot(exp1_df["Factor"], exp1_df["R (no SJ)"], "s-", label="Recall", color="#ff7f0e")
    ax2.plot(exp1_df["Factor"], exp1_df["F1 (no SJ)"], "^-", label="F1", color="#2ca02c")
    ax2.axhline(0.9, color="red", linestyle="--", alpha=0.3)
    ax2.axhline(0.8, color="orange", linestyle="--", alpha=0.3)
    ax2.axvline(1.0, color="gray", linestyle=":", alpha=0.5, label="Elbow (factor=1.0)")
    ax2.set_xlabel("Elbow Factor")
    ax2.set_ylabel("Score (no SJ)")
    ax2.set_title("Elbow Factor Tuning")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    fig.suptitle("실험 1: Elbow Point Threshold", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "exp_elbow_point.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 차트 2: Percentile 비교 ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Ratio percentile
    ratio_rows = exp2_df[exp2_df["Method"].str.startswith("Ratio")]
    ax = axes[0]
    ax.plot(ratio_rows["Percentile"], ratio_rows["P (no SJ)"], "o-", label="Precision", color="#1f77b4")
    ax.plot(ratio_rows["Percentile"], ratio_rows["R (no SJ)"], "s-", label="Recall", color="#ff7f0e")
    ax.plot(ratio_rows["Percentile"], ratio_rows["F1 (no SJ)"], "^-", label="F1", color="#2ca02c")
    ax.axhline(0.9, color="red", linestyle="--", alpha=0.3)
    ax.set_xlabel("Percentile Cutoff")
    ax.set_ylabel("Score (no SJ)")
    ax.set_title("Error Ratio Percentile Cutoff")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Holdout 설명력 기준
    hold_rows = exp2_df[exp2_df["Method"].str.startswith("Holdout")]
    ax2 = axes[1]
    x_labels = hold_rows["Method"].tolist()
    x_pos = np.arange(len(x_labels))
    width = 0.25
    ax2.bar(x_pos - width, hold_rows["P (no SJ)"], width, label="Precision", color="#1f77b4")
    ax2.bar(x_pos, hold_rows["R (no SJ)"], width, label="Recall", color="#ff7f0e")
    ax2.bar(x_pos + width, hold_rows["F1 (no SJ)"], width, label="F1", color="#2ca02c")
    ax2.axhline(0.9, color="red", linestyle="--", alpha=0.3)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel("Score (no SJ)")
    ax2.set_title("Holdout Error Percentile (Explained Variance)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_ylim(0, 1.1)

    fig.suptitle("실험 2: Percentile 기반 Threshold", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "exp_percentile.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 차트 3: 종합 비교 ──
    fig, ax = plt.subplots(figsize=(14, 8))

    methods = exp3_df["Method"].tolist()
    x = np.arange(len(methods))
    width = 0.25

    ax.bar(x - width, exp3_df["P (no SJ)"], width, label="Precision (no SJ)", color="#1f77b4")
    ax.bar(x, exp3_df["R (no SJ)"], width, label="Recall (no SJ)", color="#ff7f0e")
    ax.bar(x + width, exp3_df["F1 (no SJ)"], width, label="F1 (no SJ)", color="#2ca02c")

    ax.axhline(0.9, color="red", linestyle="--", alpha=0.3, label="P=0.9")
    ax.axhline(0.8, color="orange", linestyle="--", alpha=0.3, label="P=0.8")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title("AE Threshold 방법 종합 비교 (Sudden Jump 제외)", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="y")

    for i, (p, r, f1, fp) in enumerate(zip(
        exp3_df["P (no SJ)"], exp3_df["R (no SJ)"],
        exp3_df["F1 (no SJ)"], exp3_df["FP"]
    )):
        ax.text(i, max(p, r, f1) + 0.03, f"FP={fp}", ha="center", fontsize=7, color="red")

    fig.tight_layout()
    fig.savefig(output_dir / "exp_threshold_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 차트 4: P vs R Scatter ──
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = {"IQR": "#1f77b4", "Elbow": "#d62728", "Ratio": "#ff7f0e",
              "Holdout": "#2ca02c", "Dual": "#9467bd"}
    markers = {"IQR": "o", "Elbow": "D", "Ratio": "s",
               "Holdout": "^", "Dual": "P"}

    for _, row in exp3_df.iterrows():
        method = row["Method"]
        key = method.split(" ")[0].split("(")[0].rstrip()
        if key == "Dual-Path":
            key = "Dual"
        color = colors.get(key, "#7f7f7f")
        marker = markers.get(key, "x")
        ax.scatter(row["R (no SJ)"], row["P (no SJ)"], s=100,
                   c=color, marker=marker, zorder=3, edgecolors="black", linewidth=0.5)
        ax.annotate(method, (row["R (no SJ)"], row["P (no SJ)"]),
                    fontsize=7, xytext=(5, 5), textcoords="offset points")

    ax.axhline(0.9, color="red", linestyle="--", alpha=0.3)
    ax.axhline(0.8, color="orange", linestyle="--", alpha=0.3)
    ax.axvline(0.8, color="orange", linestyle="--", alpha=0.3)
    ax.set_xlabel("Recall (no SJ)", fontsize=12)
    ax.set_ylabel("Precision (no SJ)", fontsize=12)
    ax.set_title("AE Threshold: Precision vs Recall (Sudden Jump 제외)", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4",
               markersize=10, label="IQR"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#d62728",
               markersize=10, label="Elbow Point"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#ff7f0e",
               markersize=10, label="Ratio Percentile"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#2ca02c",
               markersize=10, label="Holdout Percentile"),
        Line2D([0], [0], marker="P", color="w", markerfacecolor="#9467bd",
               markersize=10, label="Dual-Path"),
    ]
    ax.legend(handles=legend_elements, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "exp_threshold_pr_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 60)
    print("  AE Threshold 전략 실험")
    print("  Elbow Point vs Percentile vs IQR vs Dual-Path")
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

    # AE 학습 + Error 산출 (1회만)
    print("\n[AE 학습 및 Error 산출]")
    t0 = time.time()
    ae_results = train_ae_and_get_errors(
        dataset.ref_data, dataset.comp_data,
        hidden_dims=[256, 128, 64], epochs=100,
    )
    print(f"  AE 학습 완료: {time.time() - t0:.2f}초")
    print(f"  Error ratio 범위: [{ae_results['error_ratio'].min():.4f}, "
          f"{ae_results['error_ratio'].max():.4f}]")
    print(f"  Error ratio 중앙값: {np.median(ae_results['error_ratio']):.4f}")

    # 실험 실행
    exp1_df, sorted_ratios, elbow_idx, elbow_threshold = experiment_elbow(dataset, ae_results)
    exp2_df = experiment_percentile(dataset, ae_results)
    exp3_df = experiment_comprehensive(dataset, ae_results)

    # 결과 저장
    output_dir = Path("docs/ae_threshold")
    output_dir.mkdir(parents=True, exist_ok=True)

    exp1_df.to_csv(output_dir / "exp1_elbow.csv", index=False)
    exp2_df.to_csv(output_dir / "exp2_percentile.csv", index=False)
    exp3_df.to_csv(output_dir / "exp3_comprehensive.csv", index=False)

    # 시각화
    print("\n[시각화 생성]")
    visualize_threshold_results(
        exp1_df, exp2_df, exp3_df,
        sorted_ratios, elbow_idx, elbow_threshold,
        ae_results["error_ratio"], dataset, output_dir
    )

    # 최종 요약
    print("\n" + "=" * 60)
    print("  최종 요약")
    print("=" * 60)

    from tabulate import tabulate
    print("\n[실험 1: Elbow Point]")
    print(tabulate(exp1_df, headers="keys", tablefmt="grid", floatfmt=".3f"))

    print("\n[실험 2: Percentile]")
    print(tabulate(exp2_df, headers="keys", tablefmt="grid", floatfmt=".3f"))

    print("\n[실험 3: 종합 비교]")
    print(tabulate(exp3_df, headers="keys", tablefmt="grid", floatfmt=".3f"))

    # Best 방법 찾기
    print("\n" + "=" * 60)
    print("  P >= 0.9 조건에서 F1 최고 (no SJ)")
    print("=" * 60)

    high_p = exp3_df[exp3_df["P (no SJ)"] >= 0.9].sort_values("F1 (no SJ)", ascending=False)
    if len(high_p) > 0:
        for i, (_, row) in enumerate(high_p.head(5).iterrows()):
            print(f"    {i+1}. {row['Method']:25s}: P={row['P (no SJ)']:.3f}, "
                  f"R={row['R (no SJ)']:.3f}, F1={row['F1 (no SJ)']:.3f}, FP={row['FP']}")

    print(f"\n[완료] 결과가 {output_dir}/ 에 저장되었습니다.")


if __name__ == "__main__":
    main()
