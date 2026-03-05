"""
EDS BIN 변경점 탐지 벤치마크 실행 스크립트 (Wafer 기반)
6가지 방법: 통계검정 4종 + AE Standalone + AE Dual-Path Pipeline
"""
import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_generation import BINDataGenerator
from src.dual_path_pipeline import DualPathPipeline
from src.detectors import (
    MannWhitneyDetector,
    KSTestDetector,
    TTestDetector,
    WelchTTestDetector,
    AutoencoderDetector,
)
from src.evaluation import BenchmarkEvaluator
from src.benchmark_visualization import BenchmarkVisualizer
from tabulate import tabulate
from collections import Counter


def main():
    print("=" * 70)
    print("  EDS BIN Feature 유의차 검출 벤치마크 (Wafer 기반)")
    print("  Ref 1000 wafers x 500 features / Comp 100 wafers")
    print("=" * 70)

    # ──────────────────────────────────────────────
    # [1] 합성 BIN 데이터 생성
    # ──────────────────────────────────────────────
    print("\n[1] 합성 BIN 데이터 생성 중...")
    generator = BINDataGenerator(
        n_ref=1000,
        n_comp=100,
        n_features=500,
        n_anomaly_per_type_per_difficulty=10,
        random_state=42,
    )
    dataset = generator.generate()

    print(f"    Feature 수: {dataset.ref_data.shape[1]}")
    print(f"    Ref wafers: {dataset.ref_data.shape[0]}")
    print(f"    Comp wafers: {dataset.comp_data.shape[0]}")
    print(f"    Anomaly Feature 수: {dataset.labels.sum()}")
    print(f"    정상 Feature 수: {(dataset.labels == 0).sum()}")

    type_counts = Counter(dataset.anomaly_types.values())
    diff_counts = Counter(dataset.difficulty_levels.values())
    print(f"    유형별 분포: {dict(type_counts)}")
    print(f"    난이도별 분포: {dict(diff_counts)}")

    # ──────────────────────────────────────────────
    # [2] 탐지기 초기화
    # ──────────────────────────────────────────────
    print("\n[2] 탐지기 초기화...")
    detectors = [
        MannWhitneyDetector(alpha=0.05),
        KSTestDetector(alpha=0.05),
        TTestDetector(alpha=0.05),
        WelchTTestDetector(alpha=0.05),
        AutoencoderDetector(
            hidden_dims=[256, 128, 64],
            epochs=100,
            batch_size=64,
            lr=1e-3,
            train_ratio=0.8,
        ),
    ]
    for d in detectors:
        print(f"    - {d.name}")
    print(f"    - AE Dual-Path Pipeline (ECO)")

    # ──────────────────────────────────────────────
    # [3] 기본 탐지기 벤치마크 실행
    # ──────────────────────────────────────────────
    print("\n[3] 벤치마크 실행...")
    evaluator = BenchmarkEvaluator(dataset)

    # 각 탐지기별 검출된 feature 저장
    all_method_detections = {}

    for detector in detectors:
        print(f"    [{detector.name}] 평가 중...", end="", flush=True)
        result = evaluator.evaluate_detector(detector)
        m = result.overall_metrics
        print(
            f" -> P={m['precision']:.3f}, R={m['recall']:.3f}, "
            f"F1={m['f1']:.3f}, Time={result.execution_time:.3f}s"
        )

        # 검출된 feature indices 저장
        detected = np.array([r.is_detected for r in result.detection_results], dtype=bool)
        confidences = np.array([r.confidence for r in result.detection_results])
        all_method_detections[detector.name] = {
            "detected": detected,
            "confidences": confidences,
        }

    # ──────────────────────────────────────────────
    # [4] AE Dual-Path Pipeline (ECO 방법론)
    # ──────────────────────────────────────────────
    print("\n[4] AE Dual-Path Pipeline (ECO 방법론) 실행...")
    dual_pipeline = DualPathPipeline(
        hidden_dims=[256, 128, 64],
        epochs=100,
        batch_size=64,
        lr=1e-3,
        alpha=0.05,
        fdr_method="fdr_bh",
        train_ratio=0.8,
        random_state=42,
    )

    dp_start = time.time()
    dual_result = dual_pipeline.run(
        dataset.ref_data,
        dataset.comp_data,
        dataset.feature_names,
    )
    dp_elapsed = time.time() - dp_start

    # Dual-path 결과를 evaluator에 등록
    dp_benchmark = evaluator.evaluate_dual_path(dual_result, method_name="AE Dual-Path")
    dp_benchmark.execution_time = dp_elapsed

    m = dp_benchmark.overall_metrics
    print(f"    -> P={m['precision']:.3f}, R={m['recall']:.3f}, "
          f"F1={m['f1']:.3f}, Time={dp_elapsed:.3f}s")

    # AE Dual-Path 검출 결과도 저장
    all_method_detections["AE Dual-Path"] = {
        "detected": dual_result.intersection.astype(bool),
        "confidences": np.array([
            float(1.0 - min(dual_result.raw_pvalues_mw[j], dual_result.raw_pvalues_ks[j]))
            for j in range(len(dual_result.intersection))
        ]),
    }

    # ──────────────────────────────────────────────
    # [5] 결과 요약
    # ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  [5] 전체 성능 요약")
    print("=" * 70)

    summary_df = evaluator.summary_table()
    print(tabulate(summary_df, headers="keys", tablefmt="grid", floatfmt=".3f"))

    high_precision = summary_df[summary_df["Precision"] >= 0.8]
    print(f"\n* Precision >= 0.8 달성 방법: {len(high_precision)}개")
    if len(high_precision) > 0:
        for _, row in high_precision.iterrows():
            print(f"    - {row['Method']}: P={row['Precision']:.3f}, R={row['Recall']:.3f}, F1={row['F1']:.3f}")

    # 유형별 Recall
    print("\n" + "=" * 70)
    print("  [6] Anomaly 유형별 Recall")
    print("=" * 70)
    type_df = evaluator.type_breakdown_table()
    print(tabulate(type_df, headers="keys", tablefmt="grid", floatfmt=".3f"))

    # Dual-path 상세 결과
    print("\n" + "=" * 70)
    print("  [7] AE Dual-Path Pipeline 상세 결과")
    print("=" * 70)
    print(f"    Step 2 - AE Error 유의 Feature: {dual_result.n_ae_significant}개")
    print(f"    Step 3 - Raw Feature 유의 Feature: {dual_result.n_raw_significant}개")
    print(f"    Step 4 - 교집합 (최종): {dual_result.n_intersection}개")
    print(f"             AE만: {int(dual_result.ae_only.sum())}개")
    print(f"             Raw만: {int(dual_result.raw_only.sum())}개")

    # Ground truth 대비 분석
    gt_anomaly_idx = set(np.where(dataset.labels == 1)[0])
    dp_detected_idx = set(np.where(dual_result.intersection)[0])
    dp_tp = len(gt_anomaly_idx & dp_detected_idx)
    dp_fp = len(dp_detected_idx - gt_anomaly_idx)
    dp_fn = len(gt_anomaly_idx - dp_detected_idx)
    print(f"\n    Ground Truth 대비:")
    print(f"      TP (정확 검출): {dp_tp}")
    print(f"      FP (오탐): {dp_fp}")
    print(f"      FN (미탐): {dp_fn}")

    # 검출된 유의 Feature 목록 (상위 20개)
    print(f"\n    검출된 유의 Feature (교집합, 상위 20개):")
    detected_features = np.where(dual_result.intersection)[0]
    for j in detected_features[:20]:
        gt_label = "ANOMALY" if dataset.labels[j] == 1 else "NORMAL"
        atype = dataset.anomaly_types.get(int(j), "-")
        diff = dataset.difficulty_levels.get(int(j), "-")
        ks = dual_result.raw_ks_statistics[j]
        print(f"      {dataset.feature_names[j]:>8} | KS={ks:.4f} | {gt_label} ({atype}, {diff})")
    if len(detected_features) > 20:
        print(f"      ... 외 {len(detected_features) - 20}개")

    # ──────────────────────────────────────────────
    # [8] 계산비용 비교 분석
    # ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  [8] 계산비용 비교 분석")
    print("=" * 70)

    cost_rows = []
    for r in evaluator.results:
        n_features = dataset.ref_data.shape[1]
        per_feature_ms = (r.execution_time / n_features) * 1000 if n_features > 0 else 0

        # 확장성 추정 (1000 features, 5000 features)
        if r.method_name in ["Autoencoder", "AE Dual-Path"]:
            # AE는 feature 수에 따라 비선형 증가 (네트워크 크기 증가)
            est_1000 = r.execution_time * (1000 / n_features) * 1.3
            est_5000 = r.execution_time * (5000 / n_features) * 2.0
            complexity = "O(n_wafers * n_features * epochs)"
        else:
            # 통계검정은 feature 수에 비례
            est_1000 = r.execution_time * (1000 / n_features)
            est_5000 = r.execution_time * (5000 / n_features)
            complexity = "O(n_wafers * n_features)"

        cost_rows.append({
            "Method": r.method_name,
            "Total(s)": round(r.execution_time, 3),
            "Per Feature(ms)": round(per_feature_ms, 3),
            "F1": round(r.overall_metrics.get("f1", 0), 3),
            "Precision": round(r.overall_metrics.get("precision", 0), 3),
            "Est.1000feat(s)": round(est_1000, 2),
            "Est.5000feat(s)": round(est_5000, 2),
            "Complexity": complexity,
        })

    cost_df = pd.DataFrame(cost_rows).sort_values("Total(s)")
    print(tabulate(cost_df, headers="keys", tablefmt="grid", floatfmt=".3f"))

    # 효율성 지표: F1/Time ratio
    print("\n  효율성 지표 (F1 / Time):")
    for _, row in cost_df.iterrows():
        if row["Total(s)"] > 0:
            efficiency = row["F1"] / row["Total(s)"]
            print(f"    - {row['Method']}: {efficiency:.2f} F1/sec")

    # ──────────────────────────────────────────────
    # [9] 시각화
    # ──────────────────────────────────────────────
    print("\n[9] 시각화 생성 중...")
    viz = BenchmarkVisualizer(output_dir="docs/benchmark_results")
    viz.plot_all(evaluator, dataset, dual_result=dual_result,
                 all_method_detections=all_method_detections)

    # ──────────────────────────────────────────────
    # [10] 인사이트
    # ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  [10] 핵심 인사이트")
    print("=" * 70)

    best = summary_df.iloc[0]
    print(f"\n  1) 최고 F1 Score: {best['Method']} (F1={best['F1']:.3f})")

    best_p = summary_df.sort_values("Precision", ascending=False).iloc[0]
    print(f"  2) 최고 Precision: {best_p['Method']} (P={best_p['Precision']:.3f})")

    fast_df = summary_df[summary_df["F1"] > 0.3]
    if len(fast_df) > 0:
        fastest = fast_df.sort_values("Time(s)").iloc[0]
        print(f"  3) 가장 빠른 유효 방법: {fastest['Method']} ({fastest['Time(s)']:.3f}s, F1={fastest['F1']:.3f})")

    print(f"\n  4) 난이도별 평균 Recall:")
    for diff in ["easy", "medium", "hard"]:
        col = f"Recall_{diff}"
        if col in summary_df.columns:
            avg = summary_df[col].mean()
            print(f"     - {diff.capitalize()}: {avg:.3f}")

    print(f"\n  5) AE Dual-Path Pipeline:")
    print(f"     - 교집합 기반 cross-validation으로 FP 억제")
    print(f"     - AE Error + Raw Feature 이중 검증으로 신뢰도 향상")

    print(f"\n  6) 계산비용 분석:")
    stat_time = np.mean([r.execution_time for r in evaluator.results
                         if r.method_name not in ["Autoencoder", "AE Dual-Path"]])
    ae_time = np.mean([r.execution_time for r in evaluator.results
                       if r.method_name in ["Autoencoder", "AE Dual-Path"]])
    print(f"     - 통계검정 평균: {stat_time:.3f}s (실시간 서비스 적합)")
    print(f"     - AE 기반 평균: {ae_time:.3f}s (배치 분석 적합)")
    print(f"     - AE는 통계검정 대비 약 {ae_time/stat_time:.0f}배 느림")

    # 결과 저장
    summary_df.to_csv("docs/benchmark_results/summary_table.csv", index=False)
    type_df.to_csv("docs/benchmark_results/type_breakdown.csv", index=False)
    cost_df.to_csv("docs/benchmark_results/cost_analysis.csv", index=False)
    print(f"\n[완료] 모든 결과가 docs/benchmark_results/ 에 저장되었습니다.")


if __name__ == "__main__":
    main()
