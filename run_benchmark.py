"""
EDS BIN 변경점 탐지 벤치마크 실행 스크립트 (Wafer 기반)
7가지 방법: 통계검정 4종 + PCA+T² + AE Standalone + AE Dual-Path Pipeline
"""
import sys
import os
import time
import warnings
import numpy as np
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_generation import BINDataGenerator
from src.dual_path_pipeline import DualPathPipeline
from src.detectors import (
    MannWhitneyDetector,
    KSTestDetector,
    TTestDetector,
    WelchTTestDetector,
    PCAHotellingAdapter,
    AutoencoderDetector,
)
from src.evaluation import BenchmarkEvaluator
from src.benchmark_visualization import BenchmarkVisualizer
from tabulate import tabulate
from collections import Counter


def main():
    print("=" * 70)
    print("  EDS BIN Feature 유의차 검출 벤치마크 (Wafer 기반)")
    print("  Ref 1000 wafers × 500 features / Comp 100 wafers")
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
        PCAHotellingAdapter(n_components=0.95, alpha=0.01),
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

    for detector in detectors:
        print(f"    [{detector.name}] 평가 중...", end="", flush=True)
        result = evaluator.evaluate_detector(detector)
        m = result.overall_metrics
        print(
            f" -> P={m['precision']:.3f}, R={m['recall']:.3f}, "
            f"F1={m['f1']:.3f}, Time={result.execution_time:.3f}s"
        )

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
    # [8] 시각화
    # ──────────────────────────────────────────────
    print("\n[8] 시각화 생성 중...")
    viz = BenchmarkVisualizer(output_dir="docs/benchmark_results")
    viz.plot_all(evaluator, dataset, dual_result=dual_result)

    # ──────────────────────────────────────────────
    # [9] 인사이트
    # ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  [9] 핵심 인사이트")
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

    # 결과 저장
    summary_df.to_csv("docs/benchmark_results/summary_table.csv", index=False)
    type_df.to_csv("docs/benchmark_results/type_breakdown.csv", index=False)
    print(f"\n[완료] 모든 결과가 docs/benchmark_results/ 에 저장되었습니다.")


if __name__ == "__main__":
    main()
