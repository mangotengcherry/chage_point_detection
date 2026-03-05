"""
EDS BIN 변경점 탐지 벤치마크 실행 스크립트
10가지 방법 × 5가지 anomaly 유형 × 3단계 난이도 = 종합 성능 비교
"""
import sys
import os
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_generation import BINDataGenerator
from src.detectors import (
    MannWhitneyDetector,
    KSTestDetector,
    TTestDetector,
    WelchTTestDetector,
    CUSUMDetector,
    RupturesPeltDetector,
    RupturesBinsegDetector,
    RupturesWindowDetector,
    PCAHotellingAdapter,
    AutoencoderDetector,
)
from src.evaluation import BenchmarkEvaluator
from src.benchmark_visualization import BenchmarkVisualizer
from tabulate import tabulate


def main():
    print("=" * 70)
    print("  EDS BIN 변경점 탐지 벤치마크")
    print("  5가지 Anomaly 유형 × 3단계 난이도 × 10가지 탐지 방법")
    print("=" * 70)

    # ──────────────────────────────────────────────
    # [1] 합성 BIN 데이터 생성
    # ──────────────────────────────────────────────
    print("\n[1] 합성 BIN 데이터 생성 중...")
    generator = BINDataGenerator(
        n_timepoints=300,
        n_bins=470,
        n_anomaly_per_type_per_difficulty=10,
        ref_ratio=0.5,
        random_state=42,
    )
    dataset = generator.generate()

    print(f"    전체 BIN 수: {dataset.data.shape[1]}")
    print(f"    시계열 길이: {dataset.data.shape[0]} (Ref: {dataset.ref_end_index}, Comp: {dataset.data.shape[0] - dataset.ref_end_index})")
    print(f"    Anomaly BIN 수: {dataset.labels.sum()}")
    print(f"    정상 BIN 수: {(dataset.labels == 0).sum()}")

    # Anomaly 유형별 분포
    from collections import Counter
    type_counts = Counter(dataset.anomaly_types.values())
    diff_counts = Counter(dataset.difficulty_levels.values())
    print(f"    유형별 분포: {dict(type_counts)}")
    print(f"    난이도별 분포: {dict(diff_counts)}")

    # ──────────────────────────────────────────────
    # [2] 탐지기 초기화
    # ──────────────────────────────────────────────
    print("\n[2] 탐지기 초기화...")
    detectors = [
        # 통계검정 (비모수)
        MannWhitneyDetector(alpha=0.05),
        KSTestDetector(alpha=0.05),
        # 통계검정 (모수)
        TTestDetector(alpha=0.05),
        WelchTTestDetector(alpha=0.05),
        # 시계열 변경점
        CUSUMDetector(threshold=5.0, drift=0.5),
        # ruptures
        RupturesPeltDetector(model="rbf", pen=3.0),
        RupturesBinsegDetector(model="rbf", n_bkps=2),
        RupturesWindowDetector(model="rbf", width=30, n_bkps=2),
        # 다변량
        PCAHotellingAdapter(n_components=0.95, alpha=0.01),
        # Deep Learning
        AutoencoderDetector(
            hidden_dims=[128, 64, 32],
            epochs=50,
            batch_size=32,
            lr=1e-3,
        ),
    ]
    for d in detectors:
        print(f"    - {d.name}")

    # ──────────────────────────────────────────────
    # [3] 벤치마크 실행
    # ──────────────────────────────────────────────
    print("\n[3] 벤치마크 실행...")
    evaluator = BenchmarkEvaluator(dataset)

    for detector in detectors:
        print(f"    [{detector.name}] 평가 중...", end="", flush=True)
        result = evaluator.evaluate_detector(detector)
        m = result.overall_metrics
        print(
            f" → P={m['precision']:.3f}, R={m['recall']:.3f}, "
            f"F1={m['f1']:.3f}, Time={result.execution_time:.3f}s"
        )

    # ──────────────────────────────────────────────
    # [4] 결과 요약
    # ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  [4] 전체 성능 요약")
    print("=" * 70)

    summary_df = evaluator.summary_table()
    print(tabulate(summary_df, headers="keys", tablefmt="grid", floatfmt=".3f"))

    # Precision 0.8 이상 달성 방법
    high_precision = summary_df[summary_df["Precision"] >= 0.8]
    print(f"\n★ Precision ≥ 0.8 달성 방법: {len(high_precision)}개")
    if len(high_precision) > 0:
        for _, row in high_precision.iterrows():
            print(f"    - {row['Method']}: P={row['Precision']:.3f}, R={row['Recall']:.3f}, F1={row['F1']:.3f}")

    # 유형별 Recall breakdown
    print("\n" + "=" * 70)
    print("  [5] Anomaly 유형별 Recall")
    print("=" * 70)
    type_df = evaluator.type_breakdown_table()
    print(tabulate(type_df, headers="keys", tablefmt="grid", floatfmt=".3f"))

    # ──────────────────────────────────────────────
    # [6] 시각화
    # ──────────────────────────────────────────────
    print("\n[6] 시각화 생성 중...")
    viz = BenchmarkVisualizer(output_dir="docs/benchmark_results")
    viz.plot_all(evaluator, dataset)

    # ──────────────────────────────────────────────
    # [7] 인사이트 요약
    # ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  [7] 핵심 인사이트")
    print("=" * 70)

    # 최고 F1 방법
    best = summary_df.iloc[0]
    print(f"\n  1) 최고 F1 Score: {best['Method']} (F1={best['F1']:.3f})")

    # 최고 Precision 방법
    best_p = summary_df.sort_values("Precision", ascending=False).iloc[0]
    print(f"  2) 최고 Precision: {best_p['Method']} (P={best_p['Precision']:.3f})")

    # 가장 빠른 방법 (F1 > 0.3 이상)
    fast_df = summary_df[summary_df["F1"] > 0.3]
    if len(fast_df) > 0:
        fastest = fast_df.sort_values("Time(s)").iloc[0]
        print(f"  3) 가장 빠른 유효 방법: {fastest['Method']} ({fastest['Time(s)']:.3f}s, F1={fastest['F1']:.3f})")

    # 난이도별 요약
    print(f"\n  4) 난이도별 평균 Recall:")
    for diff in ["easy", "medium", "hard"]:
        col = f"Recall_{diff}"
        if col in summary_df.columns:
            avg = summary_df[col].mean()
            print(f"     - {diff.capitalize()}: {avg:.3f}")

    # 배포 전략 권고
    print(f"\n  5) 배포 전략 권고:")
    print(f"     - 즉시 배포: 상위 통계검정 방법 (빠르고 해석 가능)")
    print(f"     - 후속 적용: Autoencoder (학습 시간 필요, 비선형 패턴 포착)")

    # 결과 저장
    summary_df.to_csv("docs/benchmark_results/summary_table.csv", index=False)
    type_df.to_csv("docs/benchmark_results/type_breakdown.csv", index=False)
    print(f"\n[완료] 모든 결과가 docs/benchmark_results/ 에 저장되었습니다.")


if __name__ == "__main__":
    main()
