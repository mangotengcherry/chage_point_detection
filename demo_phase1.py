"""
Phase 1 데모: PCA + Hotelling T² 기반 변경점 분석

반도체 EDS 데이터를 시뮬레이션하여 PCA+T² 방법론의 유효성을 검증한다.

시나리오:
  - Ref Group: 정상 공정 조건의 wafer 데이터
  - Comp Group A: Ref와 유의차 없는 조건 (정상)
  - Comp Group B: 특정 feature에 유의차 존재 (ECO 평가 조건)

실행: python demo_phase1.py
"""

import numpy as np
import time
from src.pca_hotelling import PCAHotellingT2
from src.preprocessing import DataPreprocessor
from src.visualization import ChangePointVisualizer


def generate_semiconductor_data(
    n_wafers: int = 200,
    n_features: int = 5000,
    n_binary: int = 200,
    n_discrete: int = 300,
    random_state: int = 42,
) -> tuple[np.ndarray, list[str]]:
    """
    반도체 EDS BIN 데이터 시뮬레이션 (Ref Group)

    다양한 분포 유형을 포함한 현실적인 데이터 생성:
    - Binary features (Pass/Fail BIN)
    - Discrete features (Fail bit count 등)
    - Continuous Normal features (전압/전류 등)
    - Continuous Skewed features (Leakage current 등)
    """
    rng = np.random.RandomState(random_state)

    feature_names = []
    columns = []

    # Binary features (Pass/Fail)
    for i in range(n_binary):
        col = rng.binomial(1, 0.95, n_wafers).astype(float)
        columns.append(col)
        feature_names.append(f"BIN_PF_{i:04d}")

    # Discrete features
    for i in range(n_discrete):
        col = rng.poisson(3, n_wafers).astype(float)
        columns.append(col)
        feature_names.append(f"BIN_CNT_{i:04d}")

    # Continuous Normal features
    n_normal = n_features - n_binary - n_discrete - (n_features // 10)
    for i in range(n_normal):
        mean = rng.uniform(50, 200)
        std = rng.uniform(2, 15)
        col = rng.normal(mean, std, n_wafers)
        columns.append(col)
        feature_names.append(f"EDS_MEAS_{i:04d}")

    # Continuous Skewed features (Leakage-like)
    n_skewed = n_features - len(columns)
    for i in range(n_skewed):
        col = rng.lognormal(mean=3, sigma=0.5, size=n_wafers)
        columns.append(col)
        feature_names.append(f"EDS_LEAK_{i:04d}")

    data = np.column_stack(columns)
    return data, feature_names


def generate_comp_group_no_change(
    ref_data: np.ndarray,
    n_wafers: int = 100,
    random_state: int = 123,
) -> np.ndarray:
    """Comp Group A: Ref와 유의차 없는 데이터 (자연 산포 내)"""
    rng = np.random.RandomState(random_state)
    means = np.mean(ref_data, axis=0)
    stds = np.std(ref_data, axis=0)

    # Ref와 동일한 분포에서 샘플링 (약간의 노이즈 추가)
    data = np.zeros((n_wafers, ref_data.shape[1]))
    for j in range(ref_data.shape[1]):
        data[:, j] = rng.normal(means[j], stds[j] * 1.02, n_wafers)

    return data


def generate_comp_group_with_change(
    ref_data: np.ndarray,
    n_wafers: int = 100,
    changed_features: list[int] | None = None,
    shift_magnitude: float = 3.0,
    random_state: int = 456,
) -> tuple[np.ndarray, list[int]]:
    """
    Comp Group B: 특정 feature에 유의차가 존재하는 데이터

    Parameters
    ----------
    changed_features : list[int]
        유의차를 주입할 feature 인덱스. None이면 랜덤 선택.
    shift_magnitude : float
        평균 이동량 (표준편차 배수)
    """
    rng = np.random.RandomState(random_state)
    means = np.mean(ref_data, axis=0)
    stds = np.std(ref_data, axis=0)

    # 기본은 Ref와 동일 분포
    data = np.zeros((n_wafers, ref_data.shape[1]))
    for j in range(ref_data.shape[1]):
        data[:, j] = rng.normal(means[j], stds[j], n_wafers)

    # 특정 feature에 변경점 주입
    if changed_features is None:
        n_change = 30  # 30개 feature에 유의차 주입
        # continuous feature 영역에서만 선택 (binary/discrete 제외)
        continuous_start = 500  # binary(200) + discrete(300) 이후
        changed_features = sorted(
            rng.choice(
                range(continuous_start, ref_data.shape[1]),
                size=n_change,
                replace=False,
            )
        )

    for idx in changed_features:
        # 평균 이동 (mean shift)
        data[:, idx] += stds[idx] * shift_magnitude
        # 일부는 분산 증가도 추가
        if rng.random() > 0.5:
            data[:, idx] += rng.normal(0, stds[idx] * 0.5, n_wafers)

    return data, changed_features


def evaluate_detection(
    result,
    true_changed: list[int],
    continuous_feature_indices: np.ndarray,
    feature_names: list[str],
    top_k: int = 50,
) -> dict:
    """
    검출 성능 평가 (Precision@K, Recall@K)

    Parameters
    ----------
    result : HotellingResult
    true_changed : list[int]
        실제 변경된 feature의 원본 인덱스
    continuous_feature_indices : np.ndarray
        continuous feature의 원본 인덱스 매핑
    """
    # 상위 K개 검출 feature (continuous feature 내 인덱스)
    detected_local = result.significant_features[:top_k]

    # 원본 인덱스로 변환
    detected_original = [int(continuous_feature_indices[i]) for i in detected_local]

    true_set = set(true_changed)
    detected_set = set(detected_original)

    tp = len(true_set & detected_set)
    precision = tp / len(detected_set) if detected_set else 0
    recall = tp / len(true_set) if true_set else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "top_k": top_k,
        "true_changed_count": len(true_changed),
        "detected_count": len(detected_set),
        "true_positives": tp,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "detected_features": detected_original[:10],  # 상위 10개만 표시
    }


def main():
    print("=" * 70)
    print("  Phase 1: PCA + Hotelling T² 변경점 분석 데모")
    print("=" * 70)

    # =========================================================
    # 1. 데이터 생성
    # =========================================================
    print("\n[1] 반도체 EDS 데이터 시뮬레이션...")

    N_FEATURES = 5000
    N_REF_WAFERS = 200
    N_COMP_WAFERS = 100

    ref_data, feature_names = generate_semiconductor_data(
        n_wafers=N_REF_WAFERS,
        n_features=N_FEATURES,
    )
    print(f"    Ref Group: {ref_data.shape[0]} wafers x {ref_data.shape[1]} features")

    comp_a_data = generate_comp_group_no_change(ref_data, n_wafers=N_COMP_WAFERS)
    print(f"    Comp Group A (유의차 없음): {comp_a_data.shape[0]} wafers")

    comp_b_data, true_changed = generate_comp_group_with_change(
        ref_data, n_wafers=N_COMP_WAFERS, shift_magnitude=3.0,
    )
    print(f"    Comp Group B (유의차 있음): {comp_b_data.shape[0]} wafers")
    print(f"    → 실제 변경 feature 수: {len(true_changed)}")

    # =========================================================
    # 2. 데이터 전처리 (Feature 유형 분류)
    # =========================================================
    print("\n[2] 데이터 전처리 (Feature 유형 분류)...")

    preprocessor = DataPreprocessor()
    feature_types = preprocessor.analyze_features(ref_data, feature_names)
    summary = preprocessor.summary()

    print(f"    총 Feature 수: {summary['total_features']}")
    print(f"    유형 분포: {summary['type_distribution']}")
    print(f"    PCA 대상 (Continuous): {summary['continuous_count']}")
    print(f"    별도 검정 (Discrete/Binary): {summary['discrete_count']}")

    # Continuous feature만 추출
    ref_cont = preprocessor.get_continuous_features(ref_data)
    comp_a_cont = preprocessor.get_continuous_features(comp_a_data)
    comp_b_cont = preprocessor.get_continuous_features(comp_b_data)
    cont_names = preprocessor.get_continuous_feature_names()

    # Continuous feature의 원본 인덱스 매핑
    cont_original_indices = np.where(preprocessor._continuous_mask)[0]

    print(f"    PCA 입력 차원: {ref_cont.shape[1]} features")

    # =========================================================
    # 3. PCA + Hotelling T² 모델 학습 및 분석
    # =========================================================
    print("\n[3] PCA + Hotelling T² 모델 학습...")

    model = PCAHotellingT2(n_components=0.95, alpha=0.01)

    t_start = time.time()
    model.fit(ref_cont)
    t_fit = time.time() - t_start

    info = model.get_model_info()
    print(f"    학습 시간: {t_fit:.3f}초")
    print(f"    주성분 수: {info['n_components']}")
    print(f"    설명 분산: {info['total_explained_variance']:.4f} ({info['total_explained_variance']*100:.1f}%)")

    # --- Comp Group A 분석 ---
    print("\n[4] Comp Group A 분석 (유의차 없음 기대)...")

    t_start = time.time()
    result_a = model.analyze(comp_a_cont)
    t_infer_a = time.time() - t_start

    n_sig_t2_a = np.sum(result_a.is_significant_t2)
    n_sig_spe_a = np.sum(result_a.is_significant_spe)
    print(f"    추론 시간: {t_infer_a:.3f}초")
    print(f"    T² 유의차 wafer: {n_sig_t2_a}/{N_COMP_WAFERS} ({n_sig_t2_a/N_COMP_WAFERS*100:.1f}%)")
    print(f"    SPE 유의차 wafer: {n_sig_spe_a}/{N_COMP_WAFERS} ({n_sig_spe_a/N_COMP_WAFERS*100:.1f}%)")

    # --- Comp Group B 분석 ---
    print("\n[5] Comp Group B 분석 (유의차 있음 기대)...")

    t_start = time.time()
    result_b = model.analyze(comp_b_cont)
    t_infer_b = time.time() - t_start

    n_sig_t2_b = np.sum(result_b.is_significant_t2)
    n_sig_spe_b = np.sum(result_b.is_significant_spe)
    print(f"    추론 시간: {t_infer_b:.3f}초")
    print(f"    T² 유의차 wafer: {n_sig_t2_b}/{N_COMP_WAFERS} ({n_sig_t2_b/N_COMP_WAFERS*100:.1f}%)")
    print(f"    SPE 유의차 wafer: {n_sig_spe_b}/{N_COMP_WAFERS} ({n_sig_spe_b/N_COMP_WAFERS*100:.1f}%)")

    # =========================================================
    # 4. 검출 성능 평가
    # =========================================================
    print("\n[6] 변경 Feature 검출 성능 평가 (Comp Group B)...")

    for k in [10, 20, 30, 50]:
        metrics = evaluate_detection(
            result_b, true_changed, cont_original_indices, feature_names, top_k=k,
        )
        print(f"    Precision@{k:2d}: {metrics['precision']:.3f}  |  "
              f"Recall@{k:2d}: {metrics['recall']:.3f}  |  "
              f"F1@{k:2d}: {metrics['f1_score']:.3f}  |  "
              f"TP: {metrics['true_positives']}/{metrics['true_changed_count']}")

    # =========================================================
    # 5. 시각화
    # =========================================================
    print("\n[7] 시각화 생성 중...")

    viz = ChangePointVisualizer(output_dir="outputs")

    # Comp A 대시보드
    viz.plot_dashboard(
        result_a,
        feature_names=cont_names,
        title="Comp Group A (No Change Expected) - Dashboard",
        save_name="dashboard_comp_a.png",
    )
    print("    → outputs/dashboard_comp_a.png 저장 완료")

    # Comp B 대시보드
    viz.plot_dashboard(
        result_b,
        feature_names=cont_names,
        title="Comp Group B (Change Expected) - Dashboard",
        save_name="dashboard_comp_b.png",
    )
    print("    → outputs/dashboard_comp_b.png 저장 완료")

    # Error Rank Curve 비교
    viz.plot_error_rank_curve(
        result_a,
        title="Error Rank Curve - Comp A (No Change)",
        save_name="rank_curve_comp_a.png",
    )
    viz.plot_error_rank_curve(
        result_b,
        title="Error Rank Curve - Comp B (With Change)",
        save_name="rank_curve_comp_b.png",
    )
    print("    → outputs/rank_curve_comp_a.png, rank_curve_comp_b.png 저장 완료")

    # =========================================================
    # 6. 요약
    # =========================================================
    print("\n" + "=" * 70)
    print("  Phase 1 데모 결과 요약")
    print("=" * 70)
    print(f"""
    데이터 규모: {N_FEATURES} features x {N_REF_WAFERS} ref wafers
    PCA 주성분: {info['n_components']}개 (설명 분산 {info['total_explained_variance']*100:.1f}%)
    학습 시간: {t_fit:.3f}초 (CPU only)
    추론 시간: {t_infer_b:.3f}초/group

    Comp A (정상):
      T² 유의차 wafer: {n_sig_t2_a}/{N_COMP_WAFERS} → 기대: 약 {int(N_COMP_WAFERS*0.01)}개 (alpha=0.01)
      → {'PASS' if n_sig_t2_a <= N_COMP_WAFERS * 0.05 else 'CHECK'}: 정상 그룹을 정상으로 잘 판정

    Comp B (변경):
      T² 유의차 wafer: {n_sig_t2_b}/{N_COMP_WAFERS}
      SPE 유의차 wafer: {n_sig_spe_b}/{N_COMP_WAFERS}
      → {'PASS' if n_sig_t2_b > N_COMP_WAFERS * 0.5 else 'CHECK'}: 변경 그룹을 변경으로 잘 탐지

    Feature 검출 (Top-30):
      변경된 {len(true_changed)}개 feature 중 Top-30에 포함된 수: 위 Precision@30 참조
    """)

    print("    시각화 결과: ./outputs/ 디렉토리를 확인하세요.")
    print("=" * 70)


if __name__ == "__main__":
    main()
