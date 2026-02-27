"""
Phase 1 실데이터 검증: SECOM 반도체 제조 공정 데이터

UCI SECOM Dataset을 활용한 PCA + Hotelling T² 변경점 분석 검증.

데이터셋:
  - 출처: UCI Machine Learning Repository (McCann & Johnston, 2008)
  - 구성: 1,567 wafers × 590 sensor features
  - 라벨: Pass (-1) = 1,463, Fail (1) = 104

시나리오 매핑:
  - Ref Group  = Pass wafer (정상 공정 조건)
  - Comp Group = Fail wafer (불량 발생 조건, ECO 평가 조건에 대응)
  → "Ref와 Comp 사이에 유의차가 있는가?" + "어떤 feature가 기여했는가?"

실행: python demo_secom.py
"""

import numpy as np
import pandas as pd
import time
import os
from pathlib import Path

from src.pca_hotelling import PCAHotellingT2
from src.preprocessing import DataPreprocessor
from src.visualization import ChangePointVisualizer


def load_secom_data(data_dir: str = "data") -> tuple[np.ndarray, np.ndarray, list[str]]:
    """SECOM 데이터 로드"""
    data = pd.read_csv(f"{data_dir}/secom.data", sep=" ", header=None)
    labels = pd.read_csv(f"{data_dir}/secom_labels.data", sep=" ", header=None)

    feature_names = [f"Sensor_{i:03d}" for i in range(data.shape[1])]
    y = labels[0].values  # -1=Pass, 1=Fail

    return data.values, y, feature_names


def clean_features(
    data: np.ndarray,
    feature_names: list[str],
    nan_threshold: float = 0.3,
    var_threshold: float = 1e-10,
) -> tuple[np.ndarray, list[str]]:
    """
    Feature 정제:
    1. NaN 비율이 높은 feature 제거
    2. 분산이 0인(상수) feature 제거
    3. 남은 NaN은 column median으로 대체
    """
    n_samples, n_features = data.shape
    keep_mask = np.ones(n_features, dtype=bool)

    # NaN 비율 기준 제거
    nan_ratios = np.sum(np.isnan(data), axis=0) / n_samples
    keep_mask &= nan_ratios < nan_threshold
    n_removed_nan = np.sum(nan_ratios >= nan_threshold)

    # 분산 기준 제거 (NaN 무시하고 계산)
    for j in range(n_features):
        if keep_mask[j]:
            col = data[:, j]
            valid = col[~np.isnan(col)]
            if len(valid) < 10 or np.var(valid) < var_threshold:
                keep_mask[j] = False
    n_removed_var = n_features - np.sum(keep_mask) - n_removed_nan

    # 필터링
    data_clean = data[:, keep_mask]
    names_clean = [n for n, k in zip(feature_names, keep_mask) if k]

    # NaN 대체 (column median)
    for j in range(data_clean.shape[1]):
        col = data_clean[:, j]
        nan_mask = np.isnan(col)
        if np.any(nan_mask):
            median_val = np.nanmedian(col)
            data_clean[nan_mask, j] = median_val

    print(f"    Feature 정제 결과:")
    print(f"      원본: {n_features}")
    print(f"      NaN>{nan_threshold*100:.0f}% 제거: -{n_removed_nan}")
    print(f"      상수/저분산 제거: -{n_removed_var}")
    print(f"      최종: {data_clean.shape[1]}")

    return data_clean, names_clean


def main():
    print("=" * 70)
    print("  SECOM 실데이터 검증: PCA + Hotelling T² 변경점 분석")
    print("=" * 70)

    # =========================================================
    # 1. 데이터 로드
    # =========================================================
    print("\n[1] SECOM 데이터 로드...")

    if not os.path.exists("data/secom.data"):
        print("    데이터 파일이 없습니다. 다운로드 중...")
        import urllib.request
        os.makedirs("data", exist_ok=True)
        urllib.request.urlretrieve(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data",
            "data/secom.data",
        )
        urllib.request.urlretrieve(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data",
            "data/secom_labels.data",
        )
        print("    다운로드 완료.")

    data_raw, labels, feature_names = load_secom_data()
    n_pass = np.sum(labels == -1)
    n_fail = np.sum(labels == 1)
    print(f"    전체: {data_raw.shape[0]} wafers × {data_raw.shape[1]} features")
    print(f"    Pass: {n_pass}, Fail: {n_fail}")

    # =========================================================
    # 2. Feature 정제
    # =========================================================
    print("\n[2] Feature 정제...")
    data_clean, names_clean = clean_features(data_raw, feature_names)

    # =========================================================
    # 3. Ref / Comp 분리
    # =========================================================
    print("\n[3] Ref / Comp Group 분리...")

    pass_idx = np.where(labels == -1)[0]
    fail_idx = np.where(labels == 1)[0]

    # Pass 그룹을 Ref(학습)와 Comp_Pass(정상 대조군)로 분할
    rng = np.random.RandomState(42)
    rng.shuffle(pass_idx)
    n_ref = int(len(pass_idx) * 0.7)

    ref_idx = pass_idx[:n_ref]
    comp_pass_idx = pass_idx[n_ref:]

    ref_data = data_clean[ref_idx]
    comp_pass_data = data_clean[comp_pass_idx]
    comp_fail_data = data_clean[fail_idx]

    print(f"    Ref Group (Pass 70%): {ref_data.shape[0]} wafers")
    print(f"    Comp Group A (Pass 30%, 유의차 없음 기대): {comp_pass_data.shape[0]} wafers")
    print(f"    Comp Group B (Fail, 유의차 있음 기대): {comp_fail_data.shape[0]} wafers")

    # =========================================================
    # 4. 데이터 전처리 (Feature 유형 분류)
    # =========================================================
    print("\n[4] 데이터 전처리...")
    preprocessor = DataPreprocessor()
    preprocessor.analyze_features(ref_data, names_clean)
    summary = preprocessor.summary()

    print(f"    유형 분포: {summary['type_distribution']}")
    print(f"    PCA 대상 (Continuous): {summary['continuous_count']}")
    print(f"    별도 검정 (Discrete/Binary): {summary['discrete_count']}")

    ref_cont = preprocessor.get_continuous_features(ref_data)
    comp_pass_cont = preprocessor.get_continuous_features(comp_pass_data)
    comp_fail_cont = preprocessor.get_continuous_features(comp_fail_data)
    cont_names = preprocessor.get_continuous_feature_names()

    print(f"    PCA 입력: {ref_cont.shape[1]} features")

    # =========================================================
    # 5. PCA + Hotelling T² 모델 학습 및 분석
    # =========================================================
    print("\n[5] PCA + Hotelling T² 모델 학습...")

    model = PCAHotellingT2(n_components=0.95, alpha=0.01)

    t_start = time.time()
    model.fit(ref_cont)
    t_fit = time.time() - t_start

    info = model.get_model_info()
    print(f"    학습 시간: {t_fit:.3f}초")
    print(f"    주성분 수: {info['n_components']}")
    print(f"    설명 분산: {info['total_explained_variance']*100:.1f}%")

    # --- Comp A (Pass) 분석 ---
    print("\n[6] Comp Group A 분석 (Pass - 유의차 없음 기대)...")

    t_start = time.time()
    result_a = model.analyze(comp_pass_cont)
    t_infer_a = time.time() - t_start

    n_sig_t2_a = np.sum(result_a.is_significant_t2)
    n_sig_spe_a = np.sum(result_a.is_significant_spe)
    n_sig_any_a = np.sum(result_a.is_significant_t2 | result_a.is_significant_spe)
    n_a = len(comp_pass_cont)

    print(f"    추론 시간: {t_infer_a:.3f}초")
    print(f"    T² 유의차: {n_sig_t2_a}/{n_a} ({n_sig_t2_a/n_a*100:.1f}%)")
    print(f"    SPE 유의차: {n_sig_spe_a}/{n_a} ({n_sig_spe_a/n_a*100:.1f}%)")
    print(f"    T²∪SPE 유의차: {n_sig_any_a}/{n_a} ({n_sig_any_a/n_a*100:.1f}%)")

    # --- Comp B (Fail) 분석 ---
    print("\n[7] Comp Group B 분석 (Fail - 유의차 있음 기대)...")

    t_start = time.time()
    result_b = model.analyze(comp_fail_cont)
    t_infer_b = time.time() - t_start

    n_sig_t2_b = np.sum(result_b.is_significant_t2)
    n_sig_spe_b = np.sum(result_b.is_significant_spe)
    n_sig_any_b = np.sum(result_b.is_significant_t2 | result_b.is_significant_spe)
    n_b = len(comp_fail_cont)

    print(f"    추론 시간: {t_infer_b:.3f}초")
    print(f"    T² 유의차: {n_sig_t2_b}/{n_b} ({n_sig_t2_b/n_b*100:.1f}%)")
    print(f"    SPE 유의차: {n_sig_spe_b}/{n_b} ({n_sig_spe_b/n_b*100:.1f}%)")
    print(f"    T²∪SPE 유의차: {n_sig_any_b}/{n_b} ({n_sig_any_b/n_b*100:.1f}%)")

    # =========================================================
    # 6. 분리도 평가
    # =========================================================
    print("\n[8] Pass vs Fail 분리도 평가...")

    # T² 기준 분리도
    t2_a_mean = np.mean(result_a.t2_values)
    t2_b_mean = np.mean(result_b.t2_values)
    t2_a_std = np.std(result_a.t2_values)
    t2_b_std = np.std(result_b.t2_values)

    print(f"    T² 통계량:")
    print(f"      Comp A (Pass): mean={t2_a_mean:.2f}, std={t2_a_std:.2f}")
    print(f"      Comp B (Fail): mean={t2_b_mean:.2f}, std={t2_b_std:.2f}")
    print(f"      UCL: {result_a.t2_ucl:.2f}")
    if t2_a_std + t2_b_std > 0:
        fisher_ratio_t2 = abs(t2_b_mean - t2_a_mean) / (t2_a_std + t2_b_std)
        print(f"      Fisher Ratio: {fisher_ratio_t2:.3f}")

    # SPE 기준 분리도
    spe_a_mean = np.mean(result_a.spe_values)
    spe_b_mean = np.mean(result_b.spe_values)
    spe_a_std = np.std(result_a.spe_values)
    spe_b_std = np.std(result_b.spe_values)

    print(f"    SPE 통계량:")
    print(f"      Comp A (Pass): mean={spe_a_mean:.2f}, std={spe_a_std:.2f}")
    print(f"      Comp B (Fail): mean={spe_b_mean:.2f}, std={spe_b_std:.2f}")
    print(f"      UCL: {result_a.spe_ucl:.2f}")
    if spe_a_std + spe_b_std > 0:
        fisher_ratio_spe = abs(spe_b_mean - spe_a_mean) / (spe_a_std + spe_b_std)
        print(f"      Fisher Ratio: {fisher_ratio_spe:.3f}")

    # =========================================================
    # 7. Top 기여 Feature 분석
    # =========================================================
    print("\n[9] Top 기여 Feature 분석 (Fail group)...")

    top_k = 20
    top_indices = result_b.significant_features[:top_k]
    importance = result_b.feature_importance

    print(f"    Top-{top_k} 변경 기여 Feature:")
    print(f"    {'Rank':<5} {'Feature':<15} {'Contribution':<15}")
    print(f"    {'-'*35}")
    for rank, idx in enumerate(top_indices, 1):
        print(f"    {rank:<5} {cont_names[idx]:<15} {importance[idx]:.4f}")

    # Top-K 기여도 집중도
    total_imp = np.sum(importance)
    top10_imp = np.sum(importance[result_b.significant_features[:10]])
    top20_imp = np.sum(importance[result_b.significant_features[:20]])
    top50_imp = np.sum(importance[result_b.significant_features[:50]])

    print(f"\n    기여도 집중도:")
    print(f"      Top-10: {top10_imp/total_imp*100:.1f}%")
    print(f"      Top-20: {top20_imp/total_imp*100:.1f}%")
    print(f"      Top-50: {top50_imp/total_imp*100:.1f}%")

    # =========================================================
    # 8. 시각화
    # =========================================================
    print("\n[10] 시각화 생성...")

    viz = ChangePointVisualizer(output_dir="outputs")

    viz.plot_dashboard(
        result_a,
        feature_names=cont_names,
        title="SECOM - Comp A (Pass Group, No Change Expected)",
        save_name="secom_dashboard_pass.png",
    )
    print("    -> outputs/secom_dashboard_pass.png")

    viz.plot_dashboard(
        result_b,
        feature_names=cont_names,
        title="SECOM - Comp B (Fail Group, Change Expected)",
        save_name="secom_dashboard_fail.png",
    )
    print("    -> outputs/secom_dashboard_fail.png")

    viz.plot_error_rank_curve(
        result_a,
        feature_names=cont_names,
        title="SECOM - Error Rank Curve (Pass Group)",
        save_name="secom_rank_pass.png",
    )
    viz.plot_error_rank_curve(
        result_b,
        feature_names=cont_names,
        title="SECOM - Error Rank Curve (Fail Group)",
        save_name="secom_rank_fail.png",
    )
    print("    -> outputs/secom_rank_pass.png, secom_rank_fail.png")

    viz.plot_contribution(
        result_b,
        feature_names=cont_names,
        top_k=30,
        title="SECOM - Top-30 Feature Contributions (Fail Group)",
        save_name="secom_contribution_fail.png",
    )
    print("    -> outputs/secom_contribution_fail.png")

    # =========================================================
    # 9. 요약
    # =========================================================
    print("\n" + "=" * 70)
    print("  SECOM 실데이터 검증 결과 요약")
    print("=" * 70)
    print(f"""
    Dataset: UCI SECOM (1,567 wafers x 590 features)
    정제 후: {data_clean.shape[1]} features → PCA 대상: {ref_cont.shape[1]}
    PCA 주성분: {info['n_components']}개 (설명 분산 {info['total_explained_variance']*100:.1f}%)
    학습 시간: {t_fit:.3f}초 (CPU)

    ┌─────────────────────────────────────────────────────┐
    │ 유의차 판정 결과                                      │
    ├──────────┬──────────────────┬────────────────────────┤
    │          │ Comp A (Pass)    │ Comp B (Fail)          │
    │ T²       │ {n_sig_t2_a:>4}/{n_a:<4} ({n_sig_t2_a/n_a*100:>5.1f}%) │ {n_sig_t2_b:>4}/{n_b:<4} ({n_sig_t2_b/n_b*100:>5.1f}%)      │
    │ SPE      │ {n_sig_spe_a:>4}/{n_a:<4} ({n_sig_spe_a/n_a*100:>5.1f}%) │ {n_sig_spe_b:>4}/{n_b:<4} ({n_sig_spe_b/n_b*100:>5.1f}%)      │
    │ T²∪SPE   │ {n_sig_any_a:>4}/{n_a:<4} ({n_sig_any_a/n_a*100:>5.1f}%) │ {n_sig_any_b:>4}/{n_b:<4} ({n_sig_any_b/n_b*100:>5.1f}%)      │
    └──────────┴──────────────────┴────────────────────────┘

    기여도 집중도: Top-10={top10_imp/total_imp*100:.1f}%, Top-20={top20_imp/total_imp*100:.1f}%
    시각화: ./outputs/secom_*.png
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()
