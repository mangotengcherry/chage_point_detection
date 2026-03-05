# EDS BIN Feature 유의차 검출 벤치마크

반도체 EDS BIN Item(BIN130~BIN629)에 대해 Ref Group vs Comp Group 간 **유의차가 있는 Feature를 검출**하고, 다양한 탐지 방법의 성능을 정량 비교하는 프로젝트.

---

## 1. 실험 배경

### 1.1 문제 정의

- EDS Item 400~500개 대상 Precision 0.8+ 변경점 분석 모델을 Dashboard로 배포 필요
- BIN 데이터는 대부분 0-skewed 비정규 분포 -> 비모수 검정이 유리
- **목표**: 어떤 Feature가 Ref/Comp 간 유의차를 보이는지 자동 검출

### 1.2 접근 전략

1. **통계검정 즉시 배포**: Mann-Whitney U, KS Test 등 비모수 검정으로 빠르게 서비스
2. **AE Dual-Path Pipeline**: Autoencoder + 통계검정 교차검증으로 FP 억제
3. **벤치마크를 통해 최적 방법론 선정**

---

## 2. 합성 데이터 설명

### 2.1 데이터 구조 (Wafer 기반)

| 항목 | 값 |
|------|-----|
| Ref Group | 1,000 wafers x 500 features |
| Comp Group | 100 wafers x 500 features |
| Feature 범위 | BIN130 ~ BIN629 |
| 정상 Feature | 350개 |
| Anomaly Feature | 150개 (5유형 x 3난이도 x 10개) |

### 2.2 BIN Baseline 유형 (3가지)

| 유형 | 비율 | 특성 |
|------|------|------|
| Zero-heavy | 60% | 90%+ 가 0, 희소한 비영 값 (exponential) |
| Low-rate | 30% | 평균 0.1~0.5% 불량률 |
| Moderate-rate | 10% | 평균 1~3% 만성 불량 |

### 2.3 Anomaly 유형 (5가지)

| 유형 | 설명 |
|------|------|
| Sporadic Spikes | 10~20% wafer에서 큰 스파이크 발생 |
| Level Shift | Comp 전체 wafer의 평균 상승 |
| Gradual Trend | Wafer 순서대로 점진적 증가 |
| Complex Trend | 급상승 -> 정상화 -> 재상승 (3단계) |
| Sudden Jump | 1~3 wafer만 스파이크 후 즉시 정상 |

### 2.4 난이도 (3단계)

| 난이도 | Spike 배수 | Shift 배수 | Trend 배수 |
|--------|-----------|-----------|-----------|
| Easy | 15~20x std | 6~8x std | 4~6x std |
| Medium | 8~12x std | 3~5x std | 2~3x std |
| Hard | 3~5x std | 1.5~2.5x std | 1~1.5x std |

---

## 3. 탐지 방법

### 3.1 통계검정 (4종)

| # | 방법 | 특성 |
|---|------|------|
| 1 | Mann-Whitney U | 비모수, rank-based, 0-skewed에 강건 |
| 2 | KS Test | 분포 형태 비교, 가장 빠름 |
| 3 | T-test | 모수적 평균 비교 기준선 |
| 4 | Welch's t-test | 이분산 허용 모수 검정 |

### 3.2 딥러닝 기반 (2종)

| # | 방법 | 특성 |
|---|------|------|
| 5 | Autoencoder | FC-AE reconstruction error, IQR threshold |
| 6 | **AE Dual-Path** | ECO 방법론: AE + 통계검정 교차검증 |

### 3.3 AE Dual-Path Pipeline (ECO 방법론)

```
[Step 1] AE 학습 (Train 800) -> Holdout 200 + Comp 100 에 대한 Feature별 Recon Error
[Step 2] AE Error 통계검정 (Holdout Ref vs Comp) -> 1차 후보
         Mann-Whitney U + KS Test + FDR Correction (BH)
[Step 3] Raw Feature 통계검정 (Ref vs Comp) -> 2차 후보
         Mann-Whitney U + KS Test + FDR Correction (BH)
[Step 4] 교집합 (Cross-Validation) -> 최종 유의 Feature
         AE Error 유의 AND Raw Feature 유의
```

- **교집합 전략**: AE Error와 Raw Feature 양쪽 모두에서 유의한 Feature만 최종 검출
- **FDR Correction**: Benjamini-Hochberg 방법으로 다중검정 보정
- **이중 검증**: AE의 비선형 패턴 탐지 + 통계검정의 해석 가능성을 결합

---

## 4. 실험 결과

### 4.1 전체 성능 요약

| Method | Precision | Recall | F1 | AUC | Time(s) |
|--------|-----------|--------|-----|-----|---------|
| **T-test** | **0.887** | **0.893** | **0.890** | 0.969 | 0.285 |
| **KS Test** | **0.974** | 0.753 | **0.850** | 0.903 | 0.196 |
| **Mann-Whitney U** | **0.915** | 0.787 | **0.846** | 0.890 | 0.251 |
| **Welch's t-test** | **0.823** | 0.807 | **0.815** | 0.929 | 0.251 |
| **AE Dual-Path** | **1.000** | 0.627 | 0.770 | 0.898 | 4.260 |
| **Autoencoder** | **0.972** | 0.460 | 0.624 | 0.929 | 5.054 |

> **Precision 0.8 이상 달성: 6개 방법 전체**

### 4.2 Performance Heatmap

![Performance Heatmap](docs/benchmark_results/performance_heatmap.png)

### 4.3 Method Comparison

![Method Comparison](docs/benchmark_results/method_comparison_bar.png)

### 4.4 ROC Curves

![ROC Curves](docs/benchmark_results/roc_curves.png)

### 4.5 Anomaly 유형별 Recall

| Method | level_shift | gradual_trend | complex_trend | sporadic_spikes | sudden_jump |
|--------|:---:|:---:|:---:|:---:|:---:|
| T-test | 1.000 | 1.000 | 1.000 | 1.000 | **0.467** |
| Mann-Whitney U | 1.000 | 1.000 | 1.000 | 0.867 | 0.067 |
| KS Test | 1.000 | 1.000 | 1.000 | 0.733 | 0.033 |
| Welch's t-test | 1.000 | 1.000 | 1.000 | 0.967 | 0.067 |
| AE Dual-Path | 1.000 | 0.867 | 0.933 | 0.333 | 0.000 |
| Autoencoder | 0.767 | 0.200 | 0.300 | 0.667 | 0.367 |

![Anomaly Type Breakdown](docs/benchmark_results/anomaly_type_breakdown.png)

### 4.6 난이도별 Recall

![Difficulty Breakdown](docs/benchmark_results/difficulty_breakdown.png)

---

## 5. 방법별 유의차 Feature 검출 결과

### 5.1 전체 방법 비교

각 방법이 검출한 Feature를 TP(정확 검출)/FP(오탐)/FN(미탐)/TN(정상)으로 분류하여 시각화한다.

![All Methods Feature Detection](docs/benchmark_results/all_methods_feature_detection.png)

### 5.2 방법 간 검출 일치도

Jaccard Similarity로 방법 간 검출 결과의 유사성을 비교한다.

![Detection Agreement](docs/benchmark_results/detection_agreement_heatmap.png)

---

## 6. AE Dual-Path Feature 검출 과정

### 6.1 파이프라인 단계별 결과

| 단계 | 검출 Feature 수 |
|------|:---:|
| Step 2: AE Error 유의 | 125개 |
| Step 3: Raw Feature 유의 | 102개 |
| Step 4: 교집합 (최종) | **94개** |
| AE만 유의 | 31개 |
| Raw만 유의 | 8개 |

### 6.2 Ground Truth 대비 정확도

| 지표 | 값 |
|------|:---:|
| True Positive (정확 검출) | 94 |
| False Positive (오탐) | **0** |
| False Negative (미탐) | 56 |
| **Precision** | **1.000** |
| Recall | 0.627 |

> **FP = 0**: 교집합 전략으로 False Positive를 완전 제거

### 6.3 Dual-Path Pipeline 시각화

![Dual-Path Summary](docs/benchmark_results/dual_path_summary.png)

- **좌**: AE Error 차이 (Comp - Holdout) - 유의 Feature 표시 (빨간색)
- **중**: Raw Feature KS Statistic - 유의 Feature 표시 (빨간색)
- **우**: 교집합 분류 (AE만 / 교집합 / Raw만)

### 6.4 Feature별 유의차 검출 결과

![Feature Significance](docs/benchmark_results/feature_significance.png)

- **빨간점**: 교집합 (최종 유의) - 94개
- **주황점**: AE만 유의 - 31개
- **보라점**: Raw만 유의 - 8개
- **회색점**: 정상 - 367개
- **검정 원**: Ground Truth Anomaly 위치

### 6.5 Scatter 차트 (유형별 Ref vs Comp)

| Sporadic Spikes | Level Shift |
|:---:|:---:|
| ![Spikes](docs/benchmark_results/scatter_sporadic_spikes.png) | ![Shift](docs/benchmark_results/scatter_level_shift.png) |

| Gradual Trend | Complex Trend |
|:---:|:---:|
| ![Trend](docs/benchmark_results/scatter_gradual_trend.png) | ![Complex](docs/benchmark_results/scatter_complex_trend.png) |

| Sudden Jump |
|:---:|
| ![Jump](docs/benchmark_results/scatter_sudden_jump.png) |

---

## 7. 계산비용 비교 분석

### 7.1 방법별 계산비용

| Method | Total(s) | Per Feature(ms) | F1 | Precision | Est.1000feat(s) | Est.5000feat(s) | Complexity |
|--------|----------|-----------------|-----|-----------|-----------------|-----------------|------------|
| KS Test | 0.196 | 0.39 | 0.850 | 0.974 | 0.39 | 1.96 | O(n_wafers * n_features) |
| Mann-Whitney U | 0.251 | 0.50 | 0.846 | 0.915 | 0.50 | 2.51 | O(n_wafers * n_features) |
| Welch's t-test | 0.251 | 0.50 | 0.815 | 0.823 | 0.50 | 2.51 | O(n_wafers * n_features) |
| T-test | 0.285 | 0.57 | 0.890 | 0.887 | 0.57 | 2.85 | O(n_wafers * n_features) |
| AE Dual-Path | 4.260 | 8.52 | 0.770 | 1.000 | 11.08 | 85.21 | O(n_wafers * n_features * epochs) |
| Autoencoder | 5.054 | 10.11 | 0.624 | 0.972 | 13.14 | 101.08 | O(n_wafers * n_features * epochs) |

### 7.2 효율성 지표 (F1/sec)

| Method | F1/sec | 평가 |
|--------|--------|------|
| KS Test | 4.34 | 가장 효율적 |
| Mann-Whitney U | 3.37 | 높은 효율 |
| Welch's t-test | 3.25 | 높은 효율 |
| T-test | 3.12 | 높은 효율 |
| AE Dual-Path | 0.18 | 배치 분석용 |
| Autoencoder | 0.12 | 배치 분석용 |

### 7.3 계산비용 vs 성능 Trade-off

![Cost Analysis](docs/benchmark_results/cost_analysis.png)

### 7.4 실행 시간 비교

![Execution Time](docs/benchmark_results/execution_time.png)

### 7.5 계산비용 요약

| 구분 | 통계검정 (4종) | AE 기반 (2종) |
|------|--------------|--------------|
| **평균 처리 시간** | 0.246s | 4.657s |
| **Feature당 처리 시간** | 0.39~0.57ms | 8.52~10.11ms |
| **5000 Feature 추정** | 1.96~2.85s | 85~101s |
| **속도 비율** | 1x (기준) | ~19x 느림 |
| **서비스 적합성** | 실시간 Dashboard | 배치 분석 |
| **확장성** | 선형 증가 | 비선형 증가 |

---

## 8. 핵심 인사이트

| # | 인사이트 | 상세 |
|---|---------|------|
| 1 | **통계검정이 F1 최고** | T-test (F1=0.890) > KS (0.850) > MW-U (0.846), 즉시 배포 가능 |
| 2 | **AE Dual-Path가 Precision 최고** | P=1.000 (FP=0), 교집합 전략으로 오탐 완전 제거 |
| 3 | **KS Test가 가장 효율적** | 0.196초로 가장 빠르면서 P=0.974, 효율성(F1/sec) 최고 |
| 4 | **Sudden Jump이 가장 어려운 유형** | 1~3 wafer만 변화하므로 분포 비교 방법으로는 탐지 어려움 |
| 5 | **통계검정은 난이도에 robust** | Easy~Hard 간 Recall 차이가 작아 안정적 |
| 6 | **AE는 mean shift 계열에 강함** | Level Shift(1.000), Complex Trend(0.933) 잘 탐지 |
| 7 | **교집합 전략 = FP 제거** | AE + Raw 양쪽 유의한 것만 취해 신뢰도 극대화 |
| 8 | **통계검정은 AE 대비 ~19배 빠름** | 실시간 Dashboard 배포에 적합, AE는 배치 분석에 적합 |

---

## 9. 배포 전략 권고

```
[즉시 배포]
+-- Primary: T-test (F1=0.890, P=0.887, R=0.893)
|   -> 가장 균형잡힌 P/R, 0.285초로 빠름
+-- Secondary: KS Test (P=0.974, 가장 높은 정밀도)
|   -> Precision 우선 시 사용, 가장 빠름 (0.196초)
+-- Precision 극대화: AE Dual-Path (P=1.000, FP=0)
    -> 오탐 허용 불가 시 사용 (Recall은 0.627)

[후속 개선]
+-- AE Dual-Path Recall 향상 (현재 0.627 -> 목표 0.8+)
|   -> AE 아키텍처 튜닝, FDR alpha 조정
+-- 앙상블: T-test + KS + Dual-Path 결합 -> 최적 P/R 밸런스
+-- Sudden Jump 탐지 보완 -> 이상치 탐지 방법 추가 검토
```

---

## 10. 실행 방법

```bash
# 의존성 설치
pip install -r requirements.txt

# 벤치마크 실행 (6가지 방법, ~15초)
python run_benchmark.py

# 테스트 (33/33 통과)
pytest tests/ -v
```

---

## 11. Project Structure

```
chage_point_detection/
+-- src/
|   +-- data_generation.py          # 합성 BIN 데이터 생성기 (Wafer 기반)
|   +-- dual_path_pipeline.py       # AE Dual-Path Pipeline (ECO 방법론)
|   +-- evaluation.py               # 벤치마크 평가 프레임워크
|   +-- benchmark_visualization.py  # 벤치마크 시각화 (Scatter + Feature 검출 + 계산비용)
|   +-- preprocessing.py            # 데이터 전처리
|   +-- visualization.py            # 기본 시각화
|   +-- detectors/                  # 변경점 탐지 방법 패키지
|       +-- base.py                 #   추상 베이스 클래스 (Wafer 기반)
|       +-- statistical.py          #   Mann-Whitney, KS, T-test, Welch
|       +-- autoencoder.py          #   FC-AE 탐지기
+-- tests/
|   +-- test_data_generation.py     # 데이터 생성 테스트
|   +-- test_detectors.py           # 탐지기 테스트
+-- docs/
|   +-- benchmark_results/          # 벤치마크 결과 이미지/CSV
+-- run_benchmark.py                # 벤치마크 실행 스크립트
+-- requirements.txt
+-- README.md
```

---

## 12. Usage

```python
from src.data_generation import BINDataGenerator
from src.dual_path_pipeline import DualPathPipeline

# 1. 데이터 생성 (또는 실제 데이터 로드)
gen = BINDataGenerator(n_ref=1000, n_comp=100, n_features=500)
dataset = gen.generate()

# 2. AE Dual-Path Pipeline 실행
pipeline = DualPathPipeline(
    hidden_dims=[256, 128, 64],
    epochs=100,
    alpha=0.05,
    fdr_method="fdr_bh",
)
result = pipeline.run(dataset.ref_data, dataset.comp_data, dataset.feature_names)

# 3. 결과 확인
print(f"AE Error 유의: {result.n_ae_significant}개")
print(f"Raw Feature 유의: {result.n_raw_significant}개")
print(f"교집합 (최종): {result.n_intersection}개")

# 4. 유의 Feature 목록
import numpy as np
sig_indices = np.where(result.intersection)[0]
for idx in sig_indices:
    print(f"  {dataset.feature_names[idx]}: KS={result.raw_ks_statistics[idx]:.4f}")
```

---

## 13. References

1. Hotelling, H. (1947). "Multivariate Quality Control." *Techniques of Statistical Analysis*, McGraw-Hill.
2. Jackson, J.E. & Mudholkar, G.S. (1979). "Control Procedures for Residuals Associated with PCA." *Technometrics*, 21(3).
3. Benjamini, Y. & Hochberg, Y. (1995). "Controlling the False Discovery Rate." *JRSS Series B*, 57(1), 289-300.
4. Mann, H.B. & Whitney, D.R. (1947). "On a Test of Whether One of Two Random Variables is Stochastically Larger." *Annals of Mathematical Statistics*, 18(1), 50-60.
