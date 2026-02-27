# 변경점 분석 고도화 - Phase 1 보고서

## PCA + Hotelling T² Baseline 방법론 검토 및 PoC

---

**작성일**: 2026-02-28
**담당**: 김HY, 하YJ
**문서 상태**: Phase 1 PoC 결과 보고

---

## 1. 배경 및 문제 정의

### 1.1 현재 과제의 핵심 문제

| 구분 | 내용 |
|------|------|
| 현상 | 개선 조건 적용 후 EDS Test 약 10,000개 항목 전수 검증 불가 |
| 현행 방법 | 500개 내외 대표 항목에 대해 통계분석 → 유의차 검출 → 엔지니어 리뷰 |
| 문제점 1 | 검출된 유의차 항목 중 20~30%가 가성(False Positive) → 리뷰 공수 낭비 |
| 문제점 2 | Feature 수 증가에 따라 분석 소요 시간이 선형 증가 |
| 목표 | 10월까지 정합성(Precision) 80% 이상 달성 |

### 1.2 접근 전략

기존에 AutoEncoder 기반 방법론을 1차로 검토하였으나, 아래 구조적 문제가 확인됨:

| # | 문제 | 상세 |
|---|------|------|
| 1 | RMSE 희석 | Feature 수 증가 시 1/N factor로 유의차 신호가 희석됨 |
| 2 | 보상 복원 | AE가 feature 간 상관관계를 학습하여 변경된 feature도 정상처럼 복원 |
| 3 | 임계값 근거 부족 | Elbow point 방식은 재현성이 낮고 통계적 근거 부재 |
| 4 | 데이터 이질성 | Binary/Discrete/Skewed feature를 하나의 모델로 처리 시 학습 불안정 |

이에 **PCA + Hotelling T²를 Baseline 방법론으로 먼저 검증**하고, AE와 정량 비교하는 전략을 수립함.

---

## 2. PCA + Hotelling T² 방법론

### 2.1 개요

다변량 통계적 공정 관리(MSPC)의 대표 방법론으로, 반도체 공정 모니터링에서 수십 년간 검증된 기법.

**핵심 통계량:**

- **T² (Hotelling T²)**: PCA 주성분 공간 내에서의 이상도
  ```
  T² = Σ(i=1 to k) (t_i² / λ_i)
  ```
- **SPE (Squared Prediction Error)**: PCA가 설명하지 못하는 잔차의 크기
  ```
  SPE = ||x - x̂||²
  ```
- **Contribution Plot**: T²/SPE에 대한 각 Feature의 기여도를 수학적으로 분해

### 2.2 AE 대비 구조적 장점

| 관점 | AE 방식 | PCA + T² 방식 |
|------|---------|--------------|
| 임계값 | Elbow point (경험적) | **F-분포 기반 (통계적 근거)** |
| 희석 문제 | RMSE 사용 시 발생 | **역공분산 가중으로 회피** |
| 보상 복원 | Feature 간 상관관계로 발생 | **복원 과정 없음** |
| 해석력 | 블랙박스 | **Loading/Contribution 수학적 분해** |
| 학습 시간 | 수 분~수십 분 (GPU) | **수 초 (CPU)** |
| 재현성 | 실행마다 상이 | **100% 동일 결과** |

### 2.3 T²가 RMSE 희석을 회피하는 원리

```
RMSE = sqrt(1/N × Σ error_i²)   → 모든 feature에 동일 가중치 (1/N)
T²   = (x-μ)ᵀ S⁻¹ (x-μ)        → 역공분산(S⁻¹)으로 차등 가중

→ 분산이 작은 feature에서의 작은 변화도 S⁻¹에 의해 크게 증폭
→ 소수의 변경 feature 신호가 다수의 정상 feature에 의해 희석되지 않음
```

### 2.4 참고 문헌

1. Hotelling, H. (1947). "Multivariate Quality Control." *Techniques of Statistical Analysis*, McGraw-Hill.
2. Kourti, T. & MacGregor, J.F. (1995). "Process Analysis, Monitoring and Diagnosis Using Multivariate Projection Methods." *Chemometrics and Intelligent Laboratory Systems*, 28(1), 3-21.
3. Wise, B.M. & Gallagher, N.B. (1996). "The Process Chemometrics Approach to Process Monitoring and Fault Detection." *Journal of Process Control*, 6(6), 329-348.
4. Jackson, J.E. & Mudholkar, G.S. (1979). "Control Procedures for Residuals Associated with Principal Component Analysis." *Technometrics*, 21(3), 341-349.

---

## 3. 계산 효율성 비교

### 3.1 Big O 비교 (P = Feature 수, N = Wafer 수)

| 단계 | 기존 통계분석 | AutoEncoder | PCA + T² |
|------|-------------|-------------|----------|
| 학습 | - | O(E×N×P×h) | **O(N×P×k)** |
| 추론 | O(P) 개별 검정 | O(P×h) | **O(P×k)** |
| 기여도 | - | O(P) | **O(P×k)** |

※ E=epoch(~100), h=hidden size(~256), k=주성분 수(~30)

### 3.2 실측 소요 시간 (5,000 features × 200 wafers 기준)

| 항목 | PCA + T² | AE (예상) |
|------|----------|----------|
| 학습 | < 1초 | 5~15분 |
| 추론 (100 wafers) | < 0.1초 | 1~5초 |
| GPU 필요 | **불필요** | 필요 |
| 메모리 | ~1 MB | ~40 MB+ |

### 3.3 자원 활용

| 항목 | PCA + T² | AutoEncoder |
|------|----------|-------------|
| 모델 파일 크기 | ~1 MB | ~40 MB |
| 학습 환경 | CPU (scikit-learn) | GPU (PyTorch/TF) |
| 배포 복잡도 | pickle 1개 | 모델 + 전처리 + 런타임 |
| 하이퍼파라미터 | 1개 (주성분 수) | 6개+ |

---

## 4. Phase 1 PoC 구현

### 4.1 구현 범위

```
change_point_detection/
├── src/
│   ├── pca_hotelling.py      # PCA + Hotelling T² 핵심 모델
│   ├── preprocessing.py      # 데이터 전처리 (Feature 유형 분류)
│   └── visualization.py      # 결과 시각화
├── tests/
│   └── test_pca_hotelling.py # 단위 테스트
├── demo_phase1.py            # Phase 1 데모 (합성 데이터)
└── docs/
    └── report_phase1.md      # 본 보고서
```

### 4.2 핵심 기능

1. **Feature 유형 자동 분류**: Binary/Discrete/Continuous/Skewed 자동 식별
   - Binary/Discrete → 별도 통계검정 (Chi-squared) 대상으로 분리
   - Continuous → PCA + T² 분석 대상

2. **PCA + Hotelling T² 분석**:
   - Ref Group으로 PCA 모델 학습
   - Comp Group에 대해 T², SPE 통계량 산출
   - F-분포 기반 UCL(Upper Control Limit) 산출
   - Feature별 Contribution 분해

3. **시각화**:
   - T² / SPE 관리도 (Control Chart)
   - Feature Contribution Plot (기여도 차트)
   - Error Rank Curve (AE와 동일 형태 비교용)
   - 4-panel 종합 대시보드

### 4.3 데모 시나리오

| 그룹 | 설명 | 기대 결과 |
|------|------|-----------|
| Ref | 정상 공정 조건 200 wafers | 모델 학습 대상 |
| Comp A | Ref와 동일 조건 100 wafers | 유의차 없음 판정 |
| Comp B | 30개 feature에 mean shift 주입 | 유의차 있음 판정 + 해당 feature 검출 |

---

## 5. 리스크 진단 및 대응

### 5.1 현재 과제의 맹점

| # | 맹점 | 대응 방안 |
|---|------|-----------|
| 1 | "정합성 80%"의 metric 미정의 | Precision@K 채택 권장 (K=30, Recall≥60% 조건) |
| 2 | Ground Truth 부족 | 리뷰 회의체 이력 30건+ 수집, 교차 검증 |
| 3 | AE의 RMSE 희석 문제 | PCA+T²는 역공분산 가중으로 구조적 회피 |
| 4 | AE의 보상 복원 문제 | PCA+T²는 복원 과정 없어 미발생 |
| 5 | 데이터 이질성 | Feature 유형 분류 후 분리 처리 |
| 6 | Elbow point 재현성 | F-분포 기반 UCL로 대체 |

### 5.2 PCA + T² 방법론의 한계

| 한계 | 영향도 | 보완 방안 |
|------|--------|-----------|
| 선형 관계만 포착 | 중간 | Kernel PCA 또는 AE 보완 |
| N < P 시 공분산 행렬 불안정 | 낮음 (PCA로 해소) | Truncated SVD 사용 |
| 비선형 불량 패턴 미포착 | 확인 필요 | 실데이터 검증 후 판단 |

---

## 6. 향후 계획

### Phase 0: 기반 정립 (3월)
- [ ] 정합성 metric 합의
- [ ] Ground Truth 데이터셋 구축 (30건+)
- [ ] 기존 통계분석 방식의 Precision/Recall 측정

### Phase 1: Baseline 검증 (4-5월)
- [x] PCA + T² PoC 구현 (본 보고)
- [ ] 실제 EDS 데이터 적용
- [ ] AE 모델과 정량 비교

### Phase 2: 최적화 (6-7월)
- [ ] Feature-wise 독립 AE 또는 Masked AE 실험
- [ ] 데이터 전처리 파이프라인 고도화
- [ ] ㄴ자 형태 최적화

### Phase 3: Mass 검증 (8-9월)
- [ ] 30건+ ECO case 검증
- [ ] EDS BIN 외 Measure 항목 확장
- [ ] In-Fab 센서/계측 데이터 PoC

### Phase 4: 안정화 (10월)
- [ ] 최종 성능 리포트
- [ ] 엔지니어 사용성 테스트
- [ ] 결과 보고

### Fallback 기준
- 6월 말: Precision < 60% → PCA 중심 앙상블로 피봇
- 8월 말: Precision < 70% → 대상 scope 축소 후 점진 확대

---

## 7. 결론 및 건의

1. **PCA + T²를 Baseline으로 확립**하여 AE 성능 비교의 anchor로 사용할 것을 건의
2. AE가 PCA+T²를 유의미하게 상회하는 경우에만 AE 도입 명분이 성립
3. **두 방법론의 장점을 결합한 하이브리드 앙상블**이 최적일 가능성이 높음:
   - Continuous feature → PCA + T² (빠르고 안정적)
   - 비선형 패턴 확인된 subset → AE 보완
   - Binary/Discrete feature → 개별 통계검정
4. "정합성 80%" metric의 명확한 정의가 가장 시급한 과제
