# Change Point Detection (변경점 분석 고도화)

반도체 EDS 데이터의 Ref Group vs Comp Group 간 유의차를 검출하고, 변경점이 발생한 Feature를 식별하는 프로젝트.

## Overview

DRAM/Flash 제품의 공정 개선 조건 적용 시, 기존 조건 대비 EDS Test 결과에 유의차가 존재하는 항목을 효율적으로 검출하기 위한 분석 도구.

### 방법론

| 방법론 | 역할 | 상태 |
|--------|------|------|
| **PCA + Hotelling T²** | Baseline (Phase 1) | PoC 완료 |
| AutoEncoder | 비선형 확장 (Phase 2) | 계획 |
| 하이브리드 앙상블 | 최종 통합 (Phase 3) | 계획 |

## Quick Start

```bash
# 의존성 설치
pip install -r requirements.txt

# Phase 1 데모 실행
python demo_phase1.py

# 테스트 실행
pytest tests/ -v
```

## Project Structure

```
change_point_detection/
├── src/
│   ├── pca_hotelling.py      # PCA + Hotelling T² 모델
│   ├── preprocessing.py      # 데이터 전처리 (Feature 유형 분류)
│   └── visualization.py      # 시각화 유틸리티
├── tests/
│   └── test_pca_hotelling.py # 단위 테스트
├── docs/
│   └── report_phase1.md      # Phase 1 보고서
├── demo_phase1.py            # 데모 스크립트
└── requirements.txt
```

## Usage

```python
from src import PCAHotellingT2, DataPreprocessor

# 1. 데이터 전처리
preprocessor = DataPreprocessor()
preprocessor.analyze_features(ref_data, feature_names)
ref_cont = preprocessor.get_continuous_features(ref_data)
comp_cont = preprocessor.get_continuous_features(comp_data)

# 2. 모델 학습 및 분석
model = PCAHotellingT2(n_components=0.95, alpha=0.01)
model.fit(ref_cont)
result = model.analyze(comp_cont)

# 3. 결과 확인
print(f"유의차 wafer: {result.is_significant_t2.sum()}")
print(f"T² UCL: {result.t2_ucl:.2f}")
print(f"Top 기여 feature: {result.significant_features[:10]}")
```

## References

- Hotelling, H. (1947). Multivariate Quality Control.
- Kourti, T. & MacGregor, J.F. (1995). Process Analysis, Monitoring and Diagnosis Using Multivariate Projection Methods.
- Wise, B.M. & Gallagher, N.B. (1996). The Process Chemometrics Approach to Process Monitoring and Fault Detection.
