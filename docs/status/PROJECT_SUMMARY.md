# 회전교차로 상호작용 예측 연구 프로젝트 요약

## 프로젝트 개요

**SDD Death Circle 데이터셋 기반 이기종 교통 상호작용 예측 연구**

비신호 회전교차로에서 차량, 보행자, 자전거 등 이기종 에이전트 간의 상호작용을 예측하기 위한 딥러닝 시스템을 구축합니다.

## 주요 구성 요소

### 1. 데이터 처리 파이프라인
- **호모그래피 변환**: 픽셀 좌표 → 미터 좌표 변환
- **이기종 에이전트 처리**: 차량, 보행자, 자전거 등 타입별 특징 추출
- **전처리**: 이상치 제거, 보간, 정규화

### 2. 씬 그래프 생성
- **이기종 그래프**: 에이전트 타입별 노드 및 관계 타입별 엣지
- **동적 그래프**: 시간에 따라 변화하는 그래프 구조
- **의미론적 관계**: yield, overtake, avoid, filter 등

### 3. 모델 아키텍처
- **HeteroGAT**: 이기종 Graph Attention Network
- **A3TGCN**: Attention Temporal Graph Convolutional Network
- **ST-HGNN**: Spatio-Temporal Heterogeneous GNN (통합 모델)

### 4. 학습 시스템
- **데이터 로더**: Index Batching 기반 효율적 로딩
- **손실 함수**: MSE, NLL, Safety-aware Loss
- **최적화**: Adam/AdamW, Learning Rate Scheduler, Early Stopping

### 5. 평가 시스템
- **정량 지표**: ADE, FDE, Miss Rate, Collision Rate
- **안전 지표**: TTC, PET, DRAC
- **시나리오 분석**: 타입별, 국가별 성능 비교

### 6. 안전 검증 레이어 (Plan B)
- **Safety Layer**: 모델 예측 결과의 안전성 검증
- **위험도 평가**: TTC 기반 위험도 점수 계산
- **후처리**: 위험한 예측 필터링 및 보정

### 7. 시각화
- **어텐션 가중치**: 히트맵, 맵 위 그래프 오버레이
- **양보 상황**: 특화 시각화
- **위험도 히트맵**: 공간 위험도 분포

## 프로젝트 구조

```
Roundabout_AI/
├── src/
│   ├── data_processing/      # 데이터 전처리
│   ├── scene_graph/          # 씬 그래프 생성
│   ├── models/               # 모델 아키텍처
│   │   ├── a3tgcn_model.py
│   │   ├── heterogeneous_gnn.py  # 이기종 GNN
│   │   └── trajectron_integration.py
│   ├── training/             # 학습 파이프라인
│   ├── evaluation/           # 평가 지표
│   ├── visualization/         # 시각화
│   └── integration/          # Plan A/B 통합
│       ├── hybrid_safety_layer.py
│       └── sdd_data_adapter.py
├── data/                     # 데이터셋
├── configs/                  # 설정 파일
├── scripts/                  # 유틸리티 스크립트
└── results/                  # 실험 결과
```

## 데이터셋 변경 사항

### 이전: INTERACTION Dataset
- 차량 중심
- 상대 좌표계
- Lanelet2 맵

### 현재: SDD Death Circle
- 이기종 에이전트 (6종)
- 픽셀 좌표계 → 호모그래피 변환 필요
- 맵 데이터 없음 (대안 방법 사용)

## 주요 기능

### 호모그래피 변환
- 대응점 기반 추정
- RANSAC 알고리즘
- 검증 및 오차 보정

### 이기종 그래프 처리
- 노드 타입별 특징 인코딩
- 관계 타입별 메시지 패싱
- HeteroConv 활용

### 안전 검증 통합
- 예측 결과 실시간 검증
- 위험도 기반 필터링
- 안전 라벨 생성

## 사용 방법

### 1. 환경 설정
```bash
./setup.sh
```

### 2. 데이터 준비
```bash
# SDD Death Circle 데이터 다운로드
# 호모그래피 행렬 추정
python scripts/auto_homography_estimation.py
```

### 3. 데이터 전처리
```python
from src.integration.sdd_data_adapter import SDDDataAdapter

adapter = SDDDataAdapter(homography_path='data/homography/H.txt')
converted_data = adapter.convert_pixel_to_meter(raw_data)
```

### 4. 모델 학습
```bash
python src/training/train.py --config configs/training_config.yaml
```

### 5. 안전 검증
```python
from src.integration import HybridPredictor, SafetyLayer

safety_layer = SafetyLayer()
predictor = HybridPredictor(model, safety_layer, device)
result = predictor.predict_with_safety_check(obs_data, return_risk=True)
```

## 다음 단계

1. 실제 SDD 데이터로 테스트
2. 호모그래피 행렬 정확도 개선
3. 이기종 모델 성능 튜닝
4. 최종 결과 문서화

