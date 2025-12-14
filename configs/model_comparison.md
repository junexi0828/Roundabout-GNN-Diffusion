# 모델 아키텍처 비교 분석

## 후보 모델 개요

본 연구에서는 회전교차로 상호작용 예측을 위해 다음 세 가지 모델을 비교 분석합니다:

1. **A3TGCN** (Attention Temporal Graph Convolutional Network)
2. **Trajectron++** (Dynamically-Feasible Trajectory Forecasting)
3. **Social-STGCNN** (Social Spatio-Temporal Graph Convolutional Neural Network)

## 상세 비교 분석

### 1. A3TGCN

#### 특징
- **라이브러리**: PyTorch Geometric Temporal
- **핵심 메커니즘**: Graph Convolution + GRU + Temporal Attention
- **장점**:
  - 어텐션 메커니즘으로 시간적 중요도 학습
  - PyTorch Geometric Temporal에 내장되어 구현 용이
  - 동적 그래프 처리 지원
- **단점**:
  - HD Map 정보 직접 활용 어려움
  - 차량 동역학 제약 반영 제한적

#### 아키텍처
```
Input: 시공간 그래프 시퀀스
  ↓
Graph Convolution (공간 정보 집계)
  ↓
GRU (시간 정보 갱신)
  ↓
Temporal Attention (중요 시점 강조)
  ↓
MLP Decoder
  ↓
Output: 미래 궤적 (x, y)
```

#### 회전교차로 적용성
- ✅ 상호작용 모델링: 우수 (어텐션으로 중요 차량 강조)
- ⚠️ 맵 제약 반영: 제한적 (추가 모듈 필요)
- ⚠️ 동역학 제약: 제한적

---

### 2. Trajectron++

#### 특징
- **라이브러리**: Stanford ASL (독립 구현체)
- **핵심 메커니즘**: CVAE + Dynamic Spatiotemporal Graph + HD Map Encoding
- **장점**:
  - HD Map 정보 직접 인코딩 (Lanelet2 지원)
  - 차량 동역학 모델 통합 (Dynamically-extended Unicycle)
  - 다중 모드 예측 (CVAE 기반)
  - trajdata 라이브러리와 통합 가능
- **단점**:
  - 구현 복잡도 높음
  - 학습 시간 길음
  - 메모리 사용량 큼

#### 아키텍처
```
Input: 궤적 히스토리 + HD Map (Lanelet2)
  ↓
Map Encoder (CNN)
  ↓
Temporal Encoder (LSTM)
  ↓
Spatiotemporal Graph (GNN)
  ↓
CVAE Encoder
  ↓
CVAE Decoder (동역학 모델 통합)
  ↓
Output: 다중 모드 궤적 예측 (가우스 분포)
```

#### 회전교차로 적용성
- ✅ 상호작용 모델링: 우수
- ✅ 맵 제약 반영: 우수 (Lanelet2 직접 활용)
- ✅ 동역학 제약: 우수 (물리적 타당성 보장)

---

### 3. Social-STGCNN

#### 특징
- **라이브러리**: 독립 구현체 (GitHub)
- **핵심 메커니즘**: Graph Convolution + Temporal CNN
- **장점**:
  - 매우 빠른 추론 속도 (0.002s/frame)
  - 적은 파라미터 수 (약 7.6K)
  - 적은 데이터로도 학습 가능
- **단점**:
  - 보행자 중심 설계 (차량에 부적합)
  - 맵 정보 미반영
  - 동역학 제약 없음 (물리적으로 불가능한 궤적 생성 가능)

#### 아키텍처
```
Input: 궤적 히스토리
  ↓
Graph Convolution (공간)
  ↓
Temporal CNN (시간)
  ↓
Output: 미래 궤적 (x, y)
```

#### 회전교차로 적용성
- ⚠️ 상호작용 모델링: 보통 (보행자용 설계)
- ❌ 맵 제약 반영: 없음
- ❌ 동역학 제약: 없음

---

## 종합 비교표

| 항목 | A3TGCN | Trajectron++ | Social-STGCNN |
|------|--------|--------------|---------------|
| **상호작용 모델링** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **맵 정보 활용** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ |
| **동역학 제약** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ |
| **구현 난이도** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| **학습 속도** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **추론 속도** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **회전교차로 적합성** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |

## 추천 모델 선정

### 1순위: **Trajectron++**
- **이유**: 회전교차로의 기하학적 제약과 차량 동역학을 모두 반영할 수 있음
- **활용 전략**: trajdata 라이브러리를 통한 데이터 로딩, HD Map 인코딩 활용

### 2순위: **A3TGCN**
- **이유**: 어텐션 메커니즘으로 상호작용 학습에 유리, 구현이 상대적으로 간단
- **활용 전략**: 맵 정보를 노드 특징으로 추가하여 보완

### 3순위: **Social-STGCNN**
- **이유**: 빠른 속도이지만 회전교차로 환경에 부적합
- **활용 전략**: Baseline 비교 모델로만 사용

## 최종 결정

본 연구에서는 **Trajectron++**를 주 모델로 채택하고, **A3TGCN**을 보조 모델로 구현합니다.

### 구현 전략
1. **Trajectron++**:
   - trajdata를 통한 INTERACTION Dataset 로딩
   - HD Map 인코딩 활성화
   - 동역학 모델 통합

2. **A3TGCN**:
   - PyTorch Geometric Temporal 활용
   - 맵 정보를 노드 특징으로 통합
   - 어텐션 가중치 시각화

3. **Social-STGCNN**:
   - Baseline 비교용으로만 구현
   - 성능 비교 지표 제공

