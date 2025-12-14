# 연구 방향 분석: MID와 현재 프로젝트 비교

## 🔍 MID (Motion Indeterminacy Diffusion) 개요

[MID GitHub](https://github.com/Gutianpei/MID.git)는 CVPR 2022 논문 "Stochastic Trajectory Prediction via Motion Indeterminacy Diffusion"의 구현입니다.

### MID의 핵심 아이디어

1. **Motion Indeterminacy Diffusion**:

   - 인간 행동의 불확정성(indeterminacy)을 명시적으로 모델링
   - 모든 가능한 보행 영역에서 시작하여 점진적으로 불확정성을 제거
   - 원하는 궤적에 도달할 때까지 역과정(Reverse Process) 수행

2. **다중 모달리티 처리**:

   - 기존 방법: 잠재 변수(latent variable)로 다중 모달리티 표현
   - MID: Diffusion 과정으로 명시적 다중 모달리티 생성

3. **확률적 예측**:
   - 확률 분포로 미래 궤적 생성
   - 다양성(diversity)과 확정성(determinacy)의 균형 조절

## 📊 현재 프로젝트 vs MID 비교

### 현재 프로젝트 (GNN 기반)

```
입력: 관측 궤적 + 씬 그래프
  ↓
GNN 인코딩 (A3TGCN/HeteroGAT)
  ↓
디코더 (MLP)
  ↓
출력: 단일 궤적 예측 (또는 가우시안 분포)
```

**특징**:

- ✅ 이기종 에이전트 처리 (Heterogeneous GNN)
- ✅ 씬 그래프 기반 상호작용 모델링
- ✅ Plan B 안전 검증 레이어
- ⚠️ 다중 모달리티 제한적 (가우시안 분포만)

### MID (Diffusion 기반)

```
입력: 관측 궤적
  ↓
Diffusion Process (100 steps → 2 steps with DDIM)
  ↓
출력: 다중 궤적 샘플링 (확률적)
```

**특징**:

- ✅ 강력한 다중 모달리티 생성 능력
- ✅ 확률적 예측 (다양한 미래 시나리오)
- ✅ DDIM으로 빠른 샘플링 (50x 가속)
- ⚠️ 이기종 에이전트 처리 미흡
- ⚠️ 씬 그래프 통합 없음

## 🎯 연구 방향: 하이브리드 접근

### 방향 1: GNN + Diffusion 통합 (권장)

```
입력: 관측 궤적 + 씬 그래프
  ↓
GNN 인코딩 (상호작용 모델링)
  ↓
Diffusion Process (다중 모달리티 생성)
  ↓
출력: 다중 궤적 샘플링
```

**장점**:

- GNN의 상호작용 모델링 + Diffusion의 다중 모달리티
- 이기종 에이전트 처리 가능
- 확률적 예측으로 다양한 미래 시나리오 생성

### 방향 2: MID 기반 이기종 확장

```
MID 아키텍처
  ↓
이기종 에이전트 타입 인코딩 추가
  ↓
씬 그래프 조건부 Diffusion
  ↓
출력: 이기종 에이전트별 다중 궤적
```

## 🚀 구체적 통합 전략

### 전략 A: GNN-Diffusion 하이브리드

1. **GNN Encoder**: 씬 그래프에서 상호작용 특징 추출
2. **Diffusion Decoder**: MID 기반 다중 궤적 생성
3. **Conditional Diffusion**: GNN 특징을 조건으로 사용

### 전략 B: MID 기반 이기종 확장

1. **Heterogeneous Encoder**: 에이전트 타입별 인코딩
2. **MID Diffusion**: 타입별 조건부 Diffusion
3. **Safety-Guided Sampling**: Plan B 안전 지표로 샘플링 가이드

## 📈 연구 기여도

### 현재 프로젝트의 강점

- ✅ 이기종 교통 환경 (차량, 보행자, 자전거)
- ✅ 회전교차로 특화 상호작용 모델링
- ✅ 안전 검증 레이어 (Plan B)

### MID 통합 시 추가 기여

- ✅ 확률적 다중 모달리티 예측
- ✅ 다양한 미래 시나리오 생성
- ✅ 불확실성 정량화

## 🔬 연구 질문

1. **이기종 환경에서 Diffusion이 어떻게 다중 모달리티를 생성하는가?**

   - 차량: 차선 준수, 물리적 제약
   - 보행자: 자유로운 이동
   - 자전거: 중간 특성

2. **씬 그래프 조건부 Diffusion의 효과는?**

   - 상호작용 정보가 Diffusion 과정에 미치는 영향

3. **안전 지표 기반 샘플링의 효과는?**
   - Plan B 안전 지표로 위험한 궤적 필터링

## 💡 구현 제안

### Phase 1: MID 기반 모델 추가

- MID 아키텍처 구현
- 이기종 에이전트 타입 조건 추가
- 씬 그래프 특징 통합

### Phase 2: 하이브리드 모델

- GNN Encoder + Diffusion Decoder
- 조건부 Diffusion 구현
- 다중 샘플링 및 평가

### Phase 3: 안전 가이드 샘플링

- Plan B 안전 지표로 샘플링 가이드
- 위험한 궤적 필터링
- 안전한 다중 궤적 생성

## 📚 참고 논문

- **MID**: "Stochastic Trajectory Prediction via Motion Indeterminacy Diffusion" (CVPR 2022)
- **DDIM**: "Denoising Diffusion Implicit Models" (ICLR 2021)
- **Trajectron++**: "Trajectron++: Dynamically-Feasible Trajectory Forecasting With Heterogeneous Data" (ECCV 2020)

## 🎯 최종 연구 방향

**목표**: 이기종 교통 환경에서 확률적 다중 모달리티 궤적 예측

**방법**:

1. GNN 기반 상호작용 모델링 (현재)
2. Diffusion 기반 다중 궤적 생성 (MID 통합)
3. 안전 지표 기반 샘플링 (Plan B)

**기여도**:

- 이기종 환경에서의 Diffusion 모델 확장
- 씬 그래프 조건부 Diffusion
- 안전 가이드 샘플링

---

## 📊 학술 발표 전략 (Academic Presentation Strategy)

### ❌ 잘못된 접근

```
우리 연구 = LED vs MID 성능 비교
→ 기여도: 낮음 (단순 재현 실험)
→ Novelty: 없음
```

### ✅ 올바른 접근

```
우리 연구 = GNN-Diffusion 하이브리드 (이기종 환경)
비교 대상 = 기존 베이스라인들
→ 기여도: 높음 (새로운 접근)
→ Novelty: 명확
```

---

## 🎯 최종 연구 방향 (학술 발표용)

### 연구 제목 (예시)

> "Heterogeneous Scene Graph-Conditioned Diffusion for Multi-Agent Trajectory Prediction in Roundabouts"

### 핵심 기여

1. ✅ **이기종 환경에서 Diffusion 확장**
2. ✅ **씬 그래프 조건부 생성**
3. ✅ **안전 가이드 샘플링**
4. ✅ **회전교차로 특화**

---

## 📈 실험 설계

### 1. 최종 데이터 도출

| 지표 | 설명 | 목적 |
|------|------|------|
| **ADE** | 평균 위치 오차 | 전체 정확도 |
| **FDE** | 최종 위치 오차 | 목적지 정확도 |
| **Miss Rate** | 2m 이상 오차 비율 | 실용성 |
| **Collision Rate** | 충돌 예측 비율 | 안전성 |
| **Diversity** | 다중 모달리티 다양성 | 생성 능력 |
| **TTC/PET** | 안전 지표 | Plan B 효과 |

**에이전트별 분석**:
- 차량 (Vehicle)
- 보행자 (Pedestrian)
- 자전거 (Biker)

### 2. 비교 대상 (베이스라인)

```
우리 모델: GNN-Diffusion Hybrid
    vs
베이스라인:
1. Social-STGCNN (CVPR 2020)
2. Trajectron++ (ECCV 2020)
3. A3TGCN (기존 구현)
4. 원본 MID/LED (이기종 미적용)
```

### 3. 결론 도출

**우리 모델의 우수성 입증**:
- ✅ ADE/FDE 개선 (베이스라인 대비)
- ✅ Diversity 향상 (다중 모달리티)
- ✅ 이기종 환경 적응성
- ✅ 안전성 향상 (Plan B)

---

## 💡 Diffusion 모델 선택 전략

### 옵션 1: LED 메인, MID Ablation

```
메인 모델: GNN-LED Hybrid
    ↓
Ablation Study:
1. GNN-LED vs GNN-MID (Diffusion 방식 비교)
2. GNN-LED vs LED only (GNN 효과 검증)
3. GNN-LED vs GNN only (Diffusion 효과 검증)
4. With Plan B vs Without Plan B (안전성 검증)
```

**장점**:
- LED가 더 빠르고 정확 → 메인 결과 강화
- MID는 ablation으로 → "우리가 LED를 선택한 이유" 설명
- 둘 다 구현 → 연구 깊이 증가

### 옵션 2: MID 메인, LED Ablation

```
메인 모델: GNN-MID Hybrid
    ↓
Ablation Study:
1. GNN-MID vs GNN-LED (이론적 근거 vs 속도)
2. Motion Indeterminacy 개념 강조
```

**장점**:
- 이론적 기여 강조
- "Motion Indeterminacy" 개념 활용
- 다양성 제어 메커니즘 설명

### 최종 선택 기준

| 기준 | LED 선택 | MID 선택 |
|------|---------|---------|
| **실용성 강조** | ✅ | - |
| **이론적 기여** | - | ✅ |
| **빠른 결과** | ✅ | - |
| **개념적 깊이** | - | ✅ |

**→ 현재 미결정, 구현 후 결정**

---

## 📝 학술 발표 구조 (추천)

### 1. Introduction

```
문제: 이기종 교통 환경에서 다중 모달리티 예측 어려움
기존 방법의 한계:
- GNN: 다중 모달리티 제한적
- Diffusion: 이기종 환경 미고려
우리 제안: GNN + Diffusion 하이브리드
```

### 2. Related Work

```
- Trajectory Prediction (Social-LSTM, Trajectron++)
- Graph Neural Networks (STGCNN, A3TGCN)
- Diffusion Models (MID, LED)
- 우리와의 차이점 강조
```

### 3. Methodology

```
3.1 Heterogeneous Scene Graph Construction
3.2 GNN Encoder (HeteroGAT)
3.3 Diffusion Decoder (LED or MID)
3.4 Safety-Guided Sampling (Plan B)
```

### 4. Experiments

```
4.1 Dataset: SDD Death Circle
4.2 Baselines: Social-STGCNN, Trajectron++, A3TGCN, MID/LED
4.3 Metrics: ADE, FDE, Miss Rate, Collision Rate, Diversity
4.4 Results:
    - Overall Performance (Table)
    - Agent-wise Analysis (Figure)
    - Scenario Analysis (Figure)
4.5 Ablation Study:
    - Diffusion 방식 비교 (우리 선택 정당화)
    - GNN 효과
    - Plan B 효과
```

### 5. Conclusion

```
기여:
1. 이기종 환경에서 Diffusion 확장
2. 씬 그래프 조건부 생성
3. 안전 가이드 샘플링
4. SDD Death Circle에서 SOTA 달성
```

---

## 📊 예상 Ablation Study 결과

### Table: Component Analysis

| Model | ADE ↓ | FDE ↓ | Diversity ↑ | Time (ms) ↓ |
|-------|-------|-------|-------------|-------------|
| **GNN-Diffusion (Ours)** | **0.85** | **1.65** | **0.92** | **46-886** |
| Diffusion only | 1.05 | 2.10 | 0.88 | 45-885 |
| GNN only | 1.20 | 2.50 | 0.30 | 10 |
| Social-STGCNN | 1.35 | 2.80 | 0.25 | 15 |
| Trajectron++ | 1.15 | 2.40 | 0.60 | 50 |

**설명**:
- "GNN이 Diffusion 성능을 크게 향상시킨다"
- "우리 하이브리드가 베이스라인 대비 우수하다"

---

## 🎯 예상 논문 기여도

### Main Contribution

1. 🏆 **Novel Architecture**: GNN-Diffusion Hybrid for Heterogeneous Agents
2. 🏆 **Safety Integration**: Plan B Safety-Guided Sampling
3. 🏆 **Empirical Results**: SOTA on SDD Death Circle

### Ablation Contribution

4. 📊 Diffusion 방식 비교 (LED vs MID)
5. 📊 Component Analysis (GNN, Diffusion, Plan B 각각의 효과)

---

## ✅ 학술 발표 핵심 메시지

```
"우리는 이기종 교통 환경을 위해
GNN과 Diffusion을 결합했고,
[LED/MID]를 선택하여 [빠르고 정확한/이론적으로 근거있는] 예측을 달성했으며,
Plan B로 안전성까지 보장했다."
```

**핵심**: LED vs MID가 아니라, **우리 하이브리드 vs 기존 방법들**

---

## 🔍 다음 단계

1. ⏳ LED/MID 중 하나 선택 (또는 둘 다 구현)
2. ⏳ 베이스라인 모델 구현/수집
3. ⏳ 실험 실행 및 결과 분석
4. ⏳ Ablation Study 수행
5. ⏳ 논문 작성
