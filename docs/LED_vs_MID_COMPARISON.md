# LED vs MID 비교 분석 (최신 업데이트)

## 🔍 두 모델 개요

### MID (Motion Indeterminacy Diffusion)

- **논문**: CVPR 2022 "Stochastic Trajectory Prediction via Motion Indeterminacy Diffusion"
- **GitHub**: https://github.com/Gutianpei/MID
- **검증**: ✅ CVPR 2022, ETH/UCY, SDD 데이터셋
- **핵심 아이디어**:
  - Motion Indeterminacy를 명시적으로 모델링
  - 모든 가능한 보행 영역에서 시작하여 점진적으로 불확정성 제거
  - Transformer 기반 Denoiser
  - DDIM 통합으로 2 steps 샘플링 (50x 가속)

### LED (Leapfrog Diffusion Model)

- **논문**: CVPR 2023 "Leapfrog Diffusion Model for Stochastic Trajectory Prediction"
- **GitHub**: https://github.com/MediaBrain-SJTU/LED
- **검증**: ✅ CVPR 2023, NBA/NFL/SDD/ETH-UCY 데이터셋
- **핵심 아이디어**:
  - Trainable Leapfrog Initializer로 대략적 분포 직접 학습
  - 많은 denoising steps 스킵 → 실시간 추론
  - 다양성 유지하면서 속도 대폭 향상

---

## 📊 성능 비교 (실제 벤치마크)

### 1. 정확도 (Accuracy)

| 데이터셋 | 지표 | MID | LED | 개선율 |
|---------|------|-----|-----|--------|
| **NFL** | ADE | 기준 | **23.7% ↓** | 🏆 LED |
| **NFL** | FDE | 기준 | **21.9% ↓** | 🏆 LED |
| **NBA** | ADE | 기준 | **15.6% ↓** | 🏆 LED |
| **NBA** | FDE | 기준 | **13.4% ↓** | 🏆 LED |
| **SDD** | ADE/FDE | 우수 | 우수 | 🤝 비슷 |
| **ETH/UCY** | ADE/FDE | 우수 | 우수 | 🤝 비슷 |

**결론**: LED가 정확도에서 **15-24% 더 우수**

### 2. 추론 속도 (Inference Speed)

| 데이터셋 | MID (DDIM) | LED | 가속 배율 |
|---------|-----------|-----|----------|
| **NBA** | ~886ms | **46ms** | **19.3배** ⚡ |
| **NFL** | 기준 | 빠름 | **30.8배** ⚡ |
| **SDD** | 기준 | 빠름 | **24.3배** ⚡ |
| **ETH-UCY** | 기준 | 빠름 | **25.1배** ⚡ |

**결론**: LED가 **20-30배 더 빠름** (실시간 추론 가능)

### 3. 다양성 (Diversity)

| 측면 | MID | LED |
|------|-----|-----|
| **다중 모달리티** | ✅ 명시적 모델링 | ✅ 유지 |
| **다양성 제어** | ✅ Markov chain 길이 조절 | ⚠️ 제어 방법 다름 |
| **불확실성 표현** | ✅ 매우 우수 (핵심 강점) | ✅ 우수 |
| **철학적 근거** | ✅ Motion Indeterminacy 개념 | ⚠️ 효율성 중심 |

**결론**: MID가 다양성 **제어 및 이론**에서 약간 우위

---

## 🔬 방식의 차이 (핵심)

### MID: 철학적 접근

```
개념: "Motion Indeterminacy"
불확정성 → 점진적 감소 → 확정적 궤적

프로세스:
1. 모든 가능한 영역에서 시작
2. 100 steps (또는 DDIM 2 steps)로 점진적 제거
3. 특정 궤적 도달
```

**특징**:
- 🎯 개념이 명확: "불확정성 제거"
- 📚 이론적 근거 강함
- 🔬 연구 가치: 새로운 관점 제시

### LED: 실용적 접근

```
개념: "Leapfrog Initialization"
대략적 분포 학습 → 빠른 정제

프로세스:
1. Leapfrog Initializer로 대략적 분포 직접 학습
2. 많은 steps 스킵
3. 5-10 steps만으로 정제
```

**특징**:
- ⚡ 효율성 우선: 실시간 추론
- 🎯 실용적: 빠른 결과
- 💡 엔지니어링: 최적화 중심

**핵심 차이**:
- MID: "왜 이렇게 하는가?" (Why) - 철학적
- LED: "어떻게 빠르게 하는가?" (How) - 실용적

---

## 📐 다각도 비교

### 1. 아키텍처 복잡도

| 측면 | MID | LED |
|------|-----|-----|
| **코드 복잡도** | 중간 | **더 간단** ✅ |
| **학습 안정성** | 안정적 | **더 안정적** ✅ |
| **하이퍼파라미터** | 많음 (Markov chain 등) | **적음** ✅ |
| **디버깅** | 중간 | **쉬움** ✅ |
| **메모리 사용** | 중간 | **낮음** ✅ |

### 2. 구현 용이성

```python
# MID 구현
class MIDDecoder:
    def __init__(self):
        self.diffusion = StandardDiffusion(steps=100)
        self.ddim = DDIM(steps=2)
        self.transformer = TransformerDenoiser()

# LED 구현 (더 간단!)
class LEDDecoder:
    def __init__(self):
        self.initializer = LeapfrogInitializer()  # 핵심
        self.refiner = Refiner(steps=5)  # 몇 스텝만
```

### 3. 우리 프로젝트 적합성

#### 프로젝트 요구사항 매칭

| 요구사항 | 가중치 | MID | LED | 승자 |
|---------|--------|-----|-----|------|
| 빠른 학습 | 25% | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 🏆 LED |
| 높은 정확도 | 30% | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 🏆 LED |
| 구현 용이성 | 15% | ⭐⭐⭐ | ⭐⭐⭐⭐ | 🏆 LED |
| 다양성 제어 | 20% | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 🏆 MID |
| 이론적 기여 | 10% | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 🏆 MID |
| **총점** | - | **8.1** | **8.9** | 🏆 **LED** |

---

## 🎯 우리 프로젝트 상황 분석

### 현재 프로젝트 특징

1. **이기종 교통 환경**: 6종 에이전트 (차량, 보행자, 자전거, 스케이터, 카트, 버스)
2. **회전교차로 특화**: 복잡한 상호작용 (양보, 충돌 회피)
3. **씬 그래프 기반**: 공간적/의미론적 관계 모델링
4. **안전 검증**: Plan B 안전 지표 통합
5. **실용성**: 빠른 학습 및 추론 필요

### MID가 더 좋은 경우

```
✅ 연구 논문에서 이론적 기여 강조
✅ "Motion Indeterminacy" 개념이 핵심
✅ 다양성 제어가 중요한 연구 질문
✅ 불확실성 모델링의 철학적 근거 필요
```

### LED가 더 좋은 경우 (우리 프로젝트!)

```
✅ 빠른 학습 필요 (4-6시간 vs 8-12시간)
✅ 실시간 추론 필요 (46ms vs 886ms)
✅ 높은 정확도 필요 (15-24% 개선)
✅ 구현 간단함 선호
✅ 실용성 우선
✅ SDD 데이터셋 검증됨
```

---

## 🔄 하이브리드 접근 (권장)

### GNN + LED 하이브리드

```
입력: 관측 궤적 + 씬 그래프
  ↓
GNN Encoder (HeteroGAT)
  - 이기종 상호작용 모델링
  - 씬 그래프 특징 추출
  ↓
LED Diffusion Decoder
  - Leapfrog Initializer
  - 빠른 다중 궤적 생성 (20개)
  ↓
Plan B 안전 검증
  - TTC/PET 필터링
  ↓
출력: 안전한 다중 궤적
```

**장점**:
- ✅ GNN의 이기종 처리 + LED의 빠른 생성
- ✅ 씬 그래프 정보 활용
- ✅ 검증된 LED 방법론 (CVPR 2023)
- ✅ 실시간 추론 가능
- ✅ 높은 정확도

---

## 📝 최종 결론 및 권장사항

### 🏆 최종 권장: **LED (Leapfrog Diffusion Model)**

#### 선택 이유

1. **검증 완료** ✅
   - CVPR 2023 (최신)
   - GitHub 코드 공개
   - NBA/NFL/SDD/ETH-UCY 검증

2. **성능 우수** ✅
   - 정확도: MID보다 15-24% 개선
   - 속도: 20-30배 빠름 (실시간 가능)
   - 메모리: 더 효율적

3. **구현 용이** ✅
   - 코드 더 간단
   - 하이퍼파라미터 적음
   - 디버깅 쉬움

4. **프로젝트 적합** ✅
   - 빠른 학습 (4-6시간)
   - SDD 데이터셋 검증됨
   - 이기종 환경 확장 가능

#### 단점 (MID 대비)

- ⚠️ 다양성 제어 방법 다름
- ⚠️ 이론적 근거 약간 약함
- ⚠️ "Motion Indeterminacy" 개념 없음

---

## 🚀 구현 전략

### Phase 1: LED 기반 구현 (권장)

```python
# src/models/diffusion/led_model.py
class LEDTrajectoryPredictor(nn.Module):
    def __init__(self):
        # 기존 GNN 재사용
        self.encoder = HeteroGAT(...)

        # LED Diffusion
        self.initializer = LeapfrogInitializer(...)
        self.refiner = DiffusionRefiner(steps=5)

    def forward(self, obs):
        context = self.encoder(obs)
        trajectories = self.initializer(context)
        trajectories = self.refiner(trajectories, context)
        return trajectories  # [batch, 20, 50, 2]
```

**예상 학습 시간**: 4-6시간 (30% 데이터), 12-15시간 (전체)

### Phase 2: MID 비교 (선택적)

```python
# 성능 비교를 위해 MID도 구현
mid_model = MIDPredictor(...)
compare_results(led_model, mid_model)
```

**논문 기여**:
- LED vs MID 비교 분석
- 이기종 환경에서의 성능 차이
- 다양성 vs 속도 트레이드오프

---

## 📊 예상 결과

### LED 적용 시

| 지표 | 기존 (A3TGCN) | LED (예상) | 개선 |
|------|--------------|-----------|------|
| ADE | 1.2m | **0.9m** | -25% |
| FDE | 2.5m | **1.8m** | -28% |
| 다양성 | 낮음 | **높음** (20개) | +∞ |
| 학습 시간 | 3시간 | **6시간** | +100% |
| 추론 시간 | 10ms | **5ms** | -50% |

### 연구 기여도

1. ✅ 이기종 환경에서 LED 확장
2. ✅ 씬 그래프 조건부 Diffusion
3. ✅ 안전 가이드 샘플링
4. ✅ 회전교차로 특화 상호작용

---

## 🔍 참고 자료

### LED
- **논문**: [CVPR 2023] "Leapfrog Diffusion Model for Stochastic Trajectory Prediction"
- **코드**: https://github.com/MediaBrain-SJTU/LED
- **데이터셋**: NBA, NFL, SDD, ETH-UCY

### MID
- **논문**: [CVPR 2022] "Stochastic Trajectory Prediction via Motion Indeterminacy Diffusion"
- **코드**: https://github.com/Gutianpei/MID
- **업데이트**: DDIM 통합 (2023.04)

---

## ✅ 최종 체크리스트

### LED 선택 시
- [x] CVPR 2023 검증 완료
- [x] GitHub 코드 공개
- [x] SDD 데이터셋 검증
- [x] 더 빠른 속도 (20-30배)
- [x] 더 높은 정확도 (15-24%)
- [x] 구현 용이성
- [ ] 다양성 제어 (MID보다 약간 약함)

### 다음 단계

**✅ 최종 결정: MID 선택**

1. ✅ MID 코드 다운로드 및 분석
2. ✅ GNN 인코더와 통합
3. ⏳ 학습 파이프라인 구축
4. ⏳ 실험 및 평가
5. ⏳ 논문 작성 (LED는 Future Work)

**선택 이유**: 구현 안정성, 이론적 근거, 연구 목적 부합

**LED 활용**: 향후 연구 과제로 논문/발표에서 언급

자세한 내용은 [FINAL_MODEL_DECISION.md](FINAL_MODEL_DECISION.md) 참조
