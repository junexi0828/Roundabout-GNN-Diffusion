# 최종 모델 선택: MID (Motion Indeterminacy Diffusion)

## 🎯 최종 결정

**메인 모델**: MID (Motion Indeterminacy Diffusion)
**향후 연구**: LED (Leapfrog Diffusion Model)

---

## 📊 선택 근거

### 우리 프로젝트 상황

1. **학부 연구 프로젝트**
   - 제한된 시간과 자원
   - 구현 성공 확률이 가장 중요

2. **제한된 GPU 자원**
   - 학습 안정성 우선
   - 디버깅 용이성 필요

3. **연구 목표**
   - 실시간성보다 **정확도/다양성** 중시
   - "회전교차로의 다양한 시나리오를 얼마나 그럴싸하게 만들었냐"가 핵심

---

## 🔍 MID vs LED 비교

### 핵심 철학

| 모델 | 철학 | 비유 |
|------|------|------|
| **MID** | "처음부터 끝까지 정석대로 꼼꼼하게" | 정석 화가 |
| **LED** | "대충 그리고(Leapfrog), 다듬자" | 스피드 페인터 |

### 상세 비교

| 비교 항목 | MID (우리 선택) | LED |
|----------|----------------|-----|
| **구현 난이도** | ✅ **낮음** (표준 DDPM) | ❌ 높음 (Leapfrog Initializer) |
| **학습 안정성** | ✅ **안정적** | ⚠️ 까다로움 |
| **코드 복잡도** | ✅ **간단** | ❌ 복잡 |
| **레퍼런스** | ✅ **많음** | ⚠️ 적음 |
| **추론 속도** | ⚠️ 느림 (~0.8초) | ✅ 매우 빠름 (~0.04초) |
| **정확도** | ✅ 우수 | ✅ 더 우수 |
| **다양성** | ✅ **매우 우수** | ✅ 우수 |
| **이론적 근거** | ✅ **명확** (Motion Indeterminacy) | ⚠️ 실용적 |
| **디버깅** | ✅ **쉬움** | ❌ 어려움 |
| **우리 목적** | ✅ **연구용 최적** | ⚠️ 상용화용 |

---

## ✅ MID 선택 이유

### 1. 구현 난이도가 낮음

**MID**:
```python
# 표준 Diffusion 프로세스
class MIDModel:
    def __init__(self):
        self.encoder = HeteroGAT()  # 우리 GNN
        self.diffusion = DDPM()     # 표준 구현
        self.denoiser = Transformer()  # 일반적
```

**LED**:
```python
# 복잡한 구조
class LEDModel:
    def __init__(self):
        self.encoder = HeteroGAT()
        self.leapfrog_init = LeapfrogInitializer()  # 별도 학습 필요!
        self.diffusion = CustomDiffusion()  # 커스텀
        self.refiner = Refiner()
```

→ **MID가 오픈소스 코드 활용 및 통합이 훨씬 쉬움**

### 2. 연구 목적에 부합

**우리 목표**:
```
"GNN으로 상호작용을 잘 보고,
생성형 AI로 다양한 미래를 예측했다"
```

**MID의 강점**:
- ✅ "Motion Indeterminacy" 개념 → 논문에서 설명하기 좋음
- ✅ 다양성 표현 매우 우수 → 회전교차로 시나리오 생성
- ✅ 이론적 근거 명확 → 학술적 가치

**LED의 강점**:
- ⚠️ "0.04초 만에 예측" → 속도 자랑 (우리 목표 아님)
- ⚠️ 실시간 자율주행용 → 상용화 목적

→ **MID가 우리 연구 목적에 더 적합**

### 3. 자원 효율성

**학습 과정**:
- MID: 정석적 → **디버깅 쉬움** ✅
- LED: 까다로움 → 에러 발생 확률 높음 ❌

**GPU 자원**:
- MID: 표준 학습 → 안정적
- LED: Leapfrog Initializer 추가 학습 → 자원 더 필요

→ **제한된 자원에서 MID가 안전**

### 4. 성공 확률

| 단계 | MID | LED |
|------|-----|-----|
| 구현 | ✅ 높음 | ⚠️ 중간 |
| 학습 | ✅ 높음 | ⚠️ 중간 |
| 결과 | ✅ 높음 | ✅ 높음 |
| **전체** | **✅ 높음** | **⚠️ 중간** |

→ **MID가 성공 확률이 가장 높은 안전한 길** 🛣️

---

## 📝 논문/발표 전략

### 메인 모델: MID

**논문 작성**:
```
"우리는 안정적인 학습과 높은 다양성 표현을 위해
MID (Motion Indeterminacy Diffusion) 구조를 채택했다.

MID는 motion indeterminacy를 명시적으로 모델링하여
회전교차로의 복잡한 상호작용 시나리오를
다양하고 그럴싸하게 생성할 수 있다."
```

**강조 포인트**:
1. ✅ Motion Indeterminacy 개념
2. ✅ 다양한 미래 시나리오 생성
3. ✅ 이기종 환경 적응
4. ✅ 안전 가이드 샘플링

### 향후 연구: LED

**논문/발표에서**:
```
"향후 연구 과제 (Future Work):

실시간 자율주행 시스템 적용을 위해
LED (Leapfrog Diffusion Model)와 같은
경량화 모델 도입을 검토할 수 있다.

LED는 Leapfrog Initializer를 통해
추론 속도를 20-30배 향상시킬 수 있어,
실시간성이 요구되는 환경에 적합하다."
```

**방어 전략**:
- 교수님: "속도 너무 느린 거 아니냐?"
- 답변: "현재는 정확도와 다양성에 집중했으며, 향후 LED 도입으로 실시간성을 확보할 수 있습니다."

---

## 🚀 구현 계획

### Phase 1: MID 기반 구현

```python
# src/models/diffusion/mid_model.py
class MIDTrajectoryPredictor(nn.Module):
    def __init__(self):
        # 기존 GNN 재사용
        self.encoder = HeteroGAT(
            node_types=['car', 'pedestrian', 'biker'],
            edge_types=[('car', 'yield', 'pedestrian'), ...]
        )

        # MID Diffusion (표준 DDPM)
        self.diffusion = MotionIndeterminacyDiffusion(
            timesteps=100,  # 또는 DDIM 2 steps
            beta_schedule='linear'
        )

        # Transformer Denoiser
        self.denoiser = TransformerDenoiser(
            hidden_dim=128,
            num_layers=4
        )

    def forward(self, obs):
        # 1. GNN으로 상호작용 특징 추출
        context = self.encoder(obs)

        # 2. MID로 다중 궤적 생성
        trajectories = self.diffusion.sample(
            context=context,
            num_samples=20
        )

        return trajectories  # [batch, 20, 50, 2]
```

### Phase 2: 학습 및 평가

**학습 시간 예상**:
- 빠른 학습 (30% 데이터): 6-8시간
- 전체 학습 (100% 데이터): 18-24시간

**평가 지표**:
- ADE, FDE (정확도)
- Diversity (다양성)
- TTC, PET (안전성)

### Phase 3: 논문 작성

**구조**:
1. Introduction: 이기종 환경 + 다중 모달리티 필요성
2. Methodology: GNN-MID Hybrid
3. Experiments: 베이스라인 대비 우수성
4. Future Work: LED 도입 가능성

---

## 📊 예상 결과

### 베이스라인 대비

| Model | ADE ↓ | FDE ↓ | Diversity ↑ |
|-------|-------|-------|-------------|
| **GNN-MID (Ours)** | **0.92** | **1.78** | **0.90** |
| Social-STGCNN | 1.35 | 2.80 | 0.25 |
| Trajectron++ | 1.15 | 2.40 | 0.60 |
| A3TGCN | 1.20 | 2.50 | 0.30 |
| MID only | 1.05 | 2.10 | 0.88 |

### Ablation Study

| Component | ADE ↓ | FDE ↓ | Diversity ↑ |
|-----------|-------|-------|-------------|
| GNN-MID (Full) | 0.92 | 1.78 | 0.90 |
| MID only | 1.05 | 2.10 | 0.88 |
| GNN only | 1.20 | 2.50 | 0.30 |
| + Plan B | **0.85** | **1.65** | 0.90 |

---

## 🎯 핵심 메시지

### 학술 발표

```
"우리는 이기종 회전교차로 환경에서
GNN과 MID를 결합하여
다양하고 안전한 미래 궤적 예측을 달성했다.

MID의 Motion Indeterminacy 개념을
씬 그래프 조건부 생성에 적용하여
복잡한 상호작용 시나리오를
효과적으로 모델링했다."
```

### LED 언급

```
"향후 연구로 LED와 같은 경량화 모델을 도입하여
실시간 자율주행 시스템에 적용할 수 있다."
```

---

## ✅ 최종 체크리스트

### MID 선택 이유
- [x] 구현 난이도 낮음
- [x] 학습 안정성 높음
- [x] 레퍼런스 풍부
- [x] 이론적 근거 명확
- [x] 연구 목적 부합
- [x] 성공 확률 높음

### LED 활용
- [x] 향후 연구 과제로 제시
- [x] 개념 설명 준비
- [x] 실시간성 강조
- [x] 방어 전략 수립

---

## 🔗 참고 자료

### MID
- **논문**: [CVPR 2022] "Stochastic Trajectory Prediction via Motion Indeterminacy Diffusion"
- **코드**: https://github.com/Gutianpei/MID
- **개념**: Motion Indeterminacy 명시적 모델링

### LED
- **논문**: [CVPR 2023] "Leapfrog Diffusion Model for Stochastic Trajectory Prediction"
- **코드**: https://github.com/MediaBrain-SJTU/LED
- **개념**: Leapfrog Initializer로 빠른 샘플링

---

## 🚀 다음 단계

1. ✅ MID 코드 다운로드 및 분석
2. ✅ GNN 인코더와 통합
3. ⏳ 학습 파이프라인 구축
4. ⏳ 실험 및 평가
5. ⏳ 논문 작성 (LED는 Future Work)

**MID 구현을 시작합니다!** 🎯
