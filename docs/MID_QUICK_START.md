# MID 모델 빠른 시작 가이드

## ✅ 구현 완료

MID (Motion Indeterminacy Diffusion) 모델이 완전히 구현되었습니다!

## 📁 구현된 파일

1. **`src/models/mid_model.py`**: MID 메인 모델
   - `MIDModel`: 기본 MID 모델
   - `HybridGNNMID`: GNN + MID 하이브리드
   - `TransformerDenoiser`: Transformer 기반 Denoiser
   - `ObservationEncoder`: 관측 궤적 인코더

2. **`src/training/mid_trainer.py`**: MID 학습 클래스
   - Diffusion 특화 학습 로직
   - 노이즈 예측 손실
   - Mixed Precision Training

3. **`configs/mid_config.yaml`**: MID 설정 파일

4. **`scripts/train_mid.py`**: 학습 스크립트

## 🚀 빠른 시작

### 1. 모델 생성 및 테스트

```python
from src.models.mid_model import create_mid_model
import torch

# GNN + MID 하이브리드 모델
model = create_mid_model(
    obs_steps=30,
    pred_steps=50,
    hidden_dim=128,
    num_diffusion_steps=100,
    use_gnn=True,
    node_features=9
)

print(f"파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
```

### 2. 샘플링 (추론)

```python
# 그래프 데이터로 샘플링
from torch_geometric.data import Data

graph_data = Data(
    x=torch.randn(10, 9),  # 10개 노드, 9개 특징
    edge_index=torch.randint(0, 10, (2, 20))
)

# 20개 궤적 샘플링 (DDIM 2 steps)
samples = model.sample(
    graph_data=graph_data,
    num_samples=20,
    ddim_steps=2
)

print(f"생성된 궤적: {samples.shape}")  # [20, batch, 50, 2]
```

### 3. 학습 실행

```bash
# 기본 학습
python scripts/train_mid.py --config configs/mid_config.yaml

# 데이터 디렉토리 지정
python scripts/train_mid.py \
    --config configs/mid_config.yaml \
    --data_dir data/processed
```

## 🎯 MID 핵심 개념

### Motion Indeterminacy

```
모든 가능한 영역 (불확정)
  ↓ (Diffusion Process)
점진적 불확정성 제거
  ↓
특정 궤적 (확정)
```

### 학습 과정

1. **Forward Diffusion**: 실제 궤적에 노이즈 추가
2. **노이즈 예측**: Denoiser가 노이즈 예측
3. **Loss 계산**: 예측 노이즈 vs 실제 노이즈 (MSE)

### 추론 과정

1. **노이즈에서 시작**: 랜덤 노이즈
2. **Reverse Diffusion**: 점진적으로 노이즈 제거
3. **조건부 생성**: 관측 정보를 조건으로 사용
4. **다중 샘플링**: 20개 이상의 다양한 궤적 생성

## 📊 모델 구조

```
입력: 관측 궤적 [batch, 30, 2]
  ↓
ObservationEncoder (LSTM)
  ↓
조건 임베딩 [batch, 128]
  ↓
Diffusion Process
  ├─ Forward: q_sample
  └─ Reverse: p_sample
       ↓
TransformerDenoiser
  ├─ 타임스텝 임베딩
  ├─ 조건 결합
  └─ 노이즈 예측
  ↓
출력: 예측 노이즈 [batch, 50, 2]
```

## 🔧 주요 파라미터

- **obs_steps**: 30 (3초, 10Hz)
- **pred_steps**: 50 (5초, 10Hz)
- **num_diffusion_steps**: 100 (학습), 2 (DDIM 추론)
- **hidden_dim**: 128
- **num_samples**: 20 (다중 모달리티)

## 📈 예상 성능

- **학습 시간**: 6-8시간 (30% 데이터), 18-24시간 (전체)
- **추론 시간**: ~0.8초 (100 steps), ~0.04초 (DDIM 2 steps)
- **다중 모달리티**: 20개 다양한 궤적 생성

## ✅ 다음 단계

1. ✅ MID 모델 구현 완료
2. ⏳ 데이터 전처리 확인
3. ⏳ 학습 실행
4. ⏳ 평가 및 비교

**MID 구현이 완료되었습니다! 학습을 시작할 수 있습니다.** 🎉

