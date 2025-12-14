# 🚀 빠른 학습 가이드 (1-2시간 완료)

## 속도 향상 방법 요약

### 1. Mixed Precision Training (FP16) ⚡
**속도 향상: 약 2배**

```python
# 자동으로 활성화됨 (CUDA 사용 시)
use_amp = True  # Mixed Precision Training
```

**장점**:
- 메모리 사용량 50% 감소
- 학습 속도 2배 향상
- 정확도 손실 거의 없음

### 2. 데이터 샘플링 📊
**속도 향상: 약 3배 (30% 샘플링 시)**

```python
sample_ratio = 0.3  # 30%만 사용
```

**전략**:
- 빠른 실험: 10-30% 샘플링
- 최종 학습: 100% 사용
- 검증 데이터는 100% 유지

### 3. 경량 모델 🪶
**속도 향상: 약 1.5배**

```python
hidden_channels = 32  # 기본 64 → 32로 축소
```

**트레이드오프**:
- 속도: ⬆️ 1.5배
- 정확도: ⬇️ 약 5-10% (보통 허용 가능)

### 4. 큰 배치 크기 📦
**속도 향상: 약 1.2-1.5배**

```python
batch_size = 64  # 기본 32 → 64
```

**주의사항**:
- GPU 메모리 필요 (16GB 이상 권장)
- 학습률 조정 필요 (큰 배치 = 큰 학습률)

### 5. Early Stopping 🛑
**시간 절약: 불필요한 에폭 제거**

```python
patience = 5  # 기본 10 → 5로 단축
```

## 빠른 학습 실행

### Colab에서 실행

```python
# 빠른 학습 (1-2시간)
!python scripts/training/fast_train.py \
    --data_dir data/processed \
    --batch_size 64 \
    --epochs 30 \
    --sample_ratio 0.3 \
    --use_amp \
    --lightweight
```

### 로컬에서 실행

```bash
python scripts/training/fast_train.py \
    --data_dir data/processed \
    --batch_size 64 \
    --epochs 30 \
    --sample_ratio 0.3 \
    --use_amp \
    --lightweight
```

## 속도 비교

| 설정 | 시간 | 정확도 | 메모리 |
|------|------|--------|--------|
| **기본 학습** | 4-6시간 | 100% | 8GB |
| **빠른 학습** | **1-2시간** | 90-95% | 4GB |
| **초고속 학습** | **30분-1시간** | 85-90% | 2GB |

### 초고속 학습 설정

```python
!python scripts/training/fast_train.py \
    --batch_size 128 \
    --epochs 20 \
    --sample_ratio 0.1 \  # 10%만 사용
    --use_amp \
    --lightweight
```

## 단계별 전략

### Phase 1: 빠른 실험 (30분)
```python
--sample_ratio 0.1 --epochs 10 --lightweight
```
목적: 코드 검증, 하이퍼파라미터 탐색

### Phase 2: 빠른 학습 (1-2시간)
```python
--sample_ratio 0.3 --epochs 30 --use_amp --lightweight
```
목적: 기본 모델 학습, 성능 확인

### Phase 3: 최종 학습 (4-6시간)
```python
--sample_ratio 1.0 --epochs 50 --use_amp
```
목적: 최종 모델, 논문용 결과

## GPU 메모리 최적화

### Colab 무료 (T4, 16GB)
```python
batch_size = 64  # 가능
sample_ratio = 0.3
use_amp = True
```

### Colab Pro (V100, 16GB)
```python
batch_size = 128  # 가능
sample_ratio = 0.5
use_amp = True
```

### 로컬 GPU (8GB)
```python
batch_size = 32  # 권장
sample_ratio = 0.3
use_amp = True
```

## 모니터링

### TensorBoard
```python
# 별도 셀에서 실행
!tensorboard --logdir runs --port 6006
```

### 학습 진행 확인
```python
# 학습 중 실시간 확인
import time
start_time = time.time()
# ... 학습 실행 ...
elapsed = time.time() - start_time
print(f"학습 시간: {elapsed/3600:.2f}시간")
```

## 문제 해결

### Out of Memory
```python
# 배치 크기 줄이기
--batch_size 32  # 또는 16
```

### 학습이 여전히 느림
```python
# 더 공격적인 샘플링
--sample_ratio 0.1  # 10%만 사용
--epochs 20  # 에폭 수 줄이기
```

### 정확도가 너무 낮음
```python
# 샘플링 비율 증가
--sample_ratio 0.5  # 50% 사용
--lightweight  # 제거 (전체 모델 사용)
```

## 최종 권장사항

**하루 내 완료 목표**:
1. 빠른 학습 모드 사용 (`fast_train.py`)
2. 30% 데이터 샘플링
3. Mixed Precision Training 활성화
4. 경량 모델 사용
5. Early Stopping (patience=5)

**예상 시간**: 1-2시간 (기존 4-6시간 대비 3-4배 빠름)

