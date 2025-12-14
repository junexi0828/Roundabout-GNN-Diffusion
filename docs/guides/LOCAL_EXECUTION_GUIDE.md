# 로컬 실행 가이드 (MacBook Air)

## 시스템 환경

- **기기**: MacBook Air (Apple Silicon)
- **메모리**: 8GB
- **CPU**: 8코어 (4 performance + 4 efficiency)
- **GPU**: 없음 (CPU 모드 또는 MPS 사용)
- **디스크**: 충분 (46GB 여유)

## 로컬 실행 가능 여부

### ✅ 실행 가능하지만 최적화 필요

**제약사항**:
1. GPU 없음 → CPU 또는 MPS (Metal Performance Shaders) 사용
2. 메모리 8GB → 작은 배치 크기 필요
3. Apple Silicon → PyTorch MPS 지원 가능

## 설치 단계

### 1. PyTorch 설치 (Apple Silicon용)

```bash
# 가상환경 활성화
source venv/bin/activate

# PyTorch 설치 (Apple Silicon 최적화)
pip install torch torchvision torchaudio

# PyTorch Geometric 설치
pip install torch-geometric torch-geometric-temporal

# 기타 필수 라이브러리
pip install -r requirements.txt
```

### 2. MPS (Metal Performance Shaders) 확인

```python
import torch
print(f"MPS 사용 가능: {torch.backends.mps.is_available()}")
print(f"MPS 빌드됨: {torch.backends.mps.is_built()}")
```

**MPS 장점**:
- Apple Silicon GPU 활용
- CPU보다 빠른 학습 속도
- 메모리 효율적

## 최적화된 실행 설정

### 방법 1: CPU 모드 (안정적)

```bash
# 작은 배치 크기와 데이터 샘플링
python scripts/training/fast_train.py \
    --data_dir data/processed \
    --batch_size 8 \
    --epochs 30 \
    --sample_ratio 0.2 \
    --lightweight \
    --device cpu
```

**설정**:
- 배치 크기: 8 (메모리 제한)
- 데이터 샘플링: 20% (빠른 실험)
- 경량 모델: hidden_channels=32
- 예상 시간: 2-4시간

### 방법 2: MPS 모드 (권장, 더 빠름)

```bash
# MPS 사용 (Apple Silicon GPU)
python scripts/training/fast_train.py \
    --data_dir data/processed \
    --batch_size 16 \
    --epochs 30 \
    --sample_ratio 0.3 \
    --lightweight \
    --device mps
```

**설정**:
- 배치 크기: 16 (MPS 사용 시)
- 데이터 샘플링: 30%
- 경량 모델: hidden_channels=32
- 예상 시간: 1-2시간

### 방법 3: 최소 설정 (가장 빠름)

```bash
# 최소 설정으로 빠른 테스트
python scripts/training/fast_train.py \
    --data_dir data/processed \
    --batch_size 4 \
    --epochs 20 \
    --sample_ratio 0.1 \
    --lightweight \
    --device cpu
```

**설정**:
- 배치 크기: 4
- 데이터 샘플링: 10%
- 에폭: 20
- 예상 시간: 30분-1시간

## 메모리 최적화 팁

### 1. 배치 크기 조정
```python
# 메모리 부족 시
batch_size = 4  # 또는 8

# 메모리 여유 시
batch_size = 16  # MPS 사용 시
```

### 2. 데이터 샘플링
```python
# 빠른 실험
sample_ratio = 0.1  # 10%만 사용

# 기본 학습
sample_ratio = 0.3  # 30% 사용

# 최종 학습
sample_ratio = 1.0  # 전체 사용
```

### 3. 경량 모델
```python
# 기본 모델
hidden_channels = 64

# 경량 모델 (메모리 절약)
hidden_channels = 32
```

### 4. Gradient Accumulation
```python
# 작은 배치를 여러 번 누적
accumulation_steps = 4  # batch_size=4 * 4 = 효과적 배치 크기 16
```

## 실행 전 체크리스트

- [ ] PyTorch 설치 확인
- [ ] MPS 사용 가능 여부 확인
- [ ] 데이터 전처리 완료
- [ ] 메모리 여유 공간 확인 (최소 2GB)
- [ ] 배치 크기 설정 확인

## 예상 성능

| 설정 | 배치 크기 | 샘플링 | 예상 시간 | 메모리 사용 |
|------|----------|--------|----------|------------|
| **CPU 최소** | 4 | 10% | 30분-1시간 | ~2GB |
| **CPU 기본** | 8 | 20% | 2-4시간 | ~4GB |
| **MPS 기본** | 16 | 30% | 1-2시간 | ~4GB |
| **MPS 최적** | 16 | 50% | 2-3시간 | ~6GB |

## 문제 해결

### Out of Memory (OOM) 에러
```bash
# 배치 크기 줄이기
--batch_size 4

# 데이터 샘플링 증가
--sample_ratio 0.1
```

### 학습이 너무 느림
```bash
# MPS 사용 (가능한 경우)
--device mps

# 더 공격적인 샘플링
--sample_ratio 0.1
```

### MPS 사용 불가
```bash
# CPU 모드로 전환
--device cpu
```

## 권장 워크플로우

### Phase 1: 빠른 테스트 (30분)
```bash
python scripts/training/fast_train.py \
    --batch_size 4 \
    --epochs 10 \
    --sample_ratio 0.1 \
    --lightweight \
    --device mps
```
목적: 코드 검증, 기본 동작 확인

### Phase 2: 기본 학습 (1-2시간)
```bash
python scripts/training/fast_train.py \
    --batch_size 16 \
    --epochs 30 \
    --sample_ratio 0.3 \
    --lightweight \
    --device mps
```
목적: 기본 모델 학습, 성능 확인

### Phase 3: 최종 학습 (2-4시간)
```bash
python scripts/training/fast_train.py \
    --batch_size 16 \
    --epochs 50 \
    --sample_ratio 0.5 \
    --device mps
```
목적: 최종 모델, 논문용 결과

## 결론

**로컬 실행 가능합니다!**

다만 다음을 권장합니다:
1. ✅ MPS 사용 (가능한 경우)
2. ✅ 작은 배치 크기 (4-16)
3. ✅ 데이터 샘플링 (10-30%)
4. ✅ 경량 모델 사용
5. ✅ Early Stopping 활성화

이 설정으로 MacBook Air에서도 충분히 학습이 가능합니다.

