# 로컬 실행 상태 확인

## ✅ 현재 상태

### 설치 완료
- ✅ **PyTorch 2.9.1** 설치 완료
- ✅ **MPS 사용 가능** (Apple Silicon GPU 활용 가능)
- ✅ **기본 라이브러리** (pandas, numpy, matplotlib 등) 설치 완료

### 설치 필요
- ⚠️ **torch-geometric** (설치 시도 중)
- ⚠️ **torch-geometric-temporal** (선택사항, A3TGCN 사용 시 필요)

## 로컬 실행 가능 여부

### ✅ **가능합니다!**

**이유**:
1. PyTorch 설치 완료
2. MPS (Metal Performance Shaders) 사용 가능 → Apple Silicon GPU 활용
3. 데이터 준비 완료
4. 메모리 8GB → 최적화된 설정으로 실행 가능

## 최적화된 실행 방법

### 방법 1: MPS 사용 (권장, 가장 빠름)

```bash
source venv/bin/activate
python scripts/training/fast_train.py \
    --data_dir data/processed \
    --batch_size 16 \
    --epochs 30 \
    --sample_ratio 0.3 \
    --lightweight \
    --device mps
```

**예상 시간**: 1-2시간
**메모리 사용**: ~4-6GB

### 방법 2: CPU 모드 (안정적)

```bash
source venv/bin/activate
python scripts/training/fast_train.py \
    --data_dir data/processed \
    --batch_size 8 \
    --epochs 30 \
    --sample_ratio 0.2 \
    --lightweight \
    --device cpu
```

**예상 시간**: 2-4시간
**메모리 사용**: ~3-4GB

### 방법 3: 최소 설정 (가장 빠름, 테스트용)

```bash
source venv/bin/activate
python scripts/training/fast_train.py \
    --data_dir data/processed \
    --batch_size 4 \
    --epochs 20 \
    --sample_ratio 0.1 \
    --lightweight \
    --device mps
```

**예상 시간**: 30분-1시간
**메모리 사용**: ~2GB

## MPS vs CPU 비교

| 항목 | MPS | CPU |
|------|-----|-----|
| **속도** | ⚡ 빠름 (2-3배) | 보통 |
| **메모리** | 비슷 | 비슷 |
| **안정성** | 높음 | 매우 높음 |
| **권장 배치 크기** | 16 | 8 |

## 주의사항

### 1. 메모리 관리
- 8GB 메모리 제한
- 배치 크기 4-16 권장
- 데이터 샘플링 권장 (10-30%)

### 2. torch-geometric-temporal
- A3TGCN 모델 사용 시 필요
- 설치 실패해도 기본 GAT 모델은 사용 가능
- 대안: Heterogeneous GNN 모델 사용

### 3. 학습 시간
- Colab GPU (T4) 대비: 약 2-3배 느림
- 하지만 로컬에서 충분히 실행 가능
- MPS 사용 시 속도 향상

## 빠른 시작

### 1. torch-geometric 설치 (필요 시)

```bash
source venv/bin/activate
pip install torch-geometric
```

### 2. 데이터 전처리 확인

```bash
source venv/bin/activate
python -c "import pandas as pd; df = pd.read_csv('data/sdd/converted/video0_converted.csv'); print(f'데이터 준비: {len(df):,}행')"
```

### 3. 빠른 학습 시작

```bash
source venv/bin/activate
python scripts/training/fast_train.py \
    --device mps \
    --batch_size 16 \
    --sample_ratio 0.3 \
    --lightweight
```

## 결론

**로컬 실행 완전히 가능합니다!** ✅

MacBook Air에서도 다음 조건으로 충분히 학습 가능:
- ✅ MPS 사용 (Apple Silicon GPU)
- ✅ 작은 배치 크기 (4-16)
- ✅ 데이터 샘플링 (10-30%)
- ✅ 경량 모델 사용

**예상 학습 시간**: 1-4시간 (설정에 따라)

Colab보다는 느리지만, 로컬에서 편리하게 실행 가능합니다!

