# 모델 학습 가이드

## 개요

회전교차로 상호작용 예측 모델의 학습 파이프라인 사용 가이드입니다.

## 주요 기능

- ✅ **하이퍼파라미터 튜닝**: Learning Rate, Batch Size, Epochs 등 자동 탐색
- ✅ **학습/검증 데이터 분할**: 자동 데이터셋 분할 (기본: 70/15/15)
- ✅ **GPU 메모리 최적화**: 자동 배치 크기 조정, 메모리 모니터링
- ✅ **학습 진행 모니터링**: TensorBoard 통합, 실시간 진행률 표시

## 사전 준비

### 1. 데이터 전처리

학습 전에 데이터를 전처리해야 합니다:

```bash
# 샘플 데이터로 테스트
python src/data_processing/preprocessor.py \
    data/interaction-dataset-repo/recorded_trackfiles/.TestScenarioForScripts/vehicle_tracks_000.csv

# 실제 데이터 전처리 (데이터셋 다운로드 후)
python src/data_processing/preprocessor.py data/interaction/DR_USA_Roundabout_FT/vehicle_tracks_000.csv
```

### 2. 설정 파일 확인

`configs/training_config.yaml` 파일을 확인하고 필요시 수정하세요.

## 기본 학습 실행

### 방법 1: Python 스크립트 직접 실행

```bash
# 기본 설정으로 학습
python src/training/train.py

# 커스텀 설정 파일 사용
python src/training/train.py --config configs/my_config.yaml

# 하이퍼파라미터 오버라이드
python src/training/train.py \
    --learning_rate 0.001 \
    --batch_size 64 \
    --epochs 150
```

### 방법 2: 쉘 스크립트 사용

```bash
./scripts/train_model.sh

# 커스텀 옵션
./scripts/train_model.sh configs/my_config.yaml data/processed --learning_rate 0.001
```

## 하이퍼파라미터 튜닝

### Random Search (권장)

```bash
python src/training/hyperparameter_tuning.py \
    --method random \
    --n_trials 20 \
    --max_epochs 30
```

### Grid Search

```bash
python src/training/hyperparameter_tuning.py \
    --method grid \
    --max_epochs 30
```

### 결과 확인

튜닝 결과는 `results/hyperparameter_tuning.json`에 저장됩니다:

```json
{
  "best_hyperparameters": {
    "learning_rate": 0.001,
    "batch_size": 32,
    "hidden_channels": 64,
    ...
  },
  "best_val_loss": 0.0234,
  "all_results": [...]
}
```

## 학습 모니터링

### TensorBoard

학습 중 실시간 모니터링:

```bash
tensorboard --logdir runs
```

브라우저에서 `http://localhost:6006` 접속

### 모니터링 항목

- **Loss/Train**: 학습 손실
- **Loss/Validation**: 검증 손실
- **Learning_Rate**: 학습률 변화
- **GPU/Memory_Allocated_GB**: GPU 메모리 사용량
- **Training/Gradient_Norm**: 그래디언트 노름

## GPU 메모리 최적화

### 자동 최적화

스크립트가 자동으로:

- GPU 메모리에 따라 배치 크기 조정
- 주기적 메모리 캐시 정리
- 비동기 데이터 전송 (non_blocking)

### 수동 조정

`configs/training_config.yaml`에서:

```yaml
data:
  batch_size: 16 # GPU 메모리가 부족하면 줄이기
  num_workers: 4 # 데이터 로딩 병렬화
```

## 체크포인트 관리

### 저장 위치

- 최고 모델: `checkpoints/best_model.pth`
- 주기적 체크포인트: `checkpoints/checkpoint_epoch_N.pth`

### 학습 재개

```bash
python src/training/train.py \
    --resume checkpoints/checkpoint_epoch_50.pth
```

## 주요 하이퍼파라미터

### Learning Rate

- **권장 범위**: 1e-4 ~ 1e-2
- **시작값**: 1e-3
- **스케줄러**: ReduceLROnPlateau (검증 손실 기반)

### Batch Size

- **GPU 메모리 8GB 미만**: 16
- **GPU 메모리 8-16GB**: 32
- **GPU 메모리 16GB 이상**: 64

### Epochs

- **기본**: 100
- **Early Stopping**: patience=10 (검증 손실 개선 없으면 조기 종료)

## 문제 해결

### Out of Memory (OOM) 오류

1. 배치 크기 줄이기: `--batch_size 16`
2. 모델 크기 줄이기: `hidden_channels` 감소
3. Gradient Accumulation 사용 (향후 구현)

### 학습이 느린 경우

1. `num_workers` 증가: `data.num_workers: 4`
2. Mixed Precision Training 사용 (향후 구현)
3. GPU 사용 확인: `nvidia-smi`

### 검증 손실이 개선되지 않는 경우

1. Learning Rate 조정
2. 모델 구조 변경 (은닉층 수, 채널 수)
3. 데이터 품질 확인

## 예제 실행

### 1. 샘플 데이터로 테스트

```bash
# 1. 샘플 데이터 전처리
python src/data_processing/preprocessor.py \
    data/interaction-dataset-repo/recorded_trackfiles/.TestScenarioForScripts/vehicle_tracks_000.csv

# 2. 학습 실행 (빠른 테스트)
python src/training/train.py \
    --data_dir data/processed \
    --epochs 10 \
    --batch_size 8
```

### 2. 실제 데이터로 학습

```bash
# 1. 모든 시나리오 전처리
for scenario in DR_USA_Roundabout_FT DR_CHN_Roundabout_LN DR_DEU_Roundabout_OF; do
    for file in data/interaction/$scenario/vehicle_tracks_*.csv; do
        python src/data_processing/preprocessor.py "$file"
    done
done

# 2. 학습 실행
python src/training/train.py \
    --data_dir data/processed \
    --epochs 100 \
    --batch_size 32
```

## 결과 확인

학습 완료 후:

1. **최고 모델**: `checkpoints/best_model.pth`
2. **학습 히스토리**: TensorBoard 로그 (`runs/`)
3. **하이퍼파라미터 튜닝 결과**: `results/hyperparameter_tuning.json`

## 다음 단계

1. 모델 평가: `src/evaluation/` (구현 예정)
2. 예측 시각화: `src/visualization/` (구현 예정)
3. 모델 비교: `src/models/model_comparison.py`
