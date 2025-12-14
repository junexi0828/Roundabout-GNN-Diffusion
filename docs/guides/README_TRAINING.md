# 모델 학습 실행 가이드

## ✅ 구현 완료 기능

### 1. 하이퍼파라미터 튜닝

- **Learning Rate**: 1e-4 ~ 1e-2 범위 자동 탐색
- **Batch Size**: 16, 32, 64 자동 테스트
- **Epochs**: 50, 100, 150 조합 탐색
- **Optimizer**: Adam, AdamW 비교
- **Scheduler**: ReduceLROnPlateau, Cosine, None 비교
- **Random Search / Grid Search** 지원

### 2. 학습/검증 데이터 분할

- 자동 데이터셋 분할 (기본: 70% 학습, 15% 검증, 15% 테스트)
- 랜덤 셔플로 편향 방지
- 전처리된 윈도우 데이터 자동 로드

### 3. GPU 메모리 최적화

- GPU 메모리에 따른 자동 배치 크기 조정
- 주기적 메모리 캐시 정리 (100 배치마다)
- 비동기 데이터 전송 (non_blocking)
- cuDNN 자동 튜닝 활성화
- 실시간 GPU 메모리 모니터링

### 4. 학습 진행 모니터링

- **TensorBoard 통합**: 실시간 손실, 학습률, GPU 메모리 추적
- **진행률 표시**: tqdm 기반 상세 진행률
- **그래디언트 모니터링**: 주기적 그래디언트 노름 로깅
- **Early Stopping**: 검증 손실 개선 없으면 자동 종료
- **체크포인트 저장**: 최고 모델 및 주기적 저장

## 빠른 시작

### 1. 필수 패키지 설치

```bash
# 가상환경 활성화
source venv/bin/activate

# 필수 패키지 설치 (torch는 별도 설치 필요)
pip install pyyaml tqdm tensorboard
```

### 2. 기본 학습 실행

```bash
# 샘플 데이터로 테스트
python src/training/train.py \
    --data_dir data/processed \
    --epochs 10 \
    --batch_size 8
```

### 3. 하이퍼파라미터 튜닝

```bash
python src/training/hyperparameter_tuning.py \
    --method random \
    --n_trials 10 \
    --max_epochs 20
```

### 4. 학습 모니터링

```bash
# 별도 터미널에서
tensorboard --logdir runs
```

## 주요 파일

- `src/training/train.py`: 메인 학습 스크립트
- `src/training/trainer.py`: 학습 파이프라인 (GPU 최적화 포함)
- `src/training/hyperparameter_tuning.py`: 하이퍼파라미터 튜닝
- `src/training/data_loader.py`: 데이터 로딩 및 분할
- `configs/training_config.yaml`: 학습 설정 파일
- `scripts/train_model.sh`: 학습 실행 쉘 스크립트
- `docs/TRAINING_GUIDE.md`: 상세 가이드

## 사용 예제

### 커스텀 하이퍼파라미터로 학습

```bash
python src/training/train.py \
    --learning_rate 0.001 \
    --batch_size 32 \
    --epochs 100
```

### 체크포인트에서 재개

```bash
python src/training/train.py \
    --resume checkpoints/checkpoint_epoch_50.pth
```

### GPU 메모리 최적화 확인

학습 중 자동으로:

- GPU 메모리 < 8GB: 배치 크기 16으로 자동 조정
- 주기적 메모리 캐시 정리
- 실시간 메모리 사용량 표시

## 출력 파일

- `checkpoints/best_model.pth`: 최고 성능 모델
- `checkpoints/checkpoint_epoch_N.pth`: 주기적 체크포인트
- `runs/`: TensorBoard 로그
- `results/hyperparameter_tuning.json`: 튜닝 결과

## 다음 단계

1. 실제 데이터셋 다운로드 및 전처리
2. 하이퍼파라미터 튜닝 실행
3. 최적 하이퍼파라미터로 전체 학습
4. 모델 평가 및 시각화
