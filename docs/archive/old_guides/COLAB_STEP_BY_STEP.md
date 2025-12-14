# Colab 단계별 실행 가이드

## 🎯 전체 프로세스 요약

```
┌─────────────────────────────────────────────────────────────┐
│  Phase 1: 환경 설정 (5분)                                  │
│  ├─ 라이브러리 설치                                         │
│  ├─ 프로젝트 파일 업로드                                    │
│  └─ 데이터셋 업로드                                         │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  Phase 2: 데이터 준비 (10분)                                │
│  ├─ 데이터 확인                                             │
│  ├─ 호모그래피 확인                                         │
│  └─ 데이터 통합                                            │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  Phase 3: 데이터 전처리 (30분)                              │
│  ├─ 좌표 변환 확인                                          │
│  ├─ 이상치 제거                                             │
│  ├─ 슬라이딩 윈도우 생성                                    │
│  └─ Feature 정규화                                         │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  Phase 4: 모델 학습 (1-4시간)                               │
│  ├─ 데이터 로더 설정                                        │
│  ├─ 모델 초기화                                             │
│  ├─ 학습 실행                                               │
│  └─ 체크포인트 저장                                        │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  Phase 5: 모델 평가 (30분)                                   │
│  ├─ 모델 로드                                               │
│  ├─ 평가 지표 계산                                          │
│  └─ 결과 저장                                               │
└─────────────────────────────────────────────────────────────┘
```

## 📝 상세 실행 코드

### 🔧 Phase 1: 환경 설정

#### 셀 1: 라이브러리 설치

```python
# Colab 첫 번째 셀
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q torch-geometric torch-geometric-temporal
!pip install -q pandas numpy scipy scikit-learn matplotlib seaborn
!pip install -q opencv-python networkx tqdm pyyaml shapely tensorboard

import torch
print(f"✓ GPU: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU 이름: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
```

#### 셀 2: 프로젝트 파일 업로드

```python
# 방법 A: GitHub 클론
!git clone https://github.com/your-repo/Roundabout_AI.git
%cd Roundabout_AI

# 방법 B: 직접 업로드
from google.colab import files
uploaded = files.upload()  # ZIP 파일 업로드
!unzip project.zip -d Roundabout_AI
%cd Roundabout_AI
```

#### 셀 3: 데이터셋 업로드

```python
# 방법 A: Google Drive 마운트 (권장)
from google.colab import drive
drive.mount('/content/drive')

import os
os.symlink('/content/drive/MyDrive/Roundabout_AI/data', '/content/Roundabout_AI/data')

# 방법 B: 직접 업로드
from google.colab import files
# data/sdd/converted/*.csv 파일들 업로드
# data/sdd/homography/H.txt 업로드
```

---

### 📊 Phase 2: 데이터 준비

#### 셀 4: 데이터 확인

```python
import sys
sys.path.append('/content/Roundabout_AI')

from pathlib import Path
import pandas as pd
import numpy as np

# 변환된 데이터 확인
data_dir = Path('/content/Roundabout_AI/data/sdd/converted')
csv_files = sorted(data_dir.glob('*.csv'))

print("=" * 60)
print("데이터 확인")
print("=" * 60)
print(f"✓ 변환된 데이터 파일: {len(csv_files)}개\n")

for f in csv_files:
    df = pd.read_csv(f)
    print(f"{f.name}:")
    print(f"  행 수: {len(df):,}")
    print(f"  트랙 수: {df['track_id'].nunique()}")
    print(f"  프레임 범위: {df['frame_id'].min()} ~ {df['frame_id'].max()}")
    print(f"  에이전트 타입: {sorted(df['agent_type'].unique())}")
    print()
```

#### 셀 5: 호모그래피 확인

```python
# 호모그래피 행렬 확인
h_path = Path('/content/Roundabout_AI/data/sdd/homography/H.txt')

if h_path.exists():
    H = np.loadtxt(h_path)
    print("✓ 호모그래피 행렬:")
    print(H)
    print(f"\n  형태: {H.shape}")
else:
    print("⚠️  호모그래피 행렬 없음")
```

---

### 🔄 Phase 3: 데이터 전처리

#### 셀 6: 데이터 통합

```python
# 모든 비디오 데이터 통합
all_data = []

for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    df['video_id'] = csv_file.stem.replace('_converted', '')
    all_data.append(df)
    print(f"✓ {csv_file.name}: {len(df):,}행")

combined_df = pd.concat(all_data, ignore_index=True)

print("\n" + "=" * 60)
print("통합 데이터 통계")
print("=" * 60)
print(f"총 행 수: {len(combined_df):,}")
print(f"총 트랙 수: {combined_df['track_id'].nunique()}")
print(f"에이전트 타입: {sorted(combined_df['agent_type'].unique())}")
print(f"프레임 범위: {combined_df['frame_id'].min()} ~ {combined_df['frame_id'].max()}")
```

#### 셀 7: 전처리 실행

```python
from src.data_processing.preprocessor import TrajectoryPreprocessor

# 전처리 설정
preprocessor = TrajectoryPreprocessor(
    obs_window=30,  # 3초 (10Hz)
    pred_window=50,  # 5초 (10Hz)
    sampling_rate=10.0
)

print("전처리 시작...")
print("1. 슬라이딩 윈도우 생성 중...")

# 샘플링 (전체 데이터가 많을 경우)
sample_ratio = 0.3  # 30%만 사용
if sample_ratio < 1.0:
    unique_tracks = combined_df['track_id'].unique()
    sampled_tracks = np.random.choice(
        unique_tracks,
        size=int(len(unique_tracks) * sample_ratio),
        replace=False
    )
    sampled_df = combined_df[combined_df['track_id'].isin(sampled_tracks)]
    print(f"  샘플링: {len(sampled_tracks)}/{len(unique_tracks)} 트랙 ({sample_ratio*100:.0f}%)")
else:
    sampled_df = combined_df

# 윈도우 생성
windows = []
for track_id in sampled_df['track_id'].unique():
    track_data = sampled_df[sampled_df['track_id'] == track_id].sort_values('frame_id')
    if len(track_data) >= 80:  # 최소 길이 확인
        track_windows = preprocessor.create_sliding_windows(track_data)
        windows.extend(track_windows)

print(f"  ✓ 생성된 윈도우: {len(windows):,}개")

# 저장
import pickle
output_dir = Path('/content/Roundabout_AI/data/processed')
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / 'sdd_windows.pkl', 'wb') as f:
    pickle.dump(windows, f)

print(f"✓ 전처리 완료 및 저장: {output_dir / 'sdd_windows.pkl'}")
```

---

### 🚀 Phase 4: 모델 학습

#### 셀 8: 학습 설정

```python
import yaml
from pathlib import Path

config = {
    'model': {
        'name': 'a3tgcn',
        'node_features': 9,
        'hidden_channels': 64,
        'num_layers': 2,
        'periods': 30,
        'pred_steps': 50
    },
    'data': {
        'data_dir': '/content/Roundabout_AI/data/processed',
        'batch_size': 32,
        'num_workers': 2,
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15
    },
    'training': {
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'scheduler': 'reduce_on_plateau',
        'loss': 'mse',
        'num_epochs': 50,
        'early_stopping': {
            'patience': 10,
            'min_delta': 0.001
        },
        'max_grad_norm': 1.0
    },
    'logging': {
        'log_dir': '/content/Roundabout_AI/runs',
        'save_dir': '/content/Roundabout_AI/checkpoints',
        'save_every': 10
    }
}

config_path = Path('/content/Roundabout_AI/configs/colab_config.yaml')
config_path.parent.mkdir(parents=True, exist_ok=True)

with open(config_path, 'w') as f:
    yaml.dump(config, f)

print("✓ 학습 설정 저장 완료")
print(f"  설정 파일: {config_path}")
```

#### 셀 9: 데이터 로더 생성

```python
from src.training.data_loader import TrajectoryDataset, create_dataloader, split_dataset
import pickle

# 전처리된 데이터 로드
with open('/content/Roundabout_AI/data/processed/sdd_windows.pkl', 'rb') as f:
    windows = pickle.load(f)

print(f"로드된 윈도우: {len(windows):,}개")

# 데이터 분할
train_windows, val_windows, test_windows = split_dataset(
    windows,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)

print(f"\n데이터 분할:")
print(f"  학습: {len(train_windows):,}개")
print(f"  검증: {len(val_windows):,}개")
print(f"  테스트: {len(test_windows):,}개")

# 데이터셋 생성
train_dataset = TrajectoryDataset(train_windows)
val_dataset = TrajectoryDataset(val_windows)

# 데이터 로더 생성
train_loader = create_dataloader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2
)

val_loader = create_dataloader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2
)

print(f"\n✓ 데이터 로더 생성 완료")
print(f"  학습 배치 수: {len(train_loader)}")
print(f"  검증 배치 수: {len(val_loader)}")
```

#### 셀 10: 모델 생성 및 학습

```python
import torch
from src.models.a3tgcn_model import create_a3tgcn_model
from src.training.trainer import create_trainer

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"디바이스: {device}")

# 모델 생성
model = create_a3tgcn_model(
    node_features=9,
    hidden_channels=64,
    pred_steps=50,
    use_map=False
)

print(f"\n모델 정보:")
print(f"  파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

# Trainer 생성
trainer_config = {
    'optimizer': 'adam',
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'scheduler': 'reduce_on_plateau',
    'loss': 'mse',
    'num_epochs': 50,
    'early_stopping': {
        'patience': 10,
        'min_delta': 0.001
    },
    'log_dir': '/content/Roundabout_AI/runs',
    'save_dir': '/content/Roundabout_AI/checkpoints',
    'max_grad_norm': 1.0
}

trainer = create_trainer(model, train_loader, val_loader, trainer_config)

# 학습 시작
print("\n" + "=" * 80)
print("학습 시작")
print("=" * 80)

trainer.train(50)
```

#### 셀 11: TensorBoard (별도 탭)

```python
# TensorBoard 실행 (별도 탭에서 확인)
%load_ext tensorboard
%tensorboard --logdir /content/Roundabout_AI/runs --port 6006
```

---

### 📈 Phase 5: 모델 평가

#### 셀 12: 모델 평가

```python
from src.evaluation.evaluator import ModelEvaluator
import torch

# 모델 로드
checkpoint_path = Path('/content/Roundabout_AI/checkpoints/best_model.pth')

if checkpoint_path.exists():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluator = ModelEvaluator(model, device)

    # 평가 실행
    metrics = evaluator.evaluate_dataset(val_loader, max_batches=50)

    print("=" * 60)
    print("평가 결과")
    print("=" * 60)
    for key, value in metrics.items():
        if 'Rate' in key:
            print(f"{key:25s}: {value:.4f} ({value*100:.2f}%)")
        else:
            print(f"{key:25s}: {value:.4f} m")
else:
    print("⚠️  체크포인트 없음")
```

---

### 💾 Phase 6: 결과 다운로드

#### 셀 13: 결과 다운로드

```python
from google.colab import files
from pathlib import Path

# 체크포인트 다운로드
checkpoint_path = Path('/content/Roundabout_AI/checkpoints/best_model.pth')
if checkpoint_path.exists():
    files.download(str(checkpoint_path))
    print("✓ 모델 체크포인트 다운로드")

# 결과 이미지 다운로드
results_dir = Path('/content/Roundabout_AI/results')
for img_file in results_dir.glob('*.png'):
    files.download(str(img_file))
    print(f"✓ {img_file.name} 다운로드")
```

---

## ⚡ 빠른 실행 (통합 스크립트)

모든 단계를 한 번에 실행:

```python
# 통합 파이프라인 실행
!python /content/Roundabout_AI/scripts/colab_full_pipeline.py \
    --data_dir /content/Roundabout_AI/data/sdd/converted \
    --output_dir /content/Roundabout_AI/data/processed \
    --batch_size 32 \
    --epochs 30 \
    --sample_ratio 0.3 \
    --use_amp
```

---

## 📊 진행 상황 모니터링

### TensorBoard

```python
# 별도 셀에서 실행
%tensorboard --logdir /content/Roundabout_AI/runs
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

---

## 🔄 병렬 실행 전략

### Colab에서 실행

- ✅ Phase 4: 모델 학습 (GPU 활용)
- ✅ Phase 5: 모델 평가
- ✅ Phase 6: 결과 시각화

### 로컬에서 실행 (선택사항)

- ✅ Phase 2: 데이터 준비 및 검증
- ✅ Phase 3: 데이터 전처리
- ✅ 결과 분석 및 문서화

---

## ⚠️ 주의사항

1. **세션 타임아웃**: Colab 무료 버전은 12시간 제한
2. **GPU 할당**: 무료 버전은 T4 GPU (16GB)
3. **데이터 저장**: Google Drive에 저장 권장
4. **체크포인트**: 주기적으로 다운로드 권장

---

## 🎯 최종 체크리스트

- [ ] 환경 설정 완료
- [ ] 데이터 업로드 완료
- [ ] 전처리 완료
- [ ] 학습 시작
- [ ] TensorBoard 모니터링
- [ ] 평가 완료
- [ ] 결과 다운로드
