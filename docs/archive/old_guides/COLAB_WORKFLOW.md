# Colab 전체 워크플로우 가이드

## 📊 전체 프로세스 흐름도

```
┌─────────────────────────────────────────────────────────────────┐
│                    Colab 환경 설정 (1회)                        │
│  - 라이브러리 설치                                              │
│  - 프로젝트 파일 업로드/클론                                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              데이터 준비 및 검증 (1회)                          │
│  - SDD 데이터셋 확인                                            │
│  - 호모그래피 행렬 확인                                         │
│  - 데이터 변환 확인                                              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              데이터 전처리 (1회)                                │
│  - 좌표 변환 (픽셀 → 미터)                                      │
│  - 슬라이딩 윈도우 생성                                          │
│  - 이상치 제거 및 보간                                          │
│  - Feature 정규화                                               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│            씬 그래프 생성 (1회 또는 필요시)                     │
│  - 노드 생성 (Agent, Map)                                       │
│  - 엣지 생성 (Spatial, Semantic)                                │
│  - PyTorch Geometric 변환                                       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              모델 학습 (반복 가능)                               │
│  - 데이터 로더 설정                                              │
│  - 모델 초기화                                                   │
│  - 학습 실행                                                     │
│  - 체크포인트 저장                                               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              모델 평가 (학습 후)                                 │
│  - ADE/FDE 계산                                                  │
│  - Miss Rate, Collision Rate                                    │
│  - 시나리오별 분석                                               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              결과 시각화 (선택사항)                              │
│  - 어텐션 가중치 시각화                                          │
│  - 궤적 예측 시각화                                              │
│  - 안전 지표 시각화                                              │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 단계별 실행 가이드

### Phase 1: Colab 환경 설정 (5분)

#### Step 1-1: 새 Colab 노트북 생성

1. [Google Colab](https://colab.research.google.com/) 접속
2. 새 노트북 생성
3. 이름: "Roundabout_AI_Training"

#### Step 1-2: 환경 설정 코드 실행

```python
# 첫 번째 셀: 환경 설정
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q torch-geometric torch-geometric-temporal
!pip install -q pandas numpy scipy scikit-learn matplotlib seaborn
!pip install -q opencv-python networkx tqdm pyyaml shapely
!pip install -q tensorboard

# GPU 확인
import torch
print(f"✓ GPU: {torch.cuda.is_available()}")
print(f"✓ GPU 이름: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

#### Step 1-3: 프로젝트 파일 업로드

**방법 A: GitHub에서 클론 (권장)**

```python
# 두 번째 셀: 프로젝트 클론
!git clone https://github.com/your-repo/Roundabout_AI.git
%cd Roundabout_AI
```

**방법 B: 직접 업로드**

```python
# 두 번째 셀: 파일 업로드
from google.colab import files
uploaded = files.upload()  # 프로젝트 ZIP 파일 업로드
!unzip project.zip -d Roundabout_AI
%cd Roundabout_AI
```

#### Step 1-4: 데이터셋 업로드

**방법 A: Google Drive 마운트 (권장)**

```python
# 세 번째 셀: Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

# 데이터 디렉토리 링크
import os
os.symlink('/content/drive/MyDrive/Roundabout_AI/data', '/content/Roundabout_AI/data')
```

**방법 B: 직접 업로드**

```python
# 세 번째 셀: 데이터 업로드
from google.colab import files
# data/sdd/converted/*.csv 파일들 업로드
# data/sdd/homography/H.txt 업로드
```

---

### Phase 2: 데이터 준비 및 검증 (10분)

#### Step 2-1: 데이터 확인

```python
# 네 번째 셀: 데이터 확인
import sys
sys.path.append('/content/Roundabout_AI')

from pathlib import Path
import pandas as pd

data_dir = Path('/content/Roundabout_AI/data/sdd/converted')
csv_files = list(data_dir.glob('*.csv'))

print(f"✓ 변환된 데이터 파일: {len(csv_files)}개")
for f in csv_files:
    df = pd.read_csv(f)
    print(f"  {f.name}: {len(df):,}행, {df['track_id'].nunique()}개 트랙")
```

#### Step 2-2: 호모그래피 확인

```python
# 다섯 번째 셀: 호모그래피 확인
import numpy as np

h_path = Path('/content/Roundabout_AI/data/sdd/homography/H.txt')
if h_path.exists():
    H = np.loadtxt(h_path)
    print(f"✓ 호모그래피 행렬:\n{H}")
    print(f"  형태: {H.shape}")
else:
    print("⚠️  호모그래피 행렬 없음 - 생성 필요")
```

---

### Phase 3: 데이터 전처리 (30분)

#### Step 3-1: 데이터 로드 및 통합

```python
# 여섯 번째 셀: 데이터 통합
import pandas as pd
from pathlib import Path

data_dir = Path('/content/Roundabout_AI/data/sdd/converted')
all_data = []

for csv_file in sorted(data_dir.glob('*.csv')):
    df = pd.read_csv(csv_file)
    df['video_id'] = csv_file.stem.replace('_converted', '')
    all_data.append(df)
    print(f"✓ {csv_file.name}: {len(df):,}행")

combined_df = pd.concat(all_data, ignore_index=True)
print(f"\n✓ 통합 데이터: {len(combined_df):,}행")
print(f"  트랙 수: {combined_df['track_id'].nunique()}")
print(f"  에이전트 타입: {sorted(combined_df['agent_type'].unique())}")
```

#### Step 3-2: 전처리 실행

```python
# 일곱 번째 셀: 전처리 파이프라인
import sys
sys.path.append('/content/Roundabout_AI')

from src.data_processing.preprocessor import (
    CoordinateTransformer,
    SlidingWindowGenerator,
    OutlierInterpolator,
    FeatureNormalizer
)

# 전처리 설정
obs_window = 30  # 3초 (10Hz)
pred_window = 50  # 5초 (10Hz)

# 1. 좌표 변환 (이미 완료됨 - 확인만)
print("1. 좌표 변환 확인...")
print(f"   X 범위: [{combined_df['x'].min():.2f}, {combined_df['x'].max():.2f}]")
print(f"   Y 범위: [{combined_df['y'].min():.2f}, {combined_df['y'].max():.2f}]")

# 2. 이상치 제거 및 보간
print("\n2. 이상치 제거 및 보간...")
interpolator = OutlierInterpolator()
# 각 트랙별로 처리
# (실제 구현 필요)

# 3. 슬라이딩 윈도우 생성
print("\n3. 슬라이딩 윈도우 생성...")
window_generator = SlidingWindowGenerator(
    obs_window=obs_window,
    pred_window=pred_window
)

windows = []
for track_id in combined_df['track_id'].unique()[:100]:  # 샘플링
    track_data = combined_df[combined_df['track_id'] == track_id].sort_values('frame_id')
    track_windows = window_generator.create_windows(track_data)
    windows.extend(track_windows)

print(f"   ✓ 생성된 윈도우: {len(windows)}개")

# 4. Feature 정규화
print("\n4. Feature 정규화...")
normalizer = FeatureNormalizer()
# 정규화 실행
# (실제 구현 필요)

print("\n✓ 전처리 완료!")
```

#### Step 3-3: 전처리 결과 저장

```python
# 여덟 번째 셀: 전처리 결과 저장
import pickle
from pathlib import Path

output_dir = Path('/content/Roundabout_AI/data/processed')
output_dir.mkdir(parents=True, exist_ok=True)

# 윈도우 데이터 저장
with open(output_dir / 'sdd_windows.pkl', 'wb') as f:
    pickle.dump(windows, f)

print(f"✓ 전처리 결과 저장: {output_dir / 'sdd_windows.pkl'}")
print(f"  윈도우 수: {len(windows)}개")
```

---

### Phase 4: 씬 그래프 생성 (선택사항, 20분)

#### Step 4-1: 씬 그래프 빌더 초기화

```python
# 아홉 번째 셀: 씬 그래프 생성 (선택사항)
import sys
sys.path.append('/content/Roundabout_AI')

from src.scene_graph.scene_graph_builder import SceneGraphBuilder

builder = SceneGraphBuilder(
    spatial_threshold=20.0,  # 20m
    use_semantic_edges=True
)

print("✓ 씬 그래프 빌더 초기화 완료")
```

#### Step 4-2: 샘플 프레임으로 테스트

```python
# 열 번째 셀: 샘플 씬 그래프 생성
import pandas as pd

# 샘플 프레임 선택
sample_frame = combined_df[combined_df['frame_id'] == combined_df['frame_id'].min() + 100]

# 씬 그래프 생성
graph = builder.build_from_frame(sample_frame)
pyg_data = builder.to_pytorch_geometric()

print(f"✓ 씬 그래프 생성 완료")
print(f"  노드 수: {pyg_data.x.size(0)}")
print(f"  엣지 수: {pyg_data.edge_index.size(1)}")
```

---

### Phase 5: 모델 학습 (1-4시간)

#### Step 5-1: 학습 설정

```python
# 열한 번째 셀: 학습 설정
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
        }
    },
    'logging': {
        'log_dir': '/content/Roundabout_AI/runs',
        'save_dir': '/content/Roundabout_AI/checkpoints'
    }
}

# 설정 파일 저장
config_path = Path('/content/Roundabout_AI/configs/colab_config.yaml')
config_path.parent.mkdir(parents=True, exist_ok=True)
with open(config_path, 'w') as f:
    yaml.dump(config, f)

print("✓ 학습 설정 저장 완료")
```

#### Step 5-2: 빠른 학습 실행

```python
# 열두 번째 셀: 빠른 학습 실행
import sys
sys.path.append('/content/Roundabout_AI')

from src.training.train import main as train_main
import argparse

# 빠른 학습 설정
args = argparse.Namespace(
    config='configs/colab_config.yaml',
    data_dir='data/processed',
    resume=None
)

print("=" * 80)
print("학습 시작")
print("=" * 80)

# 학습 실행
train_main()
```

#### Step 5-3: TensorBoard 모니터링

```python
# 열세 번째 셀: TensorBoard 실행 (별도 탭에서)
# 이 셀은 실행 후 별도 탭에서 TensorBoard 확인
%load_ext tensorboard
%tensorboard --logdir /content/Roundabout_AI/runs --port 6006
```

---

### Phase 6: 모델 평가 (30분)

#### Step 6-1: 모델 로드 및 평가

```python
# 열네 번째 셀: 모델 평가
import sys
sys.path.append('/content/Roundabout_AI')
import torch
from src.evaluation.evaluator import ModelEvaluator
from src.models.a3tgcn_model import create_a3tgcn_model

# 모델 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_a3tgcn_model(
    node_features=9,
    hidden_channels=64,
    pred_steps=50
)

checkpoint_path = Path('/content/Roundabout_AI/checkpoints/best_model.pth')
if checkpoint_path.exists():
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ 모델 로드 완료")
else:
    print("⚠️  체크포인트 없음 - 학습 먼저 실행 필요")

# 평가 실행
# (데이터 로더 필요)
```

#### Step 6-2: 평가 지표 계산

```python
# 열다섯 번째 셀: 평가 지표 계산
import sys
sys.path.append('/content/Roundabout_AI')
from src.evaluation.metrics import TrajectoryEvaluator
import numpy as np

evaluator = TrajectoryEvaluator()

# 더미 데이터로 테스트 (실제로는 모델 예측 결과 사용)
predicted = np.random.randn(10, 50, 2) * 0.5
ground_truth = predicted + np.random.randn(10, 50, 2) * 0.1

metrics = evaluator.evaluate(predicted, ground_truth)

print("=" * 60)
print("평가 지표")
print("=" * 60)
for key, value in metrics.items():
    if 'Rate' in key:
        print(f"{key:25s}: {value:.4f} ({value*100:.2f}%)")
    else:
        print(f"{key:25s}: {value:.4f} m")
```

---

### Phase 7: 결과 시각화 (선택사항, 20분)

#### Step 7-1: 어텐션 가중치 시각화

```python
# 열여섯 번째 셀: 어텐션 시각화
import sys
sys.path.append('/content/Roundabout_AI')
from src.visualization.attention_visualizer import AttentionVisualizer
import numpy as np

visualizer = AttentionVisualizer()

# 샘플 데이터
num_nodes = 10
positions = np.random.randn(num_nodes, 2) * 10
attention_matrix = visualizer.compute_distance_based_attention(positions)

# 시각화
visualizer.visualize_attention_heatmap(
    attention_matrix,
    output_path=Path('/content/Roundabout_AI/results/attention_heatmap.png')
)

print("✓ 어텐션 히트맵 저장 완료")
```

#### Step 7-2: 결과 다운로드

```python
# 열일곱 번째 셀: 결과 다운로드
from google.colab import files
from pathlib import Path

# 체크포인트 다운로드
files.download('/content/Roundabout_AI/checkpoints/best_model.pth')

# 결과 이미지 다운로드
for img_file in Path('/content/Roundabout_AI/results').glob('*.png'):
    files.download(str(img_file))

print("✓ 결과 다운로드 완료")
```

---

## 📋 체크리스트

### 환경 설정

- [ ] Colab 노트북 생성
- [ ] 라이브러리 설치 완료
- [ ] 프로젝트 파일 업로드/클론
- [ ] 데이터셋 업로드

### 데이터 준비

- [ ] 데이터 파일 확인
- [ ] 호모그래피 행렬 확인
- [ ] 데이터 통합 완료

### 전처리

- [ ] 좌표 변환 확인
- [ ] 이상치 제거 완료
- [ ] 슬라이딩 윈도우 생성
- [ ] Feature 정규화 완료

### 학습

- [ ] 학습 설정 완료
- [ ] 데이터 로더 테스트
- [ ] 모델 초기화 완료
- [ ] 학습 시작
- [ ] TensorBoard 모니터링

### 평가

- [ ] 모델 로드 완료
- [ ] 평가 지표 계산
- [ ] 결과 저장

---

## ⚡ 빠른 실행 (최소 설정)

전체 과정을 한 번에 실행하려면:

```python
# 모든 단계를 포함한 통합 스크립트
!python /content/Roundabout_AI/scripts/colab_full_pipeline.py \
    --data_dir /content/Roundabout_AI/data/sdd/converted \
    --output_dir /content/Roundabout_AI/data/processed \
    --batch_size 32 \
    --epochs 30 \
    --sample_ratio 0.3 \
    --use_amp
```

---

## 🔄 병렬 실행 전략

### Colab (GPU 학습)

- Phase 5: 모델 학습
- Phase 6: 모델 평가
- Phase 7: 결과 시각화

### 로컬 (데이터 준비)

- Phase 2: 데이터 준비 및 검증
- Phase 3: 데이터 전처리
- Phase 4: 씬 그래프 생성

이렇게 분리하면 효율적입니다!
