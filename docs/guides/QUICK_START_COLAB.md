# 🚀 Colab 빠른 시작 가이드 (하루 완료 목표)

## 1. 호모그래피 추정 (30분)

### 방법 A: 빠른 스케일링 (권장, 1분)

```bash
# 로컬에서 실행
python scripts/auto_homography_estimation.py --method quick --output data/sdd/homography/H.txt
```

**장점**: 즉시 완료, 수동 작업 불필요
**단점**: 정확도는 낮지만 연구 시작에는 충분

### 방법 B: 자동 특징점 매칭 (30분)

```bash
# 위성 지도 다운로드 (Google Earth에서 스크린샷)
# SDD 비디오 프레임 추출
python scripts/auto_homography_estimation.py \
    --method SIFT \
    --video data/sdd/deathCircle/video0/frame_0000.jpg \
    --satellite satellite_map.jpg \
    --output data/sdd/homography/H.txt
```

**장점**: 더 정확한 호모그래피
**단점**: 위성 지도 필요

### 방법 C: 기존 연구 재사용 (5분)

```python
# 다른 연구에서 사용한 호모그래피 행렬 재사용
# GitHub에서 검색: "SDD Death Circle homography"
```

## 2. Colab 환경 설정 (10분)

### Step 1: 새 Colab 노트북 생성

1. [Google Colab](https://colab.research.google.com/) 접속
2. 새 노트북 생성

### Step 2: 환경 설정 코드 실행

```python
# 첫 셀에 실행
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q torch-geometric torch-geometric-temporal
!pip install -q pandas numpy scipy scikit-learn matplotlib seaborn
!pip install -q opencv-python networkx tqdm pyyaml

# GPU 확인
import torch
print(f"✓ GPU: {torch.cuda.is_available()}")
```

### Step 3: 프로젝트 파일 업로드

```python
# 방법 1: GitHub에서 클론
!git clone https://github.com/your-repo/Roundabout_AI.git
%cd Roundabout_AI

# 방법 2: 직접 업로드
from google.colab import files
uploaded = files.upload()  # 프로젝트 ZIP 파일
!unzip project.zip
```

### Step 4: 데이터셋 업로드

```python
# 방법 1: Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')
# data/sdd/ 폴더를 Drive에 업로드 후 링크

# 방법 2: 직접 업로드
from google.colab import files
# data/sdd/converted/*.csv 파일들 업로드
```

## 3. 데이터 전처리 (30분)

```python
# Colab 노트북에서 실행
import sys
sys.path.append('/content/Roundabout_AI')

from src.data_processing.preprocessor import TrajectoryPreprocessor
import pandas as pd
from pathlib import Path

# 데이터 로드
data_dir = Path('/content/Roundabout_AI/data/sdd/converted')
df = pd.read_csv(data_dir / 'video0_converted.csv')

# 전처리
preprocessor = TrajectoryPreprocessor(
    obs_window=30,
    pred_window=50,
    sampling_rate=10.0
)

# 회전교차로 중심 계산
center = preprocessor.calculate_roundabout_center(df)
print(f"회전교차로 중심: {center}")

# 슬라이딩 윈도우 생성
windows = preprocessor.create_sliding_windows(df)
print(f"✓ 윈도우 생성: {len(windows)}개")
```

## 4. 모델 학습 (1-2시간, 최적화 버전)

### 방법 A: 빠른 학습 (권장, 1-2시간)

```python
# Colab 노트북에서 실행 - 최적화된 빠른 학습
!python scripts/fast_train.py \
    --data_dir data/processed \
    --batch_size 64 \
    --epochs 30 \
    --sample_ratio 0.3 \
    --use_amp \
    --lightweight
```

**최적화 기능**:

- ✅ Mixed Precision Training (FP16) - 약 2배 속도 향상
- ✅ 데이터 샘플링 (30%만 사용) - 약 3배 속도 향상
- ✅ 경량 모델 (hidden_channels 32) - 약 1.5배 속도 향상
- ✅ 큰 배치 크기 (64) - GPU 활용 극대화

**총 속도 향상: 약 3-4배** (4-6시간 → 1-2시간)

### 방법 B: 일반 학습 (4-6시간)

```python
# Colab 노트북에서 실행
from src.training.train import main as train_main
import yaml

# 설정 파일 생성
config = {
    'data': {
        'data_dir': '/content/Roundabout_AI/data/processed',
        'batch_size': 32,
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15
    },
    'model': {
        'name': 'A3TGCN',
        'node_features': 9,
        'hidden_channels': 64,
        'pred_steps': 50
    },
    'training': {
        'epochs': 50,
        'learning_rate': 0.001,
        'device': 'cuda'
    }
}

with open('config.yaml', 'w') as f:
    yaml.dump(config, f)

# 학습 시작
train_main()
```

**Colab Pro 사용 시**:

- 더 빠른 GPU (T4 → V100)
- 더 긴 세션 시간
- 더 많은 RAM

## 5. 결과 평가 (1시간)

```python
# 평가 스크립트 실행
from src.evaluation.evaluator import ModelEvaluator

evaluator = ModelEvaluator(model_path='results/best_model.pth')
results = evaluator.evaluate(test_loader)

print(f"ADE: {results['ade']:.3f}m")
print(f"FDE: {results['fde']:.3f}m")
```

## 시간 배분 요약

| 작업            | 시간        | 방법                      |
| --------------- | ----------- | ------------------------- |
| 호모그래피 추정 | 1분         | 빠른 스케일링 (방법 A)    |
| Colab 설정      | 10분        | 위 설정 코드 사용         |
| 데이터 전처리   | 30분        | 자동화 스크립트           |
| 모델 학습       | **1-2시간** | **빠른 학습 (방법 A)** ⚡ |
| 결과 평가       | 30분        | 평가 스크립트             |
| **총 시간**     | **2-3시간** | **하루 내 완료 가능** ✅  |

**속도 향상 팁**:

- 빠른 학습 모드: `--sample_ratio 0.3 --use_amp --lightweight`
- Colab Pro: 더 빠른 GPU (V100)
- 큰 배치 크기: `--batch_size 64` (GPU 메모리 허용 시)

## 팁

1. **호모그래피는 나중에 개선 가능**: 빠른 스케일링으로 시작하고, 연구 진행 중 정확도 개선
2. **Colab Pro 사용**: 무료 버전은 세션 제한이 있어 학습 중단될 수 있음
3. **중간 결과 저장**: 주기적으로 체크포인트 저장
4. **배치 크기 조정**: GPU 메모리 부족 시 `batch_size` 줄이기

## 문제 해결

### GPU 메모리 부족

```python
# 배치 크기 줄이기
config['data']['batch_size'] = 16  # 또는 8
```

### 세션 타임아웃

```python
# 주기적으로 체크포인트 저장
# Colab Pro 사용 권장
```

### 데이터 업로드 느림

```python
# Google Drive 사용 또는 작은 샘플로 먼저 테스트
```
