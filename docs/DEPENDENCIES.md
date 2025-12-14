# 의존성 관리 가이드

## 📦 의존성 파일 구조

프로젝트에는 세 가지 의존성 파일이 있습니다:

1. **`requirements.txt`**: 전체 의존성 (기본)
2. **`requirements-dev.txt`**: 개발용 추가 의존성
3. **`requirements-minimal.txt`**: 최소 필수 의존성

## 🚀 설치 방법

### 기본 설치 (권장)

```bash
# 가상환경 생성
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# 또는
venv\Scripts\activate     # Windows

# 패키지 설치
pip install --upgrade pip
pip install -r requirements.txt
```

### 개발 환경 설치

```bash
pip install -r requirements-dev.txt
```

### 최소 설치 (핵심 기능만)

```bash
pip install -r requirements-minimal.txt
```

## 📋 주요 의존성 설명

### Core Deep Learning

- **torch>=2.0.0**: PyTorch 메인 라이브러리
- **torchvision>=0.15.0**: 이미지 처리
- **torchaudio>=2.0.0**: 오디오 처리

### Graph Neural Networks

- **torch-geometric>=2.3.0**: 그래프 신경망 라이브러리
- **torch-geometric-temporal>=0.54.0**: 시공간 그래프 신경망

**PyTorch Geometric 의존성**:

- `xxhash`: 해시 함수
- `aiohttp`: 비동기 HTTP
- `psutil`: 시스템 정보
- `requests`: HTTP 요청

### Data Processing

- **pandas>=2.0.0**: 데이터프레임 처리
- **numpy>=1.24.0,<2.3.0**: 수치 연산 (opencv-python 호환성)
- **scipy>=1.10.0**: 과학 계산

### Graph Processing

- **networkx>=3.1**: 그래프 분석

### Geometric Operations

- **shapely>=2.0.0**: 기하학적 연산 (Plan B 안전 지표)

### Computer Vision

- **opencv-python>=4.8.0**: 이미지 처리 (SDD 데이터)

### Visualization

- **matplotlib>=3.7.0**: 플롯 생성
- **seaborn>=0.12.0**: 통계 시각화

### Utilities

- **tqdm>=4.65.0**: 진행 표시줄
- **pyyaml>=6.0**: YAML 설정 파일
- **tensorboard>=2.13.0**: 학습 모니터링

### Machine Learning

- **scikit-learn>=1.3.0**: 머신러닝 유틸리티

### Jupyter

- **jupyter>=1.0.0**: 노트북 환경
- **ipykernel>=6.25.0**: Python 커널
- **notebook>=6.5.0**: 노트북 서버

## 🔧 플랫폼별 설치

### Apple Silicon (M1/M2/M3)

```bash
# PyTorch (Apple Silicon 최적화)
pip install torch torchvision torchaudio

# 나머지 패키지
pip install -r requirements.txt
```

### CUDA (NVIDIA GPU)

```bash
# PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# PyTorch Geometric
pip install torch-geometric torch-geometric-temporal

# 나머지 패키지
pip install -r requirements.txt
```

### CPU Only

```bash
# PyTorch (CPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 나머지 패키지
pip install -r requirements.txt
```

## ⚠️ 주의사항

### NumPy 버전

`opencv-python`과의 호환성을 위해 NumPy는 `2.3.0` 미만이어야 합니다.

```bash
pip install "numpy>=1.24.0,<2.3.0"
```

### PyTorch Geometric 추가 설치

일부 시스템에서는 추가 설치가 필요할 수 있습니다:

```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
            -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

### Lanelet2 (선택사항)

맵 데이터 처리를 위해 필요하지만 C++ 의존성이 있어 별도 설치가 필요합니다:

```bash
# Conda 사용 권장
conda install -c conda-forge lanelet2

# 또는 소스에서 빌드
```

## 🧪 의존성 확인

### 설치 확인 스크립트

```bash
python scripts/check_system.py
```

### 수동 확인

```python
import torch
import torch_geometric
import pandas
import numpy
import shapely
import cv2
import networkx
import scipy

print("✓ 모든 의존성 설치 완료")
```

## 🔄 의존성 업데이트

### 전체 업데이트

```bash
pip install --upgrade -r requirements.txt
```

### 특정 패키지 업데이트

```bash
pip install --upgrade torch torch-geometric
```

## 📊 의존성 트리

```
Roundabout_AI
├── torch (2.0.0+)
│   ├── numpy
│   └── typing-extensions
├── torch-geometric (2.3.0+)
│   ├── torch
│   ├── xxhash
│   ├── aiohttp
│   ├── psutil
│   └── requests
├── pandas (2.0.0+)
│   ├── numpy
│   └── python-dateutil
├── shapely (2.0.0+)
│   └── numpy
├── opencv-python (4.8.0+)
│   └── numpy<2.3.0
└── networkx (3.1+)
    └── numpy
```

## 🐛 문제 해결

### ImportError 해결

```bash
# 가상환경 재생성
deactivate
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 버전 충돌

```bash
# 충돌하는 패키지 제거 후 재설치
pip uninstall numpy opencv-python
pip install "numpy>=1.24.0,<2.3.0"
pip install opencv-python
```

### PyTorch Geometric 설치 실패

```bash
# CPU 버전
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
            -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

# CUDA 버전
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
            -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

## ✅ 체크리스트

설치 후 다음을 확인하세요:

- [ ] PyTorch 설치 및 GPU/MPS 사용 가능 여부
- [ ] PyTorch Geometric 정상 작동
- [ ] 모든 데이터 처리 라이브러리 import 가능
- [ ] 시각화 라이브러리 작동
- [ ] TensorBoard 실행 가능

## 📚 참고 자료

- [PyTorch 설치 가이드](https://pytorch.org/get-started/locally/)
- [PyTorch Geometric 설치](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
- [NumPy 호환성](https://numpy.org/doc/stable/reference/compatibility.html)
