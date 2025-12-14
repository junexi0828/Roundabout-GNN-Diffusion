# 의존성 요약

## ✅ 필수 의존성 (20개)

### Core Deep Learning
- ✅ torch>=2.0.0
- ✅ torchvision>=0.15.0
- ✅ torchaudio>=2.0.0

### Graph Neural Networks
- ✅ torch-geometric>=2.3.0
- ⚠️ torch-geometric-temporal>=0.54.0 (선택사항, A3TGCN 사용 시)

### Data Processing
- ✅ pandas>=2.0.0
- ✅ numpy>=1.24.0,<2.3.0
- ✅ scipy>=1.10.0

### Graph Processing
- ✅ networkx>=3.1

### Geometric Operations
- ✅ shapely>=2.0.0

### Computer Vision
- ✅ opencv-python>=4.8.0

### Visualization
- ✅ matplotlib>=3.7.0
- ✅ seaborn>=0.12.0

### Utilities
- ✅ tqdm>=4.65.0
- ✅ pyyaml>=6.0
- ✅ tensorboard>=2.13.0

### Machine Learning
- ✅ scikit-learn>=1.3.0

### PyTorch Geometric 의존성
- ✅ xxhash>=3.0.0
- ✅ aiohttp>=3.8.0
- ✅ psutil>=5.8.0
- ✅ requests>=2.28.0

## ⚠️ 선택적 의존성

- torch-geometric-temporal (A3TGCN 사용 시)
- jupyter, ipykernel, notebook (노트북 사용 시)
- trajdata (Trajectron++ 사용 시)

## 📦 설치 명령

```bash
# 기본 설치
pip install -r requirements.txt

# 최소 설치
pip install -r requirements-minimal.txt

# 개발 환경
pip install -r requirements-dev.txt
```

## ✅ 확인 명령

```bash
python scripts/check_dependencies.py
```

