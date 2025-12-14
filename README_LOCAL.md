# 로컬 실행 가이드

로컬 환경(MacBook, Linux, Windows)에서 HSG-Diffusion 모델을 학습하는 가이드입니다.

## 빠른 시작 (자동화 파이프라인)

**단 한 줄로 전체 파이프라인 실행!**

```bash
# 1. 가상환경 활성화
source venv/bin/activate

# 2. 자동 파이프라인 실행 (Fast 모드, 2-3시간)
python scripts/local_auto_pipeline.py --mode fast

# 또는 Full 모드 (6-8시간)
python scripts/local_auto_pipeline.py --mode full
```

이 스크립트는 다음을 자동으로 수행합니다:
1. ✅ 환경 확인 (PyTorch, PyTorch Geometric 등)
2. ✅ 데이터 확인 (있으면 사용, 없으면 안내)
3. ✅ 데이터 전처리 (필요시 자동 실행)
4. ✅ HSG-Diffusion 학습
5. ✅ 결과 시각화

---

## 초기 설정 (처음 한 번만)

### 1. 저장소 클론
```bash
git clone https://github.com/junexi0828/Roundabout-GNN-Diffusion.git
cd Roundabout-GNN-Diffusion
```

### 2. 가상환경 설정
```bash
# 가상환경 생성 및 패키지 설치
./setup.sh

# 또는 수동으로
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. 데이터 준비

**옵션 A: 이미 데이터가 있는 경우**
```bash
# 데이터를 다음 위치에 배치:
# data/sdd/converted/*.csv  (전처리된 데이터)
# 또는
# data/sdd/deathCircle/     (원본 SDD 데이터)
```

**옵션 B: Colab에서 다운로드**
```python
# Colab에서 실행
!python scripts/colab/colab_auto_pipeline.py --mode fast
# Google Drive에서 data/ 디렉토리를 로컬로 복사
```

---

## 실행 모드

### Fast 모드 (권장, 2-3시간)
```bash
python scripts/local_auto_pipeline.py --mode fast
```
- 30% 데이터 샘플링
- 50 에폭
- 빠른 검증용

### Full 모드 (6-8시간)
```bash
python scripts/local_auto_pipeline.py --mode full
```
- 전체 데이터 사용
- 100 에폭
- 최고 성능

---

## 수동 실행 (단계별)

자동화 스크립트 대신 개별 단계를 실행하려면:

### 1. 데이터 전처리
```bash
python -c "
from src.data_processing.sdd_adapter import SDDAdapter
from pathlib import Path

adapter = SDDAdapter('data/sdd/deathCircle')
adapter.convert_all_videos('data/sdd/converted')
"
```

### 2. 모델 학습
```bash
python scripts/training/train_mid.py \
    --config configs/mid_config_fast.yaml \
    --data_dir data/sdd/converted
```

### 3. 결과 시각화
```bash
python scripts/utils/visualize_results.py
```

---

## 결과 확인

### 체크포인트
```bash
ls checkpoints/mid_fast/
# best_model.pth - 최고 성능 모델
# latest_model.pth - 최신 체크포인트
```

### TensorBoard
```bash
tensorboard --logdir runs/mid_fast
# 브라우저에서 http://localhost:6006 열기
```

### 시각화 결과
```bash
open results/visualizations/
# training_curves.png - 학습 곡선
# sample_trajectories.png - 예측 궤적 샘플
```

---

## 성능 비교 (Colab vs 로컬)

| 환경 | GPU | Fast 모드 시간 | Full 모드 시간 |
|------|-----|--------------|--------------|
| Colab | Tesla T4 | 2-3시간 | 6-8시간 |
| MacBook Air M2 | MPS | 4-6시간 | 12-16시간 |
| CPU Only | - | 10-15시간 | 30-40시간 |

---

## 문제 해결

### 패키지 누락
```bash
# 환경 확인
python scripts/utils/setup_local.py

# 또는 수동 설치
pip install torch torch-geometric pandas numpy matplotlib
```

### 메모리 부족
```bash
# 배치 크기 줄이기
# configs/mid_config_fast.yaml 수정:
# batch_size: 64 → 32
```

### Apple Silicon (M1/M2) GPU 사용
```python
# MPS 사용 확인
import torch
print(torch.backends.mps.is_available())  # True여야 함
```

---

## 고급 옵션

### 커스텀 설정 파일
```bash
python scripts/local_auto_pipeline.py \
    --mode fast \
    --data-dir /path/to/custom/data
```

### 백그라운드 실행
```bash
nohup python scripts/local_auto_pipeline.py --mode fast \
    > logs/training.log 2>&1 &

# 진행 상황 모니터링
tail -f logs/training.log
```

---

## Colab과 병렬 실행

**권장 워크플로우:**

1. **Colab**: GPU로 빠르게 학습
2. **로컬**: CPU/MPS로 동일 실험 재현

```bash
# Colab 실행 후 로컬에서도 실행
python scripts/local_auto_pipeline.py --mode fast

# 결과 비교
python scripts/evaluation/compare_baselines.py \
    --mid_checkpoint checkpoints/mid_fast/best_model.pth \
    --data_dir data/sdd/converted
```

---

## 다음 단계

학습 완료 후:

1. **결과 분석**
   ```bash
   jupyter notebook notebooks/analyze_results.ipynb
   ```

2. **평가**
   ```bash
   python scripts/evaluation/evaluate_mid.py \
       --checkpoint checkpoints/mid_fast/best_model.pth
   ```

3. **추론**
   ```python
   from src.models.mid_integrated import create_fully_integrated_mid

   model = create_fully_integrated_mid()
   model.load_state_dict(torch.load('checkpoints/mid_fast/best_model.pth'))
   ```

---

**문의사항**: GitHub Issues 또는 README.md 참조
