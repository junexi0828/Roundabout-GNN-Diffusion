# Colab 완전 자동화 가이드

## 🚀 빠른 시작

### Git Clone 후 바로 실행

```python
# Colab 노트북에서
!git clone https://github.com/your-repo/Roundabout_AI.git
%cd Roundabout_AI

# Step 1: SDD 데이터 다운로드 및 전처리
!python scripts/colab/colab_download_and_preprocess.py \
    --output_dir data/sdd/deathCircle \
    --converted_dir data/sdd/converted

# Step 2: 모델 학습
!python scripts/colab/colab_auto_pipeline.py --mode fast
```

## 📋 스크립트 구조

### Colab 관련

- `scripts/colab/colab_download_and_preprocess.py` - 데이터 다운로드 및 전처리
- `scripts/colab/colab_auto_pipeline.py` - 자동화 파이프라인
- `scripts/colab/colab_setup.py` - 환경 설정

### 데이터 관련

- `scripts/data/download_sdd_deathcircle.py` - SDD 데이터 다운로드
- `scripts/data/preprocess_sdd.py` - 데이터 전처리
- `scripts/data/auto_homography_estimation.py` - 호모그래피 추정

### 학습 관련

- `scripts/training/train_a3tgcn.py` - A3TGCN 학습
- `scripts/training/fast_train.py` - 빠른 학습

### 평가 관련

- `scripts/evaluation/evaluate_mid.py` - 모델 평가
- `scripts/evaluation/compare_baselines.py` - 베이스라인 비교

## 📊 데이터 최적화

**Colab에서 원본 데이터 다운로드 + 전처리** (Drive 업로드 불필요)

- 속도: 5-10배 빠름
- 편의성: 수동 작업 불필요

## ✅ 검증

로컬에서 확인:

```bash
python scripts/utils/check_colab_readiness.py
```

## 📊 자동 실행 프로세스

```
[1/9] 환경 설정
  ├─ 라이브러리 설치 (PyTorch, PyG 등)
  └─ GPU 확인

[2/9] GitHub 저장소 클론
  └─ 프로젝트 코드 다운로드

[3/9] Google Drive 마운트
  └─ 데이터 접근

[4/9] 데이터 준비
  └─ SDD 데이터 확인

[5/9] 데이터 전처리
  ├─ Homography 추정
  ├─ 궤적 추출
  └─ 윈도우 생성

[6/9] 모델 학습 ⚡
  ├─ MID 모델 생성
  ├─ 학습 시작
  └─ 체크포인트 저장

[7/9] 모델 평가
  ├─ ADE/FDE 계산
  ├─ Diversity 계산
  └─ Collision Rate 계산

[8/9] 결과 시각화
  ├─ 학습 곡선
  ├─ 샘플 궤적
  └─ 평가 결과

[9/9] 결과 저장
  └─ Google Drive에 저장
```

## 📁 결과물 구조

학습 완료 후 다음 결과를 얻습니다:

```
results/
├── checkpoints/
│   └── best_model.pth          # 학습된 모델
│
├── metrics/
│   ├── evaluation_results.json # ADE, FDE, Diversity
│   └── comparison_table.csv     # 베이스라인 비교
│
├── visualizations/
│   ├── training_curves.png      # Loss 그래프
│   ├── sample_trajectories.png # 20개 예측 궤적
│   └── evaluation_results.png  # 평가 결과
│
└── paper_ready/
    ├── results_table.tex        # 논문용 표
    └── figures/                 # 논문용 그림
```
