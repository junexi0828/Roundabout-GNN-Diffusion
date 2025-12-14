# Scripts 디렉토리 구조

## 📁 디렉토리 구조

```
scripts/
├── colab/          # Colab 자동화 스크립트
├── data/           # 데이터 다운로드 및 전처리
├── training/       # 모델 학습
├── evaluation/     # 모델 평가 및 비교
└── utils/          # 유틸리티 및 검증
```

## 📋 스크립트 목록

### Colab (`scripts/colab/`)

- `colab_download_and_preprocess.py` - SDD 데이터 다운로드 및 전처리
- `colab_auto_pipeline.py` - 완전 자동화 파이프라인
- `colab_full_pipeline.py` - MID 전체 파이프라인
- `colab_one_click.py` - 원클릭 실행
- `colab_setup.py` - 환경 설정

### Data (`scripts/data/`)

- `download_sdd_deathcircle.py` - SDD Death Circle 다운로드
- `preprocess_sdd.py` - SDD 데이터 전처리
- `auto_homography_estimation.py` - 호모그래피 자동 추정
- `verify_sdd_data.py` - SDD 데이터 검증
- `verify_data_consistency.py` - 데이터 일관성 검증
- `verify_dataset.py` - 데이터셋 검증

### Training (`scripts/training/`)

- `train_a3tgcn.py` - A3TGCN 모델 학습
- `train_mid.py` - MID 모델 학습
- `fast_train.py` - 빠른 학습 (최적화 버전)

### Evaluation (`scripts/evaluation/`)

- `evaluate_mid.py` - MID 모델 평가
- `compare_baselines.py` - 베이스라인 모델 비교

### Utils (`scripts/utils/`)

- `check_colab_readiness.py` - Colab 실행 준비 상태 확인
- `check_dependencies.py` - 의존성 확인
- `check_system.py` - 시스템 확인
- `setup_local.py` - 로컬 환경 설정
- `test_attention_extraction.py` - 어텐션 추출 테스트
- `test_attention_with_sample_data.py` - 샘플 데이터로 어텐션 테스트
- `visualize_attention_sample.py` - 어텐션 시각화
- `visualize_results.py` - 결과 시각화

## 🚀 사용 예시

### Colab에서 실행

```python
# 데이터 다운로드 및 전처리
!python scripts/colab/colab_download_and_preprocess.py

# 모델 학습
!python scripts/training/train_a3tgcn.py --config configs/a3tgcn_config.yaml
```

### 로컬에서 실행

```bash
# 데이터 검증
python scripts/data/verify_sdd_data.py

# 빠른 학습
python scripts/training/fast_train.py --batch_size 32
```
