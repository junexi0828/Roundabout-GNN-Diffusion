# 데이터셋 정보

이 디렉토리는 프로젝트에서 사용하는 데이터셋 관련 정보와 가이드를 포함합니다.

## 📊 데이터셋 가이드

### 주요 데이터셋

1. **[SDD_DATASET_GUIDE.md](SDD_DATASET_GUIDE.md)**
   - Stanford Drone Dataset 사용 가이드
   - Death Circle 데이터 다운로드 및 전처리

### 데이터셋 상태 및 검토

2. **[DATASET_STATUS.md](DATASET_STATUS.md)** - 현재 데이터셋 상태
3. **[DATASET_COMPLIANCE_REVIEW.md](DATASET_COMPLIANCE_REVIEW.md)** - 데이터셋 규정 검토

### 대안 데이터셋

4. **[DATASET_ALTERNATIVES.md](DATASET_ALTERNATIVES.md)** - 대안 데이터셋 목록
5. **[데이터셋 대안_ 즉시 다운로드 가능한 궤적 데이터.md](데이터셋%20대안_%20즉시%20다운로드%20가능한%20궤적%20데이터.md)** - 즉시 사용 가능한 데이터셋 상세 정보

### Kaggle & AI Hub

6. **[KAGGLE_AIHUB_DATASETS.md](KAGGLE_AIHUB_DATASETS.md)** - Kaggle 및 AI Hub 데이터셋
7. **[KAGGLE_DATASETS_SUMMARY.md](KAGGLE_DATASETS_SUMMARY.md)** - Kaggle 데이터셋 요약

## 🎯 주요 데이터셋

### Stanford Drone Dataset (SDD)
- **장소**: Death Circle (회전교차로)
- **에이전트**: 차량, 보행자, 자전거, 스케이트보더, 카트, 버스
- **형식**: 픽셀 좌표 → 호모그래피 변환 필요
- **비디오**: video0 ~ video4 (5개)

### 대안 데이터셋
- INTERACTION Dataset
- Argoverse
- nuScenes
- Waymo Open Dataset

## 📥 데이터 준비

1. SDD 다운로드: `python scripts/download_sdd_deathcircle.py`
2. 호모그래피 추정: `python scripts/auto_homography_estimation.py`
3. 데이터 검증: `python scripts/verify_sdd_data.py`
