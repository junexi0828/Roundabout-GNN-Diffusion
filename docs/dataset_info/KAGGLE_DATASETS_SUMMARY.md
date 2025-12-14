# Kaggle 바로 다운로드 가능 데이터셋 요약

## 🔍 검색 결과 분석

### 이미지에서 확인된 Kaggle 데이터셋

1. **Roundabout Aerial Images for Vehicle Detection**
   - 타입: 이미지 데이터셋 (YOLO)
   - 용도: 객체 탐지
   - 궤적 예측: ❌ **부적합**

2. **Spanish Roundabouts Traffic Dataset (YOLO)**
   - 타입: 이미지 데이터셋
   - 용도: 객체 탐지
   - 궤적 예측: ❌ **부적합**

3. **Traffic Aerial Images for Vehicle Detection**
   - 타입: 이미지 데이터셋
   - 용도: 객체 탐지
   - 궤적 예측: ❌ **부적합**

**⚠️ 이들은 모두 이미지 데이터셋입니다!**
궤적 예측 연구에는 **CSV 형태의 궤적 데이터**가 필요합니다.

## ✅ 추천: Argoverse Motion Forecasting Dataset

### Kaggle 링크
https://www.kaggle.com/datasets/fedesoriano/argoverse-motion-forecasting-dataset

### 특징
- ✅ **바로 다운로드 가능** (승인 불필요)
- ✅ 궤적 예측 전용 데이터셋
- ✅ CSV 형태의 궤적 데이터
- ✅ 교차로 및 합류 구간 포함
- ✅ ADE/FDE 평가 가능
- ✅ 320,000개 이상의 시나리오

### 데이터 구조
- 관측: 2초 (20 프레임, 10Hz)
- 예측: 3초 (30 프레임)
- 컬럼: track_id, x, y, vx, vy, timestamp 등

### 한계
- ❌ 회전교차로 전용 아님 (일반 교차로/합류 구간)
- ❌ 미국 데이터만 (국가별 비교 불가)

### 다운로드 방법

```bash
# 1. Kaggle CLI 설치
pip install kaggle

# 2. API 토큰 설정
# Kaggle 계정 > Settings > API > Create New Token
# ~/.kaggle/kaggle.json 파일에 저장

# 3. 데이터셋 다운로드
kaggle datasets download -d fedesoriano/argoverse-motion-forecasting-dataset

# 또는 스크립트 사용
python scripts/download_argoverse.py
```

## 대안 데이터셋

### Waymo Open Dataset (Kaggle)
- 링크: https://www.kaggle.com/datasets/google/waymo-open-dataset
- 특징: 대규모 궤적 데이터
- 한계: TFRecord 포맷 (변환 필요)

## 결론

**즉시 사용 가능한 최선의 선택: Argoverse Motion Forecasting Dataset**

1. ✅ 바로 다운로드 가능
2. ✅ 궤적 예측 전용
3. ✅ 모델 개발 및 검증에 적합
4. ⚠️ 회전교차로는 아니지만 상호작용 예측 연구 가능

**다음 단계**:
1. Argoverse 데이터셋 다운로드
2. 데이터 어댑터 개발 (Argoverse → 프로젝트 포맷)
3. 모델 학습 및 검증
4. 향후 INTERACTION/rounD 승인 후 회전교차로 특화 분석
