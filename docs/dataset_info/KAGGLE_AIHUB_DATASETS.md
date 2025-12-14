# Kaggle & AI Hub 바로 다운로드 가능한 데이터셋

## ⚠️ 중요: 이미지 데이터셋 vs 궤적 데이터셋

Kaggle에서 검색된 대부분의 데이터셋은 **이미지 데이터셋**입니다:

- Roundabout Aerial Images for Vehicle Detection
- Spanish Roundabouts Traffic Dataset (YOLO)
- Roundabout Aerial Images YOLO data

이들은 **객체 탐지(Object Detection)**용이며, **궤적 예측(Trajectory Prediction)** 연구에는 부적합합니다.

## 궤적 예측에 적합한 Kaggle 데이터셋

### 1. Argoverse Motion Forecasting Dataset

**Kaggle 링크**: https://www.kaggle.com/datasets/fedesoriano/argoverse-motion-forecasting-dataset

**특징**:

- ✅ 궤적 예측 전용 데이터셋
- ✅ 차량 궤적 데이터 포함 (CSV)
- ✅ 교차로 및 합류 구간 포함
- ✅ 바로 다운로드 가능 (승인 불필요)
- ✅ ADE/FDE 평가 가능

**한계**:

- ❌ 회전교차로 전용 아님
- ❌ 미국 데이터만 (국가별 비교 불가)

**데이터 구조**:

```
- track_id, x, y, vx, vy, timestamp 등
- 관측 2초, 예측 3초
```

### 2. Waymo Open Dataset (일부)

**Kaggle 링크**: https://www.kaggle.com/datasets/google/waymo-open-dataset

**특징**:

- ✅ 대규모 궤적 데이터
- ✅ 다양한 도로 환경
- ✅ 바로 다운로드 가능

**한계**:

- ❌ 회전교차로 전용 아님
- ❌ 데이터 포맷 변환 필요 (TFRecord)
- ❌ 미국 데이터만

### 3. nuScenes Prediction Dataset

**Kaggle 링크**: https://www.kaggle.com/datasets/nuscenes/nuscenes

**특징**:

- ✅ 궤적 예측 데이터 포함
- ✅ 다양한 시나리오

**한계**:

- ❌ 회전교차로 전용 아님
- ❌ 승인 필요할 수 있음

## AI Hub 데이터셋

### 자율주행 관련 데이터셋 검색

AI Hub에서 "자율주행", "교통", "차량 궤적" 등으로 검색:

**URL**: https://aihub.or.kr/

**검색 키워드**:

- 자율주행
- 교통 데이터
- 차량 궤적
- 도로 교통

**예상 결과**:

- 교통 흐름 데이터
- 차량 탐지 데이터 (이미지)
- 궤적 데이터 (있는 경우)

## 대안 전략

### 전략 1: Argoverse 활용 (즉시 가능)

```python
# Argoverse 데이터셋 구조
# - 관측: 2초 (20 프레임, 10Hz)
# - 예측: 3초 (30 프레임)
# - 교차로 및 합류 구간 포함

# 회전교차로는 아니지만, 상호작용 예측 연구 가능
```

**장점**:

- ✅ 바로 다운로드 가능
- ✅ 궤적 예측 전용
- ✅ 모델 검증 가능

**단점**:

- ❌ 회전교차로 아님
- ❌ 국가별 비교 불가

### 전략 2: 합성 데이터 생성

실제 데이터가 없을 경우:

1. **시뮬레이션 데이터 생성**

   - SUMO, CARLA 등 시뮬레이터 활용
   - 회전교차로 시나리오 생성
   - 다양한 국가별 운전 패턴 모델링

2. **데이터 증강**
   - 기존 샘플 데이터 확장
   - 변환 및 노이즈 추가

### 전략 3: 하이브리드 접근

1. **단기**: Argoverse로 모델 개발 및 검증
2. **중기**: 시뮬레이션 데이터로 회전교차로 특화
3. **장기**: INTERACTION/rounD 승인 후 실제 데이터 분석

## 추천 데이터셋 (우선순위)

### 1순위: Argoverse Motion Forecasting (Kaggle)

**이유**:

- ✅ 바로 다운로드 가능
- ✅ 궤적 예측 전용
- ✅ 연구용으로 충분한 데이터

**다운로드**:

```bash
# Kaggle CLI 사용
pip install kaggle
kaggle datasets download -d fedesoriano/argoverse-motion-forecasting-dataset
```

### 2순위: Waymo Open Dataset (Kaggle)

**이유**:

- ✅ 대규모 데이터
- ✅ 다양한 환경

**주의**:

- 데이터 포맷 변환 필요
- 회전교차로 비중 낮음

### 3순위: 시뮬레이션 데이터 생성

**도구**:

- SUMO (교통 시뮬레이션)
- CARLA (자율주행 시뮬레이션)

## 다음 단계

1. **Argoverse 데이터셋 다운로드 및 검증**

   ```bash
   # Kaggle에서 다운로드
   # 데이터 구조 확인
   python scripts/verify_dataset.py --dataset argoverse
   ```

2. **데이터 어댑터 개발**

   - Argoverse 포맷 → 프로젝트 포맷 변환
   - 전처리 파이프라인 적용

3. **모델 학습 및 검증**
   - Argoverse 데이터로 모델 학습
   - 회전교차로 특화는 향후 추가

## 참고 링크

- **Argoverse Kaggle**: https://www.kaggle.com/datasets/fedesoriano/argoverse-motion-forecasting-dataset
- **Waymo Kaggle**: https://www.kaggle.com/datasets/google/waymo-open-dataset
- **AI Hub**: https://aihub.or.kr/
- **SUMO 시뮬레이션**: https://www.eclipse.org/sumo/
