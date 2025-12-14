# SDD Death Circle 데이터셋 가이드

## ✅ 다운로드 및 변환 완료

### 데이터 요약

- **데이터셋**: Stanford Drone Dataset - Death Circle
- **시나리오**: 회전교차로 (Roundabout)
- **비디오 수**: 5개 (video0 ~ video4)
- **총 데이터**: 약 1,278,921행, 2,789개 트랙
- **에이전트 타입**: Biker, Pedestrian, Skater, Cart, Car, Bus
- **다운로드 방식**: GitHub에서 바로 다운로드 (승인 불필요)

### 데이터 위치

- **원본**: `data/sdd/deathCircle/`
- **변환된 데이터**: `data/sdd/converted/`
  - `video0_converted.csv` (394,497행, 692개 트랙)
  - `video1_converted.csv` (512,466행, 1,144개 트랙)
  - `video2_converted.csv` (9,387행, 35개 트랙)
  - `video3_converted.csv` (349,270행, 856개 트랙)
  - `video4_converted.csv` (13,301행, 62개 트랙)

## 데이터 포맷

### 변환된 CSV 컬럼

- `track_id`: 트랙 식별자
- `frame_id`: 프레임 번호
- `timestamp_ms`: 타임스탬프 (밀리초)
- `agent_type`: 에이전트 타입 (biker, pedestrian, skater, cart, car, bus)
- `x`, `y`: 위치 (미터 단위, 호모그래피 변환 적용)
- `vx`, `vy`: 속도 (m/s)
- `psi_rad`: 헤딩 각도 (현재 0.0, 추후 계산 가능)
- `length`, `width`: 차량 크기 (미터)

## 사용 방법

### 1. 데이터 검증

```bash
python scripts/verify_sdd_data.py
```

### 2. 전처리

```bash
# 변환된 데이터를 프로젝트 전처리 파이프라인에 적용
python src/data_processing/preprocessor.py data/sdd/converted/video0_converted.csv
```

### 3. 모델 학습

```bash
# SDD 데이터로 학습
python src/training/train.py --data_dir data/processed --dataset sdd
```

## 시나리오 타입 분석

SDD Death Circle 데이터에는 명시적인 시나리오 타입 라벨이 없지만, 데이터 특성으로 추론 가능:

### 분석 방법

1. **차량 밀도 계산**

   - 프레임별 차량 수
   - 평균 차간 거리

2. **상호작용 강도 측정**

   - 진입/진출 빈도
   - 양보 행동 빈도

3. **속도 패턴 분석**
   - 평균 속도
   - 가속도 분포

## 한계 및 주의사항

### 한계

- ❌ **국가별 비교 불가**: 미국 스탠포드 캠퍼스 데이터만
- ❌ **시나리오 타입 라벨 없음**: 자동 분류 필요
- ❌ **이기종 에이전트**: 보행자, 자전거 등 포함 (차량만 필터링 가능)

### 주의사항

- 픽셀 좌표를 미터로 변환 (호모그래피 행렬 사용)
- 프레임 레이트: 약 2 FPS (0.5초 간격)
- lost/occluded 플래그 확인 필요

## 다음 단계

1. ✅ 데이터 다운로드 완료
2. ✅ 데이터 변환 완료
3. ⏳ 전처리 파이프라인 적용
4. ⏳ 시나리오 타입 자동 분류 모듈 개발
5. ⏳ 모델 학습 및 평가

## 참고 자료

- SDD 공식 사이트: https://cvgl.stanford.edu/projects/uav_data/
- GitHub 미러: https://github.com/flclain/StanfordDroneDataset
- Trajectron++ 리포지토리: https://github.com/StanfordASL/Trajectron-plus-plus
