# 데이터셋 상태 확인 보고서

## SDD Death Circle 데이터셋

### ✅ 데이터 준비 완료

#### 1. 원본 어노테이션 파일
- **위치**: `data/sdd/deathCircle/`
- **비디오 수**: 5개 (video0 ~ video4)
- **총 데이터 행 수**: 약 2,398,316행
  - video0: 709,789행
  - video1: 883,078행
  - video2: 15,085행
  - video3: 760,984행
  - video4: 29,380행

#### 2. 변환된 데이터 (픽셀 → 미터)
- **위치**: `data/sdd/converted/`
- **파일**: `video0_converted.csv` ~ `video4_converted.csv`
- **포맷**: CSV (track_id, frame_id, timestamp_ms, agent_type, x, y, vx, vy, psi_rad, length, width)
- **에이전트 타입**: pedestrian, car, biker, skater, cart, bus 등

#### 3. 호모그래피 행렬
- **위치**: `data/sdd/homography/H.txt`
- **형태**: 3x3 행렬
- **용도**: 픽셀 좌표 → 미터 좌표 변환

### 데이터 구조 예시

**원본 어노테이션 포맷**:
```
track_id xmin ymin xmax ymax frame lost occluded generated "label"
```

**변환된 CSV 포맷**:
```csv
track_id,frame_id,timestamp_ms,agent_type,x,y,vx,vy,psi_rad,length,width
0,9491,4745500,pedestrian,17.989,40.4,0.0,0.0,0.0,0.579,1.011
```

## 깃 저장소 상태

### ✅ 클론된 저장소

1. **INTERACTION Dataset Repository**
   - 위치: `data/interaction-dataset-repo/`
   - 상태: ✅ 클론 완료

2. **Drone Dataset Tools**
   - 위치: `data/drone-dataset-tools/`
   - 상태: ✅ 클론 완료

## 다음 단계

### 1. 데이터 검증
```bash
source venv/bin/activate
python scripts/verify_sdd_data.py
```

### 2. 데이터 전처리
```python
from src.integration.sdd_data_adapter import SDDDataAdapter

adapter = SDDDataAdapter(homography_path='data/sdd/homography/H.txt')
# 변환된 데이터는 이미 준비되어 있음
```

### 3. 모델 학습 준비
- 전처리된 데이터 확인
- 학습/검증/테스트 분할
- 데이터 로더 테스트

## 데이터 통계 (예상)

- **총 트랙 수**: 약 2,000~3,000개
- **에이전트 타입**: 6종 (Pedestrian, Car, Biker, Skater, Cart, Bus)
- **시간 범위**: 약 1,000~2,000초 (비디오별 상이)
- **프레임 레이트**: 30fps (다운샘플링 가능)

## 주의사항

1. **가상환경 활성화 필요**: 모든 스크립트 실행 전 `source venv/bin/activate` 필요
2. **호모그래피 검증**: 변환 정확도 확인 필요
3. **데이터 불균형**: 비디오별 데이터 양이 크게 다름 (video2, video4는 적음)

