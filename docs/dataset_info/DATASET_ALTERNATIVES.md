# 데이터셋 대안 분석

## 최종 데이터셋 선정: SDD Death Circle

### 선정 과정 요약

본 연구는 회전교차로 상호작용 예측을 위해 여러 데이터셋을 검토하였으나, 승인 절차로 인해 즉시 사용 가능한 데이터셋을 확보하는 데 어려움을 겪었다. 최종적으로 **Stanford Drone Dataset (SDD)의 'Death Circle'**을 선택하여 연구를 진행하기로 결정하였다.

### 시도한 데이터셋 및 결과

| 데이터셋                   | 상태             | 문제점                                     |
| -------------------------- | ---------------- | ------------------------------------------ |
| **INTERACTION Dataset**    | 승인 대기 중     | USA/CHN/DEU 국가별 비교 가능하나 승인 필요 |
| **rounD Dataset**          | 승인 대기 중     | 회전교차로 전용이지만 승인 절차 필요       |
| **Argoverse (Kaggle)**     | 다운로드 가능    | 회전교차로 포함되지 않음, 일반 교차로만    |
| **Kaggle 이미지 데이터셋** | 다운로드 가능    | 궤적 데이터가 아닌 이미지 데이터 (부적합)  |
| **SDD Death Circle**       | ✅ **최종 선정** | 승인 불필요, GitHub에서 즉시 다운로드 가능 |

### 최종 선정: SDD Death Circle

**선정 이유**:

1. ✅ **승인 불필요**: GitHub 미러를 통해 `git clone`만으로 즉시 다운로드 가능
2. ✅ **회전교차로 포함**: 비신호 회전교차로 시나리오 포함
3. ✅ **이기종 에이전트**: 차량, 보행자, 자전거 등 다양한 상호작용 패턴
4. ✅ **연구 적합성**: GNN 기반 상호작용 모델링에 최적화된 데이터 구조

**데이터 규모**:

- 5개 비디오 (video0~video4)
- 총 1,278,921행, 2,789개 트랙
- 에이전트 타입: Car, Bus, Biker, Pedestrian, Skater, Cart

**한계**:

- ❌ 국가별 비교 불가 (미국 스탠포드 캠퍼스만)
- ❌ 시나리오 타입 라벨 없음 (자동 분류 필요)

**다운로드 및 변환 완료**:

- 원본 데이터: `data/sdd/deathCircle/`
- 변환된 데이터: `data/sdd/converted/` (프로젝트 표준 포맷)

## 현재 상황

### INTERACTION Dataset

- **상태**: 다운로드 불가 (승인 대기 중)
- **필요한 데이터**:
  - DR_USA_Roundabout_FT (미국)
  - DR_CHN_Roundabout_LN (중국)
  - DR_DEU_Roundabout_OF (독일)
- **특징**: 국가별 운전 문화 차이 분석 가능

### 샘플 데이터 한계

- `.TestScenarioForScripts`: 단순 테스트용
- 시나리오 타입 정보 없음 (Normal Merging, Dense Traffic, Aggressive Entry 등)
- 국가별 정보 없음
- 실제 상호작용 패턴 부재

## 대안 데이터셋

### 1. rounD Dataset (승인대기중)

- **상태**: 다운로드 불가 (승인 대기 중)

**출처**: RWTH Aachen University
**웹사이트**: https://www.drone-dataset.com/round-dataset
**논문**: "The rounD Dataset: A Drone Dataset of Road User Trajectories at Roundabouts in Germany"

**특징**:

- ✅ **회전교차로 전용 데이터셋**
- ✅ 독일 회전교차로 데이터
- ✅ 드론 기반 조감도 데이터 (INTERACTION과 유사)
- ✅ CSV 포맷 (INTERACTION과 호환 가능)
- ✅ 다운로드 가능 (공개 데이터셋)

**한계**:

- ❌ 국가별 비교 불가 (독일만)
- ❌ 시나리오 타입 라벨링 여부 불확실

**다운로드**:

```bash
# 공식 웹사이트에서 다운로드
# https://www.drone-dataset.com/round-dataset
```

### 2. inD Dataset

**출처**: RWTH Aachen University
**웹사이트**: https://www.drone-dataset.com/ind-dataset

**특징**:

- ✅ 독일 교차로 데이터
- ✅ 다양한 교차로 유형
- ✅ 드론 기반 조감도 데이터

**한계**:

- ❌ 회전교차로 전용 아님
- ❌ 국가별 비교 불가

### 3. exiD Dataset

**출처**: RWTH Aachen University
**웹사이트**: https://www.drone-dataset.com/exid-dataset

**특징**:

- ✅ 고속도로 상호작용 시나리오
- ✅ 고밀도 교통 상황

**한계**:

- ❌ 회전교차로 아님
- ❌ 고속도로 합류 구간 중심

## 데이터셋 비교

| 데이터셋    | 회전교차로 | 국가        | 시나리오 타입 | 다운로드  |
| ----------- | ---------- | ----------- | ------------- | --------- |
| INTERACTION | ✅         | USA/CHN/DEU | ❓            | 승인 필요 |
| rounD       | ✅         | DEU         | ❓            | ✅ 공개   |
| inD         | ❌         | DEU         | ❓            | ✅ 공개   |
| exiD        | ❌         | DEU         | ❓            | ✅ 공개   |

## 권장 사항

### 옵션 1: rounD Dataset 사용 (단기)

- **장점**: 즉시 사용 가능, 회전교차로 전용
- **단점**: 국가별 비교 불가
- **활용**: 독일 회전교차로 특성 분석, 모델 검증

### 옵션 2: INTERACTION 승인 대기 (장기)

- **장점**: 국가별 비교 가능, 다양한 시나리오
- **단점**: 승인 대기 시간 불확실
- **활용**: 최종 연구 목표 달성

### 옵션 3: 하이브리드 접근

1. **현재**: rounD로 모델 개발 및 검증
2. **향후**: INTERACTION 승인 후 국가별 비교 분석

## 시나리오 타입 추론 방법

실제 데이터셋에 시나리오 타입 라벨이 없더라도, 데이터 특성으로 추론 가능:

### Normal Merging

- 차량 밀도: 중간
- 속도 변화: 완만
- 진입 각도: 표준

### Dense Traffic

- 차량 밀도: 높음 (임계값: 평균 거리 < 10m)
- 속도: 낮음
- 상호작용 빈도: 높음

### Aggressive Entry

- 가속도: 높음 (임계값: > 2 m/s²)
- 진입 속도: 높음
- 간격 수락: 짧음 (Gap acceptance < 3초)

## 구현 계획

### 1단계: rounD 데이터셋 다운로드 및 검증

```bash
# 데이터셋 다운로드 후
python scripts/verify_dataset.py --dataset round
```

### 2단계: 시나리오 타입 자동 분류 모듈 개발

- 차량 밀도 계산
- 상호작용 강도 측정
- 진입 패턴 분석

### 3단계: 성능 분석 스크립트 개발

- ADE/FDE 계산
- 시나리오별 통계
- 국가별 비교 (INTERACTION 확보 시)

## Kaggle 바로 다운로드 가능 데이터셋

### ⚠️ 중요: 이미지 vs 궤적 데이터

Kaggle에서 검색된 대부분의 데이터셋은 **이미지 데이터셋**입니다:

- Roundabout Aerial Images for Vehicle Detection (이미지)
- Spanish Roundabouts Traffic Dataset (YOLO 이미지)
- Traffic Aerial Images (이미지)

**이들은 객체 탐지용이며, 궤적 예측 연구에는 부적합합니다.**

### ✅ 추천: Argoverse Motion Forecasting Dataset

**Kaggle 링크**: https://www.kaggle.com/datasets/fedesoriano/argoverse-motion-forecasting-dataset

**특징**:

- ✅ **바로 다운로드 가능** (승인 불필요)
- ✅ 궤적 예측 전용 데이터셋
- ✅ CSV 형태의 궤적 데이터
- ✅ 교차로 및 합류 구간 포함
- ✅ ADE/FDE 평가 가능

**데이터 구조**:

- 관측: 2초 (20 프레임, 10Hz)
- 예측: 3초 (30 프레임)
- 컬럼: track_id, x, y, vx, vy, timestamp 등

**한계**:

- ❌ 회전교차로 전용 아님 (일반 교차로/합류 구간)
- ❌ 미국 데이터만 (국가별 비교 불가)

**다운로드 방법**:

```bash
# Kaggle CLI 설치
pip install kaggle

# API 토큰 설정 (Kaggle 계정 필요)
# ~/.kaggle/kaggle.json 파일 생성

# 데이터셋 다운로드
kaggle datasets download -d fedesoriano/argoverse-motion-forecasting-dataset
unzip argoverse-motion-forecasting-dataset.zip
```

### 대안: Waymo Open Dataset (Kaggle)

**Kaggle 링크**: https://www.kaggle.com/datasets/google/waymo-open-dataset

**특징**:

- ✅ 바로 다운로드 가능
- ✅ 대규모 궤적 데이터

**한계**:

- ❌ TFRecord 포맷 (변환 필요)
- ❌ 회전교차로 비중 낮음

## 최종 권장 사항

### 즉시 사용 가능 (우선순위)

1. **Argoverse Motion Forecasting (Kaggle)** ⭐⭐⭐

   - 바로 다운로드 가능
   - 궤적 예측 전용
   - 모델 개발 및 검증에 적합

2. **Waymo Open Dataset (Kaggle)** ⭐⭐
   - 대규모 데이터
   - 포맷 변환 필요

### 승인 대기 중

3. **rounD Dataset** ⭐⭐⭐

   - 회전교차로 전용
   - 승인 필요

4. **INTERACTION Dataset** ⭐⭐⭐
   - 국가별 비교 가능
   - 승인 필요

## 참고 자료

- **SDD Death Circle (최종 선정)**: https://github.com/flclain/StanfordDroneDataset
- rounD Dataset: https://www.drone-dataset.com/round-dataset
- drone-dataset-tools: https://github.com/ika-rwth-aachen/drone-dataset-tools
- INTERACTION Dataset: http://interaction-dataset.com/
- Argoverse Kaggle: https://www.kaggle.com/datasets/fedesoriano/argoverse-motion-forecasting-dataset
- Waymo Kaggle: https://www.kaggle.com/datasets/google/waymo-open-dataset
