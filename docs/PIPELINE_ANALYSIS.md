# 파이프라인 구조 분석 리포트

## 현재 파이프라인 구조

### 진입점

- `scripts/train.py --full-pipeline` → `local_auto_pipeline.py` 또는 `colab_auto_pipeline.py` 호출

### 현재 사용 중인 스크립트

#### ✅ 사용 중

1. **데이터 처리**

   - `scripts/data/preprocess_sdd.py` ✅ (전처리 단계에서 사용)

2. **학습**

   - `scripts/training/train_mid.py` ✅ (MID 모델 학습)

3. **시각화**

   - `scripts/utils/visualize_results.py` ✅ (결과 시각화 단계)
   - `scripts/evaluation/generate_sample_analysis.py` ✅ (샘플 분석 생성)

4. **평가** (Colab만, 주석 처리됨)
   - `scripts/evaluation/compare_baselines.py` ⚠️ (주석 처리됨)

### ❌ 사용 안 되는 스크립트

#### 평가 스크립트

- `scripts/evaluation/evaluate_mid.py` ❌
  - **문제**: 파이프라인에 평가 단계가 없음
  - **기능**: MID 모델 평가 및 메트릭 계산
  - **권장**: 파이프라인에 평가 단계 추가

#### 학습 스크립트

- `scripts/training/train_a3tgcn.py` ❌

  - **문제**: 베이스라인 학습이 주석 처리됨
  - **기능**: A3TGCN 베이스라인 모델 학습
  - **권장**: 베이스라인 비교를 위해 활성화 필요

- `scripts/training/train_trajectron.py` ❌
  - **문제**: 베이스라인 학습이 주석 처리됨
  - **기능**: Trajectron++ 베이스라인 모델 학습
  - **권장**: 베이스라인 비교를 위해 활성화 필요

#### 데이터 검증 스크립트

- `scripts/data/verify_dataset.py` ❌

  - **기능**: 데이터셋 검증
  - **권장**: 데이터 확인 단계에 통합

- `scripts/data/verify_data_consistency.py` ❌

  - **기능**: 데이터 일관성 검증
  - **권장**: 전처리 후 검증 단계 추가

- `scripts/data/verify_sdd_data.py` ❌

  - **기능**: SDD 데이터 검증
  - **권장**: 데이터 다운로드 후 검증

- `scripts/data/download_sdd_deathcircle.py` ❌

  - **문제**: 파이프라인에서 직접 구현되어 있음
  - **기능**: SDD Death Circle 데이터 다운로드
  - **권장**: 파이프라인에서 이 스크립트 사용

- `scripts/data/auto_homography_estimation.py` ❌
  - **기능**: 호모그래피 행렬 자동 추정
  - **권장**: 전처리 단계에 통합

#### Legacy 스크립트

- `scripts/legacy/` 전체 ❌
  - **문제**: 완전히 사용되지 않음
  - **기능**: 이전 버전의 파이프라인 스크립트들
  - **권장**: 삭제 또는 아카이브

## 개선 권장 사항

### 1. 파이프라인 단계 추가

현재 파이프라인:

1. 환경 확인
2. 데이터 확인
3. 데이터 전처리
4. MID 모델 학습
5. 결과 시각화

**개선된 파이프라인:**

1. 환경 확인
2. 데이터 다운로드 (스크립트 사용)
3. 데이터 검증 (verify 스크립트 사용)
4. 데이터 전처리
5. 전처리 검증 (verify_data_consistency 사용)
6. 베이스라인 학습 (선택적, train_a3tgcn.py, train_trajectron.py)
7. MID 모델 학습
8. 모델 평가 (evaluate_mid.py 사용)
9. 베이스라인 비교 (compare_baselines.py 사용)
10. 결과 시각화

### 2. 스크립트 통합 우선순위

#### 높은 우선순위

1. ✅ `evaluate_mid.py` - 평가 단계 필수
2. ✅ `compare_baselines.py` - 베이스라인 비교 필수
3. ✅ `download_sdd_deathcircle.py` - 데이터 다운로드 표준화

#### 중간 우선순위

4. ⚠️ `train_a3tgcn.py` - 베이스라인 비교용
5. ⚠️ `train_trajectron.py` - 베이스라인 비교용
6. ⚠️ `verify_dataset.py` - 데이터 품질 보장

#### 낮은 우선순위

7. 📝 `verify_data_consistency.py` - 선택적 검증
8. 📝 `auto_homography_estimation.py` - 필요시 사용
9. 🗑️ `scripts/legacy/` - 삭제 또는 아카이브

## 결론

**현재 상황:**

- 파이프라인이 일부 스크립트만 사용하고 있음
- 평가 단계가 누락되어 있음
- 베이스라인 비교 기능이 비활성화됨
- 데이터 검증 단계가 없음

**권장 조치:**

1. 파이프라인에 평가 단계 추가 (`evaluate_mid.py` 통합)
2. 베이스라인 비교 기능 활성화 (`compare_baselines.py`, `train_a3tgcn.py`, `train_trajectron.py`)
3. 데이터 검증 단계 추가 (`verify_dataset.py` 등)
4. 데이터 다운로드 표준화 (`download_sdd_deathcircle.py` 사용)
5. Legacy 스크립트 정리 (삭제 또는 아카이브)
