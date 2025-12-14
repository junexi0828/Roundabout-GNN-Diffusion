# 베이스라인 자동화 가이드

## ✅ 완성된 스크립트

### 1. A3TGCN 학습 스크립트 ✅

**파일**: `scripts/train_a3tgcn.py`

**사용법**:

```bash
python scripts/training/train_a3tgcn.py --config configs/a3tgcn_config.yaml --data_dir data/processed
```

**기능**:

- A3TGCN 모델 학습
- 씬 그래프 자동 생성
- 체크포인트 저장
- TensorBoard 로깅

---

### 2. 비교 평가 스크립트 ✅

**파일**: `scripts/compare_baselines.py`

**사용법**:

```bash
python scripts/compare_baselines.py \
    --mid_checkpoint checkpoints/mid/best_model.pth \
    --a3tgcn_checkpoint checkpoints/a3tgcn/best_model.pth \
    --data_dir data/processed \
    --output_dir results/comparison
```

**기능**:

- HSG-Diffusion vs A3TGCN 비교
- 평가 지표 계산 (ADE, FDE, Diversity, Coverage)
- CSV 표 생성
- LaTeX 표 생성 (논문용)

---

### 3. A3TGCN 설정 파일 ✅

**파일**: `configs/a3tgcn_config.yaml`

**주요 설정**:

- 모델: node_features, hidden_channels, pred_steps
- 데이터: batch_size, train_ratio, use_scene_graph
- 학습: optimizer, learning_rate, num_epochs

---

## 🚀 Colab 자동화

### 전체 파이프라인 실행

```python
# Colab에서 실행
!python scripts/colab/colab_auto_pipeline.py --mode fast
```

**자동 실행 순서**:

1. 환경 설정
2. GitHub 저장소 클론
3. Google Drive 마운트
4. 데이터 준비
5. 데이터 전처리
6. **HSG-Diffusion 학습** ✅
7. **A3TGCN 학습** ✅ (새로 추가됨)
8. **베이스라인 비교 평가** ✅ (새로 추가됨)
9. 결과 시각화
10. 결과 저장 (Google Drive)

---

## 📊 비교 결과

### 출력 형식

**콘솔 출력**:

```
베이스라인 비교 결과
================================================================================
Model                minADE₂₀     minFDE₂₀     Diversity     Coverage
--------------------------------------------------------------------------------
HSG-Diffusion        0.9200       1.7800       0.9000       0.8500
A3TGCN               1.2000       2.5000       0.0000       0.0000
```

**CSV 파일**: `results/comparison/comparison_table.csv`

**LaTeX 파일**: `results/comparison/comparison_table.tex`

---

## 🎯 사용 시나리오

### 시나리오 1: 전체 자동화 (Colab)

```python
# Colab 노트북에서
!python scripts/colab/colab_auto_pipeline.py --mode fast
```

**결과**:

- HSG-Diffusion 학습 완료
- A3TGCN 학습 완료
- 비교 평가 완료
- 결과 자동 저장

---

### 시나리오 2: 개별 실행

```bash
# 1. HSG-Diffusion 학습
python scripts/train_mid.py --config configs/mid_config_fast.yaml

# 2. A3TGCN 학습
python scripts/training/train_a3tgcn.py --config configs/a3tgcn_config.yaml

# 3. 비교 평가
python scripts/compare_baselines.py
```

---

### 시나리오 3: 빠른 비교 (이미 학습된 모델)

```bash
# 체크포인트만 있으면 바로 비교 가능
python scripts/compare_baselines.py \
    --mid_checkpoint checkpoints/mid/best_model.pth \
    --a3tgcn_checkpoint checkpoints/a3tgcn/best_model.pth
```

---

## 📋 체크리스트

### ✅ 완료된 항목

- [x] A3TGCN 학습 스크립트 (`train_a3tgcn.py`)
- [x] A3TGCN 설정 파일 (`a3tgcn_config.yaml`)
- [x] 비교 평가 스크립트 (`compare_baselines.py`)
- [x] Colab 자동화 파이프라인 통합
- [x] CSV 표 생성
- [x] LaTeX 표 생성

### ⚠️ 선택적 항목

- [ ] Trajectron++ 통합 (복잡도 높음)
- [ ] Social-STGCNN 통합 (보행자 중심)

---

## 🎉 완료!

**베이스라인 비교가 완전히 자동화되었습니다!**

- ✅ A3TGCN 학습 자동화
- ✅ 비교 평가 자동화
- ✅ Colab 통합 완료
- ✅ 논문용 표 자동 생성

**이제 Colab에서 한 번의 실행으로 전체 비교 실험이 완료됩니다!** 🚀
