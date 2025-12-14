# Colab 완전 자동화 가이드

## 🚀 원클릭 실행

### 가장 간단한 방법

Colab 노트북에서 한 줄 실행:

```python
!git clone https://github.com/junexi0828/Roundabout-GNN-Diffusion.git && cd Roundabout-GNN-Diffusion && python scripts/colab/colab_one_click.py
```

**끝!** ☕

## 📋 수동 작업 (한 번만)

### SDD 데이터 준비

1. [Stanford Drone Dataset](https://cvgl.stanford.edu/projects/uav_data/) 접속
2. Death Circle 비디오 다운로드
3. Google Drive에 업로드:
   ```
   Google Drive/
   └── Roundabout_AI_Data/
       └── (CSV 파일 또는 annotations.txt)
   ```

## 🎯 자동 실행 프로세스

```
[1/9] 환경 설정          (2분)
[2/9] GitHub 클론        (1분)
[3/9] Google Drive 마운트 (10초)
[4/9] 데이터 준비        (30초)
[5/9] 데이터 전처리      (10-30분)
[6/9] 모델 학습 ⚡       (1-4시간)
[7/9] 모델 평가          (10분)
[8/9] 결과 시각화        (2분)
[9/9] 결과 저장          (1분)
```

## 📊 결과물

학습 완료 후 자동 생성:

```
results/
├── checkpoints/best_model.pth
├── metrics/evaluation_results.json
└── visualizations/
    ├── training_curves.png
    ├── sample_trajectories.png
    └── evaluation_results.png
```

**Google Drive 자동 저장**:
```
Google Drive/
└── Roundabout_AI_Results/
    └── YYYYMMDD_HHMMSS/
```

## 🔧 고급 옵션

### Fast 모드 (빠른 테스트)

```python
!python scripts/colab/colab_auto_pipeline.py --mode fast
```

### Full 모드 (전체 학습)

```python
!python scripts/colab/colab_auto_pipeline.py --mode full
```

### 데이터 경로 지정

```python
!python scripts/colab/colab_auto_pipeline.py --mode fast --data-dir /content/drive/MyDrive/MyData
```

## 📚 상세 가이드

- [완전 자동화 가이드](docs/COLAB_AUTO_GUIDE.md)
- [원클릭 가이드](docs/COLAB_ONE_CLICK.md)
- [의존성 가이드](docs/DEPENDENCIES.md)

## ✅ 체크리스트

- [ ] Colab 노트북 열기
- [ ] GPU 런타임 선택
- [ ] SDD 데이터 Google Drive에 업로드
- [ ] 한 줄 실행!

## 🎉 완료!

학습 완료 후:
- ✅ 모델 체크포인트
- ✅ 평가 결과
- ✅ 시각화
- ✅ Google Drive 자동 저장

**이제 논문 작성 준비 완료!** 📝

