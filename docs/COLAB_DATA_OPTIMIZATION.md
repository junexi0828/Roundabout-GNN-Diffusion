# 🚀 Colab 데이터 최적화 가이드

## 문제점

**기존 방식**:

- 로컬에서 전처리된 데이터 (143MB) 생성
- Google Drive에 업로드 (느림, 시간 소요)

**개선 방식**:

- Colab에서 원본 데이터만 다운로드 (작은 용량, 빠름)
- Colab에서 전처리 실행 (GPU 활용, 빠름)

## 속도 비교

| 방식     | 다운로드               | 전처리         | 총 시간 |
| -------- | ---------------------- | -------------- | ------- |
| **기존** | Drive 업로드 143MB     | -              | 10-30분 |
| **개선** | Git clone 어노테이션만 | Colab에서 실행 | 2-5분   |

**속도 향상: 약 5-10배 빠름!** ⚡

## 사용 방법

### Colab 노트북에서 실행

```python
# Step 1: SDD 데이터 다운로드 및 전처리
!python scripts/colab/colab_download_and_preprocess.py \
    --output_dir data/sdd/deathCircle \
    --converted_dir data/sdd/converted
```

**실행 시간**: 약 2-5분

- 다운로드: 1-2분 (어노테이션 파일만, 작은 용량)
- 전처리: 1-3분 (Colab CPU/GPU 활용)

### 옵션

```python
# 이미 다운로드된 경우
!python scripts/colab/colab_download_and_preprocess.py \
    --skip_download \
    --converted_dir data/sdd/converted

# 이미 전처리된 경우
!python scripts/colab/colab_download_and_preprocess.py \
    --skip_preprocess
```

## 데이터 용량 비교

### 원본 데이터 (다운로드)

- 어노테이션 파일만: **약 5-10MB**
- 텍스트 파일 (annotations.txt)
- 빠른 다운로드 가능

### 전처리된 데이터 (생성)

- CSV 파일: **약 143MB**
- Colab 로컬에 저장
- Drive 업로드 불필요

## 전체 워크플로우

### 기존 방식 (느림)

```
로컬 PC:
1. 원본 데이터 다운로드
2. 전처리 실행 (143MB 생성)
3. Drive에 업로드 (10-30분) ❌

Colab:
4. Drive에서 다운로드
5. 모델 학습
```

### 개선 방식 (빠름)

```
Colab:
1. 원본 데이터 다운로드 (git clone, 1-2분) ✅
2. 전처리 실행 (1-3분) ✅
3. 모델 학습
```

## 장점

1. **속도**: 5-10배 빠름
2. **편의성**: 수동 업로드 불필요
3. **효율성**: 필요한 데이터만 다운로드
4. **재현성**: 항상 최신 데이터 사용

## 주의사항

### Colab 세션 제한

- 무료 버전: 12시간 세션 제한
- 전처리된 데이터는 세션 내에서만 유지
- 학습 완료 후 결과만 Drive에 저장

### 해결책

```python
# 학습 완료 후 결과만 Drive에 저장
from google.colab import drive
drive.mount('/content/drive')

# 모델과 결과만 복사
!cp -r results /content/drive/MyDrive/
!cp -r checkpoints /content/drive/MyDrive/
```

## 최종 권장 워크플로우

```python
# Colab 노트북 전체 흐름

# 1. 환경 설정
!pip install -q torch torch-geometric pandas numpy

# 2. 프로젝트 클론
!git clone https://github.com/your-repo/Roundabout_AI.git
%cd Roundabout_AI

# 3. 데이터 다운로드 및 전처리 (2-5분)
!python scripts/colab/colab_download_and_preprocess.py

# 4. 모델 학습 (1-2시간)
!python scripts/training/train_a3tgcn.py --mode fast

# 5. 결과 저장 (선택사항)
from google.colab import drive
drive.mount('/content/drive')
!cp -r results /content/drive/MyDrive/
```

## 요약

✅ **Colab에서 원본 데이터 다운로드 + 전처리**

- 속도: 5-10배 빠름
- 편의성: 수동 작업 불필요
- 효율성: 필요한 데이터만 사용

❌ **Drive 업로드 방식**

- 느림 (10-30분)
- 수동 작업 필요
- 대용량 파일 업로드
