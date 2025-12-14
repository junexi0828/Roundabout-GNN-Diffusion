#!/bin/bash

# 모델 학습 실행 스크립트
# 사용법: ./scripts/train_model.sh [옵션]

set -e

# 프로젝트 루트 디렉토리
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 가상환경 활성화
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "가상환경을 찾을 수 없습니다. setup.sh를 먼저 실행하세요."
    exit 1
fi

# 기본 설정
CONFIG="${1:-configs/training_config.yaml}"
DATA_DIR="${2:-data/processed}"

echo "========================================="
echo "모델 학습 시작"
echo "========================================="
echo "설정 파일: $CONFIG"
echo "데이터 디렉토리: $DATA_DIR"
echo ""

# 학습 실행
python src/training/train.py \
    --config "$CONFIG" \
    --data_dir "$DATA_DIR" \
    "${@:3}"

echo ""
echo "========================================="
echo "학습 완료"
echo "========================================="
echo "TensorBoard 로그 확인: tensorboard --logdir runs"
echo "체크포인트 위치: checkpoints/"

