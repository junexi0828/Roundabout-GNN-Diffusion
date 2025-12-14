#!/bin/bash

# 회전교차로 상호작용 예측 연구 프로젝트 환경 설정 스크립트

echo "========================================="
echo "회전교차로 GNN 연구 프로젝트 환경 설정"
echo "========================================="

# Python 버전 확인
echo "Python 버전 확인 중..."
python3 --version

# 가상환경 생성
echo ""
echo "가상환경 생성 중..."
if [ -d "venv" ]; then
    echo "가상환경이 이미 존재합니다. 기존 가상환경을 사용합니다."
else
    python3 -m venv venv
    echo "가상환경 생성 완료!"
fi

# 가상환경 활성화
echo ""
echo "가상환경 활성화 중..."
source venv/bin/activate

# pip 업그레이드
echo ""
echo "pip 업그레이드 중..."
pip install --upgrade pip

# 패키지 설치
echo ""
echo "필수 패키지 설치 중..."
echo "이 작업은 몇 분이 소요될 수 있습니다..."
pip install -r requirements.txt

echo ""
echo "========================================="
echo "환경 설정 완료!"
echo "========================================="
echo ""
echo "가상환경을 활성화하려면 다음 명령어를 실행하세요:"
echo "  source venv/bin/activate"
echo ""

