#!/bin/bash
# PyTorch Geometric 설치 스크립트 (macOS Apple Silicon)

echo "PyTorch Geometric 설치 중 (macOS Apple Silicon용)..."

# 가상환경 활성화
source venv/bin/activate

# PyTorch 버전 확인
PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null | cut -d+ -f1)

if [ -z "$PYTORCH_VERSION" ]; then
    echo "❌ PyTorch가 설치되지 않았습니다"
    exit 1
fi

echo "PyTorch 버전: $PYTORCH_VERSION"

# CPU 버전으로 설치 (Apple Silicon은 MPS 사용)
# torch-geometric-temporal은 선택사항 (A3TGCN 사용 시에만 필요)
echo "torch-geometric 설치 중..."
pip install torch-geometric

# 기본 확장 라이브러리 (선택사항, 성능 향상용)
echo "확장 라이브러리 설치 시도 중..."
pip install pyg_lib torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-${PYTORCH_VERSION}+cpu.html 2>/dev/null || echo "⚠️  확장 라이브러리 설치 실패 (선택사항, 기본 기능은 사용 가능)"

# torch-geometric-temporal (선택사항)
echo "torch-geometric-temporal 설치 시도 중..."
pip install torch-geometric-temporal 2>/dev/null || echo "⚠️  torch-geometric-temporal 설치 실패 (A3TGCN 사용 시 필요)"

echo ""
echo "✓ 설치 완료!"
echo ""
echo "확인:"
python -c "import torch_geometric; print(f'torch-geometric: {torch_geometric.__version__}')" 2>/dev/null || echo "torch-geometric 확인 실패"

