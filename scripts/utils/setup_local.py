"""
로컬 실행 환경 설정 스크립트
MacBook Air (Apple Silicon) 최적화
"""

import subprocess
import sys
from pathlib import Path

def check_pytorch():
    """PyTorch 설치 확인"""
    try:
        import torch
        print(f"✓ PyTorch 설치됨: {torch.__version__}")
        return True
    except ImportError:
        print("❌ PyTorch가 설치되지 않았습니다")
        return False

def install_pytorch():
    """PyTorch 설치 (Apple Silicon용)"""
    print("\n[PyTorch 설치 중...]")
    print("Apple Silicon용 PyTorch를 설치합니다...")

    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio"
        ])
        print("✓ PyTorch 설치 완료")
        return True
    except subprocess.CalledProcessError:
        print("❌ PyTorch 설치 실패")
        return False

def check_mps():
    """MPS 사용 가능 여부 확인"""
    try:
        import torch
        if torch.backends.mps.is_available():
            print(f"✓ MPS 사용 가능 (Apple Silicon GPU)")
            return True
        else:
            print("⚠️  MPS 사용 불가 (CPU 모드)")
            return False
    except:
        return False

def install_requirements():
    """필수 라이브러리 설치"""
    print("\n[필수 라이브러리 설치 중...]")

    req_file = Path(__file__).parent.parent / "requirements.txt"
    if not req_file.exists():
        print("❌ requirements.txt를 찾을 수 없습니다")
        return False

    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(req_file)
        ])
        print("✓ 필수 라이브러리 설치 완료")
        return True
    except subprocess.CalledProcessError:
        print("❌ 라이브러리 설치 실패")
        return False

def main():
    print("=" * 80)
    print("로컬 실행 환경 설정")
    print("=" * 80)

    # PyTorch 확인
    if not check_pytorch():
        response = input("\nPyTorch를 설치하시겠습니까? (y/n): ")
        if response.lower() == 'y':
            if not install_pytorch():
                return
        else:
            print("PyTorch 설치를 건너뜁니다")
            return

    # MPS 확인
    check_mps()

    # 필수 라이브러리 확인
    print("\n[필수 라이브러리 확인]")
    libraries = ['torch_geometric', 'pandas', 'numpy', 'matplotlib']
    missing = []

    for lib in libraries:
        try:
            __import__(lib)
            print(f"  ✓ {lib}")
        except ImportError:
            print(f"  ❌ {lib} (누락)")
            missing.append(lib)

    if missing:
        response = input(f"\n누락된 라이브러리 {len(missing)}개를 설치하시겠습니까? (y/n): ")
        if response.lower() == 'y':
            install_requirements()

    print("\n" + "=" * 80)
    print("환경 설정 완료!")
    print("=" * 80)
    print("\n다음 단계:")
    print("1. 데이터 전처리: python src/data_processing/preprocessor.py")
    print("2. 빠른 학습: python scripts/training/fast_train.py --device mps --batch_size 16")

if __name__ == "__main__":
    main()

