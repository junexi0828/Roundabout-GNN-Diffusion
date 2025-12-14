"""
시스템 환경 확인 스크립트
로컬 실행 가능 여부 체크
"""

import sys
import platform
from pathlib import Path

def check_system():
    """시스템 환경 확인"""
    print("=" * 80)
    print("시스템 환경 확인")
    print("=" * 80)

    # Python 버전
    print(f"\n[Python 환경]")
    print(f"  버전: {sys.version}")
    print(f"  경로: {sys.executable}")

    # 운영체제
    print(f"\n[운영체제]")
    print(f"  시스템: {platform.system()}")
    print(f"  릴리스: {platform.release()}")
    print(f"  머신: {platform.machine()}")

    # PyTorch 확인
    try:
        import torch
        print(f"\n[PyTorch]")
        print(f"  버전: {torch.__version__}")
        print(f"  CUDA 사용 가능: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"  CUDA 버전: {torch.version.cuda}")
            print(f"  GPU 개수: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"      메모리: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        else:
            print("  ⚠️  GPU를 사용할 수 없습니다 (CPU 모드로 실행)")
    except ImportError:
        print("\n❌ PyTorch가 설치되지 않았습니다")
        return False

    # 필수 라이브러리 확인
    print(f"\n[필수 라이브러리]")
    libraries = {
        'torch': 'PyTorch',
        'torch_geometric': 'PyTorch Geometric',
        'torch_geometric_temporal': 'PyTorch Geometric Temporal',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'networkx': 'NetworkX',
        'shapely': 'Shapely'
    }

    all_ok = True
    for module, name in libraries.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ❌ {name} (설치 필요)")
            all_ok = False

    # 데이터 확인
    print(f"\n[데이터셋]")
    data_dir = Path(__file__).parent.parent / "data" / "sdd"
    if (data_dir / "converted").exists():
        csv_files = list((data_dir / "converted").glob("*.csv"))
        print(f"  ✓ 변환된 데이터: {len(csv_files)}개 파일")
    else:
        print(f"  ⚠️  변환된 데이터 없음")

    if (data_dir / "homography" / "H.txt").exists():
        print(f"  ✓ 호모그래피 행렬 존재")
    else:
        print(f"  ⚠️  호모그래피 행렬 없음")

    # 권장사항
    print(f"\n[로컬 실행 권장사항]")

    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory >= 8:
            print(f"  ✓ GPU 메모리 충분 ({gpu_memory:.1f}GB)")
            print(f"    → 배치 크기 32-64 권장")
        else:
            print(f"  ⚠️  GPU 메모리 부족 ({gpu_memory:.1f}GB)")
            print(f"    → 배치 크기 16-32 권장")
            print(f"    → Mixed Precision Training 권장")
    else:
        print(f"  ⚠️  CPU 모드")
        print(f"    → 배치 크기 8-16 권장")
        print(f"    → 데이터 샘플링 권장 (sample_ratio=0.3)")
        print(f"    → 경량 모델 사용 권장 (lightweight=True)")

    print(f"\n{'='*80}")

    if all_ok:
        print("✓ 시스템 환경 확인 완료 - 로컬 실행 가능")
    else:
        print("❌ 일부 라이브러리가 누락되었습니다")
        print("   설치: pip install -r requirements.txt")

    print(f"{'='*80}")

    return all_ok

if __name__ == "__main__":
    check_system()

