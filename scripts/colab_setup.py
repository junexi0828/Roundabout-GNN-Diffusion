"""
Colab 환경 설정 스크립트
전체 환경을 자동으로 설정
"""

import subprocess
import sys
from pathlib import Path

def setup_colab():
    """Colab 환경 자동 설정"""
    print("=" * 80)
    print("Colab 환경 설정")
    print("=" * 80)

    # 1. 라이브러리 설치
    print("\n[1/5] 라이브러리 설치 중...")
    packages = [
        "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
        "torch-geometric torch-geometric-temporal",
        "pandas numpy scipy scikit-learn",
        "matplotlib seaborn opencv-python",
        "networkx tqdm pyyaml shapely tensorboard"
    ]

    for pkg in packages:
        print(f"  설치 중: {pkg.split()[0]}...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q"
        ] + pkg.split())

    print("✓ 라이브러리 설치 완료")

    # 2. GPU 확인
    print("\n[2/5] GPU 확인 중...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU 사용 가능: {torch.cuda.get_device_name(0)}")
            print(f"  메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print("⚠️  GPU를 사용할 수 없습니다 (CPU 모드)")
    except:
        print("⚠️  PyTorch 확인 실패")

    # 3. 프로젝트 경로 설정
    print("\n[3/5] 프로젝트 경로 설정 중...")
    project_root = Path("/content/Roundabout_AI")
    if project_root.exists():
        print(f"✓ 프로젝트 디렉토리 존재: {project_root}")
    else:
        print(f"⚠️  프로젝트 디렉토리 없음: {project_root}")
        print("   GitHub에서 클론하거나 파일을 업로드하세요")

    # 4. 데이터 확인
    print("\n[4/5] 데이터 확인 중...")
    data_dir = project_root / "data" / "sdd" / "converted"
    if data_dir.exists():
        csv_files = list(data_dir.glob("*.csv"))
        print(f"✓ 변환된 데이터: {len(csv_files)}개 파일")
    else:
        print(f"⚠️  데이터 디렉토리 없음: {data_dir}")
        print("   데이터를 업로드하거나 Google Drive에서 링크하세요")

    # 5. 설정 완료
    print("\n[5/5] 설정 완료!")
    print("\n" + "=" * 80)
    print("다음 단계:")
    print("1. 데이터 전처리: python src/data_processing/preprocessor.py")
    print("2. 모델 학습: python src/training/train.py --config configs/colab_config.yaml")
    print("=" * 80)

if __name__ == "__main__":
    setup_colab()
