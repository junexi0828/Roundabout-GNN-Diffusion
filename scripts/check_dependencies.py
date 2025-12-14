"""
의존성 확인 스크립트
프로젝트에 필요한 모든 패키지가 설치되었는지 확인
"""

import sys
from typing import Dict, List, Tuple


def check_dependencies() -> Tuple[bool, List[str]]:
    """
    모든 의존성 확인

    Returns:
        (모두 설치됨 여부, 누락된 패키지 리스트)
    """
    # 필수 패키지 목록
    required_packages: Dict[str, str] = {
        # Core Deep Learning
        'torch': 'torch',
        'torchvision': 'torchvision',
        'torchaudio': 'torchaudio',
        'torch_geometric': 'torch_geometric',

        # Data Processing
        'pandas': 'pandas',
        'numpy': 'numpy',
        'scipy': 'scipy',

        # Graph Processing
        'networkx': 'networkx',

        # Geometric Operations
        'shapely': 'shapely',

        # Computer Vision
        'opencv': 'cv2',

        # Visualization
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',

        # Utilities
        'tqdm': 'tqdm',
        'yaml': 'yaml',
        'tensorboard': 'tensorboard',

        # Machine Learning
        'sklearn': 'sklearn',

        # PyTorch Geometric 의존성
        'xxhash': 'xxhash',
        'aiohttp': 'aiohttp',
        'psutil': 'psutil',
        'requests': 'requests',
    }

    # 선택적 패키지
    optional_packages: Dict[str, str] = {
        'torch_geometric_temporal': 'torch_geometric_temporal',  # A3TGCN 사용 시 필요
        'jupyter': 'jupyter',
        'ipykernel': 'ipykernel',
        'notebook': 'notebook',
        'trajdata': 'trajdata',
    }

    missing_required = []
    missing_optional = []
    installed = []

    print("=" * 80)
    print("의존성 확인")
    print("=" * 80)

    # 필수 패키지 확인
    print("\n[필수 패키지]")
    for package_name, module_name in required_packages.items():
        try:
            __import__(module_name)
            print(f"  ✓ {package_name}")
            installed.append(package_name)
        except ImportError:
            print(f"  ❌ {package_name} (누락)")
            missing_required.append(package_name)
        except Exception as e:
            print(f"  ⚠️  {package_name} (오류: {e})")
            missing_required.append(package_name)

    # 선택적 패키지 확인
    print("\n[선택적 패키지]")
    for package_name, module_name in optional_packages.items():
        try:
            __import__(module_name)
            print(f"  ✓ {package_name}")
            installed.append(package_name)
        except ImportError:
            print(f"  ⚠️  {package_name} (선택사항, 누락)")
            missing_optional.append(package_name)

    # 버전 정보
    print("\n[주요 패키지 버전]")
    version_info = {
        'torch': 'torch',
        'torch_geometric': 'torch_geometric',
        'pandas': 'pandas',
        'numpy': 'numpy',
    }

    for name, module_name in version_info.items():
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"  {name}: {version}")
        except:
            pass

    # 결과 요약
    print("\n" + "=" * 80)
    print("결과 요약")
    print("=" * 80)
    print(f"  설치됨: {len(installed)}개")
    print(f"  필수 누락: {len(missing_required)}개")
    print(f"  선택 누락: {len(missing_optional)}개")

    if missing_required:
        print(f"\n❌ 필수 패키지 누락:")
        for pkg in missing_required:
            print(f"    - {pkg}")
        print("\n설치 명령:")
        print(f"  pip install {' '.join(missing_required)}")
        return False, missing_required

    if missing_optional:
        print(f"\n⚠️  선택적 패키지 누락 (선택사항):")
        for pkg in missing_optional:
            print(f"    - {pkg}")

    print("\n✓ 모든 필수 패키지가 설치되었습니다!")
    return True, []


def main():
    """메인 함수"""
    success, missing = check_dependencies()

    if not success:
        print("\n" + "=" * 80)
        print("설치 가이드")
        print("=" * 80)
        print("다음 명령으로 누락된 패키지를 설치하세요:")
        print(f"  pip install {' '.join(missing)}")
        print("\n또는 전체 requirements.txt 설치:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    else:
        print("\n✓ 환경 설정 완료!")
        sys.exit(0)


if __name__ == "__main__":
    main()

