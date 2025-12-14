"""
Argoverse Motion Forecasting Dataset 다운로드 스크립트
Kaggle에서 바로 다운로드 가능한 궤적 예측 데이터셋
"""

import subprocess
import sys
from pathlib import Path

def check_kaggle_installed():
    """Kaggle CLI 설치 여부 확인"""
    try:
        subprocess.run(['kaggle', '--version'],
                      capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_kaggle():
    """Kaggle CLI 설치"""
    print("Kaggle CLI 설치 중...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'kaggle'],
                  check=True)
    print("✓ Kaggle CLI 설치 완료")

def download_argoverse(output_dir: Path):
    """Argoverse 데이터셋 다운로드"""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Argoverse Motion Forecasting Dataset 다운로드")
    print("=" * 80)
    print("\n⚠️  주의사항:")
    print("1. Kaggle 계정이 필요합니다 (무료 가입 가능)")
    print("2. API 토큰이 필요합니다:")
    print("   - Kaggle 계정 설정 > API > Create New Token")
    print("   - ~/.kaggle/kaggle.json 파일에 저장")
    print("\n다운로드 시작...")

    dataset_name = "fedesoriano/argoverse-motion-forecasting-dataset"

    try:
        # 데이터셋 다운로드
        cmd = ['kaggle', 'datasets', 'download', '-d', dataset_name,
               '-p', str(output_dir)]
        subprocess.run(cmd, check=True)

        print(f"\n✓ 다운로드 완료: {output_dir}")
        print("\n다음 단계:")
        print(f"1. 압축 해제: cd {output_dir} && unzip *.zip")
        print("2. 데이터 검증: python scripts/verify_dataset.py --dataset argoverse")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ 다운로드 실패: {e}")
        print("\n해결 방법:")
        print("1. Kaggle CLI 설치 확인: pip install kaggle")
        print("2. API 토큰 설정 확인: ~/.kaggle/kaggle.json")
        print("3. Kaggle 계정 로그인 확인")
        return False

    return True

def main():
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "data" / "argoverse"

    # Kaggle CLI 확인
    if not check_kaggle_installed():
        print("Kaggle CLI가 설치되지 않았습니다.")
        response = input("설치하시겠습니까? (y/n): ")
        if response.lower() == 'y':
            install_kaggle()
        else:
            print("수동 설치: pip install kaggle")
            return

    # 다운로드 실행
    success = download_argoverse(output_dir)

    if success:
        print("\n" + "=" * 80)
        print("✓ 다운로드 완료!")
        print("=" * 80)

if __name__ == "__main__":
    main()

