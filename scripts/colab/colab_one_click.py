"""
Colab 원클릭 실행 스크립트
가장 간단한 사용법
"""

import subprocess
import sys
from pathlib import Path

def main():
    """원클릭 실행"""
    print("=" * 80)
    print("🚀 Colab 완전 자동화 파이프라인")
    print("=" * 80)
    print("\n하나의 명령으로 전체 프로세스 자동 실행!")
    print("\n실행 중...")
    print("=" * 80)

    # 자동화 파이프라인 실행
    script_path = Path(__file__).parent / "colab_auto_pipeline.py"

    result = subprocess.run(
        [sys.executable, str(script_path), "--mode", "fast"],
        cwd=Path(__file__).parent.parent
    )

    if result.returncode == 0:
        print("\n" + "=" * 80)
        print("✅ 완료! 모든 결과가 Google Drive에 저장되었습니다.")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("❌ 오류 발생. 로그를 확인하세요.")
        print("=" * 80)

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())

