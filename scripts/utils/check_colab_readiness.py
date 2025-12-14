"""
Colab 실행 준비 상태 확인 스크립트
git clone 후 바로 실행 가능한지 확인
"""

from pathlib import Path
import sys


def check_colab_readiness():
    """Colab 실행 준비 상태 확인"""
    print("=" * 80)
    print("Colab 실행 준비 상태 확인")
    print("=" * 80)

    project_root = Path(".")

    # 필수 파일/디렉토리 목록
    required_files = {
        # 스크립트
        "scripts/colab/colab_download_and_preprocess.py": "SDD 데이터 다운로드 및 전처리",
        "scripts/colab/colab_auto_pipeline.py": "자동화 파이프라인",
        "scripts/training/train_a3tgcn.py": "모델 학습",
        "scripts/data/verify_data_consistency.py": "데이터 검증",
        # 소스 코드
        "src/data_processing/sdd_adapter.py": "SDD 어댑터",
        "src/training/data_loader.py": "데이터 로더",
        "src/training/trainer.py": "학습기",
        "src/baselines/a3tgcn_model.py": "A3TGCN 모델",
        # 설정 파일
        "configs/training_config.yaml": "학습 설정",
        "configs/a3tgcn_config.yaml": "A3TGCN 설정",
        # 노트북
        "notebooks/colab_auto.ipynb": "Colab 자동화 노트북",
        # 문서
        "README.md": "프로젝트 README",
        "requirements.txt": "패키지 의존성",
    }

    required_dirs = {
        "src/": "소스 코드",
        "scripts/": "스크립트",
        "configs/": "설정 파일",
        "notebooks/": "노트북",
    }

    print("\n1. 필수 파일 확인:")
    print("-" * 80)

    missing_files = []
    for file_path, description in required_files.items():
        full_path = project_root / file_path
        if full_path.exists():
            size = full_path.stat().st_size
            print(f"  ✓ {file_path:50s} ({size:,} bytes) - {description}")
        else:
            print(f"  ❌ {file_path:50s} - 누락! ({description})")
            missing_files.append(file_path)

    print("\n2. 필수 디렉토리 확인:")
    print("-" * 80)

    missing_dirs = []
    for dir_path, description in required_dirs.items():
        full_path = project_root / dir_path
        if full_path.exists() and full_path.is_dir():
            file_count = len(list(full_path.rglob("*.py"))) + len(
                list(full_path.rglob("*.yaml"))
            )
            print(f"  ✓ {dir_path:50s} ({file_count} 파일) - {description}")
        else:
            print(f"  ❌ {dir_path:50s} - 누락! ({description})")
            missing_dirs.append(dir_path)

    print("\n3. .gitignore 확인:")
    print("-" * 80)

    gitignore_path = project_root / ".gitignore"
    if gitignore_path.exists():
        with open(gitignore_path, "r") as f:
            gitignore_content = f.read()

        # 중요한 ignore 패턴 확인
        ignore_patterns = {
            "data/": "데이터 폴더 (정상 - Colab에서 다운로드)",
            "*.ipynb": "⚠️  노트북 파일 (문제 - notebooks/colab_auto.ipynb 필요!)",
            "*.csv": "CSV 파일 (정상 - Colab에서 생성)",
            "results/": "결과 폴더 (정상 - Colab에서 생성)",
            "venv/": "가상환경 (정상)",
            ".env": "환경 변수 (정상 - 비밀번호 없음)",
        }

        for pattern, status in ignore_patterns.items():
            if pattern in gitignore_content:
                print(f"  {pattern:20s} - {status}")
            else:
                print(f"  {pattern:20s} - ignore되지 않음")
    else:
        print("  ❌ .gitignore 파일 없음")

    print("\n4. 최종 결과:")
    print("=" * 80)

    if missing_files or missing_dirs:
        print(f"\n❌ 누락된 항목:")
        if missing_files:
            print(f"  파일: {len(missing_files)}개")
            for f in missing_files:
                print(f"    - {f}")
        if missing_dirs:
            print(f"  디렉토리: {len(missing_dirs)}개")
            for d in missing_dirs:
                print(f"    - {d}")
        return False
    else:
        print("\n✅ 모든 필수 파일/디렉토리 존재")

        # .gitignore에서 *.ipynb 확인
        if gitignore_path.exists():
            with open(gitignore_path, "r") as f:
                if "*.ipynb" in f.read():
                    print(
                        "\n⚠️  경고: .gitignore에 *.ipynb가 있어 notebooks/colab_auto.ipynb가 포함되지 않을 수 있습니다."
                    )
                    print(
                        "   해결: .gitignore에서 *.ipynb 제거 또는 notebooks/*.ipynb 예외 추가"
                    )

        return True


if __name__ == "__main__":
    success = check_colab_readiness()
    sys.exit(0 if success else 1)
