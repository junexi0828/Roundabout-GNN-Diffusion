"""
Colab에서 SDD 데이터 다운로드 및 전처리 자동화
원본 데이터만 받아서 Colab에서 전처리 (Drive 업로드 불필요)
"""

import subprocess
import sys
from pathlib import Path
import shutil


def download_sdd_in_colab(output_dir: Path):
    """
    Colab에서 SDD Death Circle 원본 데이터 다운로드
    어노테이션 파일만 다운로드 (작은 용량)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🚀 SDD Death Circle 데이터 다운로드 (Colab)")
    print("=" * 80)
    print("\n원본 어노테이션 파일만 다운로드 (빠름)")
    print("전처리는 Colab에서 실행\n")

    # GitHub 리포지토리 URL
    repo_url = "https://github.com/flclain/StanfordDroneDataset.git"
    temp_dir = output_dir.parent / "temp_sdd"

    try:
        # 임시 디렉토리에 클론
        if temp_dir.exists():
            print(f"기존 임시 디렉토리 삭제: {temp_dir}")
            shutil.rmtree(temp_dir)

        print(f"GitHub에서 클론 중: {repo_url}")
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(temp_dir)],
            check=True,
            capture_output=True,
        )

        # Death Circle 디렉토리 찾기
        deathcircle_dir = None
        for root_dir in [temp_dir, temp_dir / "annotations"]:
            if (root_dir / "annotations" / "deathCircle").exists():
                deathcircle_dir = root_dir / "annotations" / "deathCircle"
                break
            elif (root_dir / "deathCircle").exists():
                deathcircle_dir = root_dir / "deathCircle"
                break

        if deathcircle_dir is None:
            # 디렉토리 구조 확인
            print("\n디렉토리 구조 확인 중...")
            for item in temp_dir.rglob("*"):
                if "death" in item.name.lower() or "circle" in item.name.lower():
                    print(f"  발견: {item}")

            raise FileNotFoundError("Death Circle 디렉토리를 찾을 수 없습니다")

        print(f"\n✓ Death Circle 디렉토리 발견: {deathcircle_dir}")

        # 어노테이션 파일 찾기
        annotation_files = list(deathcircle_dir.glob("**/annotations.txt"))

        if not annotation_files:
            # 다른 패턴 시도
            annotation_files = list(deathcircle_dir.glob("**/*.txt"))

        if not annotation_files:
            print("\n⚠️  어노테이션 파일을 찾을 수 없습니다.")
            print("디렉토리 내용:")
            for item in sorted(deathcircle_dir.iterdir()):
                print(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")
            raise FileNotFoundError("어노테이션 파일을 찾을 수 없습니다")

        print(f"\n✓ 어노테이션 파일 {len(annotation_files)}개 발견")

        # 출력 디렉토리에 복사
        for ann_file in annotation_files:
            # video 디렉토리 구조 유지
            rel_path = ann_file.relative_to(deathcircle_dir)
            dest_path = output_dir / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ann_file, dest_path)
            print(f"  ✓ {ann_file.name} -> {dest_path}")

        # 호모그래피 행렬 파일 확인 (H.txt)
        h_files = list(deathcircle_dir.glob("**/H.txt"))
        if h_files:
            print(f"\n✓ 호모그래피 행렬 파일 {len(h_files)}개 발견")
            for h_file in h_files:
                rel_path = h_file.relative_to(deathcircle_dir)
                dest_path = output_dir / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(h_file, dest_path)
                print(f"  ✓ {h_file.name} -> {dest_path}")
        else:
            print("\n⚠️  호모그래피 행렬 파일 없음 (자동 생성됨)")

        # 임시 디렉토리 정리
        print(f"\n임시 디렉토리 정리 중...")
        shutil.rmtree(temp_dir)

        print(f"\n{'='*80}")
        print("✓ 다운로드 완료!")
        print(f"{'='*80}")
        print(f"\n데이터 위치: {output_dir}")
        print(
            f"용량: {sum(f.stat().st_size for f in output_dir.rglob('*') if f.is_file()) / 1024 / 1024:.2f} MB"
        )

        return True

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Git 클론 실패: {e}")
        print(f"출력: {e.stdout.decode() if e.stdout else ''}")
        print(f"오류: {e.stderr.decode() if e.stderr else ''}")
        return False
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback

        traceback.print_exc()
        return False


def preprocess_in_colab(sdd_dir: Path, output_dir: Path):
    """
    Colab에서 SDD 데이터 전처리
    """
    print("\n" + "=" * 80)
    print("🔄 SDD 데이터 전처리 (Colab)")
    print("=" * 80)

    # 프로젝트 경로 추가
    project_root = Path.cwd()
    if (project_root / "src").exists():
        sys.path.insert(0, str(project_root))

    try:
        from src.data_processing.sdd_adapter import SDDAdapter

        adapter = SDDAdapter(sdd_dir)
        adapter.convert_all_videos(output_dir)

        print(f"\n{'='*80}")
        print("✓ 전처리 완료!")
        print(f"{'='*80}")
        print(f"\n전처리된 데이터 위치: {output_dir}")

        # 용량 확인
        total_size = sum(
            f.stat().st_size for f in output_dir.glob("*.csv") if f.is_file()
        )
        print(f"총 용량: {total_size / 1024 / 1024:.2f} MB")

        return True

    except Exception as e:
        print(f"\n❌ 전처리 실패: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Colab에서 SDD 데이터 다운로드 및 전처리"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/sdd/deathCircle",
        help="원본 데이터 출력 디렉토리",
    )
    parser.add_argument(
        "--converted_dir",
        type=str,
        default="data/sdd/converted",
        help="전처리된 데이터 출력 디렉토리",
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="다운로드 건너뛰기 (이미 다운로드된 경우)",
    )
    parser.add_argument(
        "--skip_preprocess", action="store_true", help="전처리 건너뛰기"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    converted_dir = Path(args.converted_dir)

    # 1. 다운로드
    if not args.skip_download:
        success = download_sdd_in_colab(output_dir)
        if not success:
            print("\n❌ 다운로드 실패")
            return
    else:
        print("⏭️  다운로드 건너뛰기")

    # 2. 전처리
    if not args.skip_preprocess:
        success = preprocess_in_colab(output_dir, converted_dir)
        if not success:
            print("\n❌ 전처리 실패")
            return
    else:
        print("⏭️  전처리 건너뛰기")

    print("\n" + "=" * 80)
    print("✅ 모든 작업 완료!")
    print("=" * 80)
    print(f"\n다음 단계:")
    print(f"1. 전처리된 데이터 확인: ls {converted_dir}")
    print(f"2. 모델 학습 시작: python scripts/training/train_a3tgcn.py")


if __name__ == "__main__":
    main()
