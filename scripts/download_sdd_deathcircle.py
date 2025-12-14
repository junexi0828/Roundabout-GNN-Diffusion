"""
Stanford Drone Dataset (SDD) Death Circle 다운로드 스크립트
승인 불필요, GitHub에서 바로 다운로드 가능한 회전교차로 데이터셋
"""

import subprocess
import sys
from pathlib import Path
import shutil

def download_sdd_deathcircle(output_dir: Path):
    """
    SDD Death Circle 데이터 다운로드

    Args:
        output_dir: 출력 디렉토리
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Stanford Drone Dataset - Death Circle 다운로드")
    print("=" * 80)
    print("\n⚠️  주의사항:")
    print("1. GitHub 리포지토리에서 어노테이션 파일만 다운로드")
    print("2. 비디오 파일은 제외 (대용량)")
    print("3. 승인 불필요, 바로 다운로드 가능")
    print("\n다운로드 시작...")

    # GitHub 리포지토리 URL
    repo_url = "https://github.com/flclain/StanfordDroneDataset.git"
    temp_dir = output_dir.parent / "temp_sdd"

    try:
        # 임시 디렉토리에 클론
        if temp_dir.exists():
            print(f"기존 임시 디렉토리 삭제: {temp_dir}")
            shutil.rmtree(temp_dir)

        print(f"\nGitHub 리포지토리 클론 중: {repo_url}")
        subprocess.run(['git', 'clone', repo_url, str(temp_dir)],
                      check=True, capture_output=True)

        # Death Circle 데이터 확인
        deathcircle_dir = temp_dir / "deathCircle"

        if not deathcircle_dir.exists():
            # 다른 가능한 경로 확인
            possible_paths = [
                temp_dir / "DeathCircle",
                temp_dir / "death_circle",
                temp_dir / "Death_Circle",
            ]

            for path in possible_paths:
                if path.exists():
                    deathcircle_dir = path
                    break
            else:
                # 디렉토리 구조 확인
                print("\n디렉토리 구조 확인:")
                if temp_dir.exists():
                    for item in temp_dir.iterdir():
                        if item.is_dir():
                            print(f"  - {item.name}")
                raise FileNotFoundError("Death Circle 디렉토리를 찾을 수 없습니다.")

        print(f"\n✓ Death Circle 디렉토리 발견: {deathcircle_dir}")

        # 어노테이션 파일 복사
        annotation_files = list(deathcircle_dir.glob("**/annotations.txt"))
        annotation_files += list(deathcircle_dir.glob("**/*.txt"))

        if not annotation_files:
            print("\n⚠️  어노테이션 파일을 찾을 수 없습니다.")
            print("디렉토리 내용:")
            for item in deathcircle_dir.iterdir():
                print(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")
        else:
            print(f"\n✓ 어노테이션 파일 {len(annotation_files)}개 발견")

            # 출력 디렉토리에 복사
            for ann_file in annotation_files:
                rel_path = ann_file.relative_to(deathcircle_dir)
                dest_path = output_dir / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ann_file, dest_path)
                print(f"  복사: {ann_file.name} -> {dest_path}")

        # 호모그래피 행렬 파일 확인 (H.txt)
        h_files = list(deathcircle_dir.glob("**/H.txt"))
        if h_files:
            print(f"\n✓ 호모그래피 행렬 파일 {len(h_files)}개 발견")
            for h_file in h_files:
                rel_path = h_file.relative_to(deathcircle_dir)
                dest_path = output_dir / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(h_file, dest_path)
                print(f"  복사: {h_file.name} -> {dest_path}")

        # 임시 디렉토리 정리
        print(f"\n임시 디렉토리 정리 중...")
        shutil.rmtree(temp_dir)

        print(f"\n{'='*80}")
        print("✓ 다운로드 완료!")
        print(f"{'='*80}")
        print(f"\n데이터 위치: {output_dir}")
        print("\n다음 단계:")
        print("1. 데이터 포맷 확인: python scripts/verify_sdd_data.py")
        print("2. 전처리: python src/data_processing/preprocess_sdd.py")
        print("3. 모델 학습: python src/training/train.py --dataset sdd")

        return True

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Git 클론 실패: {e}")
        print("\n해결 방법:")
        print("1. Git 설치 확인: git --version")
        print("2. 인터넷 연결 확인")
        print("3. 리포지토리 URL 확인: https://github.com/flclain/StanfordDroneDataset")
        return False
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "data" / "sdd" / "deathCircle"

    success = download_sdd_deathcircle(output_dir)

    if success:
        print("\n✓ SDD Death Circle 데이터셋 준비 완료!")
        print("  회전교차로 상호작용 예측 연구를 시작할 수 있습니다.")

if __name__ == "__main__":
    main()

