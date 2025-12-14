"""
SDD Death Circle 데이터 검증 스크립트
다운로드된 데이터의 구조와 포맷 확인
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def verify_sdd_data(data_dir: Path):
    """
    SDD Death Circle 데이터 검증

    Args:
        data_dir: SDD 데이터 디렉토리
    """
    print("=" * 80)
    print("SDD Death Circle 데이터 검증")
    print("=" * 80)

    data_dir = Path(data_dir)

    if not data_dir.exists():
        print(f"\n❌ 데이터 디렉토리가 없습니다: {data_dir}")
        print("\n먼저 다운로드를 실행하세요:")
        print("  python scripts/download_sdd_deathcircle.py")
        return False

    # 어노테이션 파일 찾기
    annotation_files = list(data_dir.glob("**/annotations.txt"))

    if not annotation_files:
        # 다른 텍스트 파일 확인
        txt_files = list(data_dir.glob("**/*.txt"))
        print(f"\n⚠️  annotations.txt를 찾을 수 없습니다.")
        print(f"   발견된 텍스트 파일: {len(txt_files)}개")
        if txt_files:
            print("   파일 목록:")
            for f in txt_files[:5]:
                print(f"     - {f}")
        return False

    print(f"\n✓ 어노테이션 파일 {len(annotation_files)}개 발견")

    # 첫 번째 파일 분석
    sample_file = annotation_files[0]
    print(f"\n샘플 파일 분석: {sample_file.name}")

    try:
        # SDD 포맷: 공백 구분 텍스트
        # 컬럼: track_id, xmin, ymin, xmax, ymax, frame, lost, occluded, generated, label
        df = pd.read_csv(sample_file, sep=' ', header=None,
                        names=['track_id', 'xmin', 'ymin', 'xmax', 'ymax',
                               'frame', 'lost', 'occluded', 'generated', 'label'])

        print(f"\n데이터 통계:")
        print(f"  총 행 수: {len(df):,}")
        print(f"  트랙 수: {df['track_id'].nunique()}")
        print(f"  프레임 범위: {df['frame'].min()} ~ {df['frame'].max()}")
        print(f"  에이전트 타입: {df['label'].unique()}")

        # 중심점 계산 예시
        df['cx'] = (df['xmin'] + df['xmax']) / 2
        df['cy'] = (df['ymin'] + df['ymax']) / 2

        print(f"\n좌표 통계 (픽셀):")
        print(f"  X: [{df['cx'].min():.1f}, {df['cx'].max():.1f}]")
        print(f"  Y: [{df['cy'].min():.1f}, {df['cy'].max():.1f}]")

        # 호모그래피 행렬 확인
        h_files = list(data_dir.glob("**/H.txt"))
        if h_files:
            print(f"\n✓ 호모그래피 행렬 파일 발견: {len(h_files)}개")
            h_matrix = np.loadtxt(h_files[0])
            print(f"  행렬 형태: {h_matrix.shape}")
        else:
            print(f"\n⚠️  호모그래피 행렬 파일 없음 (픽셀→미터 변환 필요)")

        print(f"\n{'='*80}")
        print("✓ 데이터 검증 완료")
        print(f"{'='*80}")
        print("\n다음 단계:")
        print("1. 호모그래피 변환 적용 (픽셀 → 미터)")
        print("2. 프로젝트 포맷으로 변환")
        print("3. 전처리 파이프라인 적용")

        return True

    except Exception as e:
        print(f"\n❌ 파일 읽기 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "sdd" / "deathCircle"

    verify_sdd_data(data_dir)

if __name__ == "__main__":
    main()

