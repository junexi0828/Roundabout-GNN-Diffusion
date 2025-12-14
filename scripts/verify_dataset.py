#!/usr/bin/env python3
"""
INTERACTION Dataset 검증 스크립트
데이터셋이 올바르게 다운로드되었는지 확인합니다.
"""

import os
import pandas as pd
from pathlib import Path

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "interaction"

# 회전교차로 시나리오 목록
ROUNDABOUT_SCENARIOS = [
    "DR_USA_Roundabout_FT",
    "DR_CHN_Roundabout_LN",
    "DR_DEU_Roundabout_OF",
]


def check_scenario(scenario_name: str) -> dict:
    """특정 시나리오의 데이터를 검증합니다."""
    scenario_dir = DATA_DIR / scenario_name

    result = {
        "scenario": scenario_name,
        "exists": scenario_dir.exists(),
        "has_map": False,
        "track_files": [],
        "errors": [],
    }

    if not result["exists"]:
        result["errors"].append(
            f"시나리오 디렉토리가 존재하지 않습니다: {scenario_dir}"
        )
        return result

    # OSM 맵 파일 확인
    map_file = scenario_dir / f"{scenario_name}.osm"
    result["has_map"] = map_file.exists()
    if not result["has_map"]:
        result["errors"].append(f"맵 파일이 없습니다: {map_file}")

    # 궤적 파일 확인
    track_files = list(scenario_dir.glob("vehicle_tracks_*.csv"))
    result["track_files"] = [f.name for f in track_files]

    if not track_files:
        result["errors"].append("궤적 파일이 없습니다.")
    else:
        # 첫 번째 파일의 구조 확인
        try:
            sample_file = track_files[0]
            df = pd.read_csv(sample_file, nrows=5)
            required_columns = ["track_id", "frame_id", "x", "y", "vx", "vy"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                result["errors"].append(f"필수 컬럼이 없습니다: {missing_columns}")
        except Exception as e:
            result["errors"].append(f"파일 읽기 오류: {e}")

    return result


def main():
    """메인 함수"""
    print("=" * 60)
    print("INTERACTION Dataset 검증")
    print("=" * 60)
    print()

    if not DATA_DIR.exists():
        print(f"❌ 데이터 디렉토리가 없습니다: {DATA_DIR}")
        print(f"   데이터셋을 다운로드하여 {DATA_DIR}에 압축을 해제하세요.")
        print(f"   자세한 내용은 data/DATASET_DOWNLOAD.md를 참조하세요.")
        return

    print(f"데이터 디렉토리: {DATA_DIR}\n")

    all_valid = True

    for scenario in ROUNDABOUT_SCENARIOS:
        print(f"시나리오 확인: {scenario}")
        result = check_scenario(scenario)

        if result["exists"]:
            print(f"  ✓ 디렉토리 존재")
            print(f"  {'✓' if result['has_map'] else '✗'} 맵 파일: {result['has_map']}")
            print(f"  ✓ 궤적 파일 수: {len(result['track_files'])}")
            if result["track_files"]:
                print(f"    예시: {result['track_files'][0]}")

            if result["errors"]:
                print(f"  ⚠ 오류:")
                for error in result["errors"]:
                    print(f"    - {error}")
                all_valid = False
            else:
                print(f"  ✓ 검증 완료")
        else:
            print(f"  ✗ 디렉토리 없음")
            all_valid = False

        print()

    print("=" * 60)
    if all_valid:
        print("✓ 모든 시나리오가 올바르게 다운로드되었습니다!")
    else:
        print("⚠ 일부 시나리오에 문제가 있습니다. 위의 오류를 확인하세요.")
    print("=" * 60)


if __name__ == "__main__":
    main()
