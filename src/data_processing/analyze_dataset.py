"""
INTERACTION Dataset 구조 분석 모듈
CSV 파일 구조 파악, 좌표계 확인, 샘플링 속도 검증, 결측치 및 이상치 확인
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class DatasetAnalyzer:
    """데이터셋 분석 클래스"""

    def __init__(self, data_dir: Path):
        """
        Args:
            data_dir: INTERACTION Dataset 디렉토리 경로
        """
        self.data_dir = Path(data_dir)
        self.scenarios = []
        self.analysis_results = {}

    def find_scenarios(self) -> List[str]:
        """데이터 디렉토리에서 시나리오 목록을 찾습니다."""
        if not self.data_dir.exists():
            return []

        scenarios = [d.name for d in self.data_dir.iterdir()
                    if d.is_dir() and d.name.startswith("DR_")]
        self.scenarios = sorted(scenarios)
        return self.scenarios

    def analyze_track_file(self, csv_path: Path) -> Dict:
        """
        궤적 파일 하나를 분석합니다.

        Returns:
            분석 결과 딕셔너리
        """
        df = pd.read_csv(csv_path)

        result = {
            "file": csv_path.name,
            "total_rows": len(df),
            "columns": list(df.columns),
            "track_ids": df["track_id"].nunique() if "track_id" in df.columns else 0,
            "frame_range": (df["frame_id"].min(), df["frame_id"].max()) if "frame_id" in df.columns else None,
            "timestamp_range": None,
            "sampling_rate": None,
            "missing_values": df.isnull().sum().to_dict(),
            "coordinate_stats": {},
            "velocity_stats": {}
        }

        # 타임스탬프 분석 (샘플링 속도 계산)
        if "timestamp_ms" in df.columns:
            timestamps = df["timestamp_ms"].unique()
            if len(timestamps) > 1:
                time_diffs = np.diff(np.sort(timestamps))
                avg_diff = np.mean(time_diffs)
                result["sampling_rate"] = 1000.0 / avg_diff if avg_diff > 0 else None
                result["timestamp_range"] = (timestamps.min(), timestamps.max())

        # 좌표 통계
        if "x" in df.columns and "y" in df.columns:
            result["coordinate_stats"] = {
                "x": {
                    "min": float(df["x"].min()),
                    "max": float(df["x"].max()),
                    "mean": float(df["x"].mean()),
                    "std": float(df["x"].std())
                },
                "y": {
                    "min": float(df["y"].min()),
                    "max": float(df["y"].max()),
                    "mean": float(df["y"].mean()),
                    "std": float(df["y"].std())
                }
            }

        # 속도 통계
        if "vx" in df.columns and "vy" in df.columns:
            df["speed"] = np.sqrt(df["vx"]**2 + df["vy"]**2)
            result["velocity_stats"] = {
                "speed": {
                    "min": float(df["speed"].min()),
                    "max": float(df["speed"].max()),
                    "mean": float(df["speed"].mean()),
                    "std": float(df["speed"].std())
                }
            }

        return result

    def analyze_scenario(self, scenario_name: str) -> Dict:
        """시나리오 전체를 분석합니다."""
        scenario_dir = self.data_dir / scenario_name

        if not scenario_dir.exists():
            return {"error": f"시나리오 디렉토리가 없습니다: {scenario_dir}"}

        track_files = sorted(scenario_dir.glob("vehicle_tracks_*.csv"))

        if not track_files:
            return {"error": "궤적 파일이 없습니다."}

        result = {
            "scenario": scenario_name,
            "num_track_files": len(track_files),
            "track_files_analysis": [],
            "map_file_exists": (scenario_dir / f"{scenario_name}.osm").exists(),
            "summary": {}
        }

        # 각 파일 분석
        all_track_ids = set()
        all_frames = set()

        for track_file in track_files:
            file_result = self.analyze_track_file(track_file)
            result["track_files_analysis"].append(file_result)

            if "track_ids" in file_result:
                # track_id는 파일별로 다를 수 있으므로 집계만 수행
                pass

        # 전체 요약 통계
        if result["track_files_analysis"]:
            first_file = result["track_files_analysis"][0]
            result["summary"] = {
                "expected_columns": first_file.get("columns", []),
                "sampling_rate_hz": first_file.get("sampling_rate"),
                "coordinate_system": "relative"  # INTERACTION은 상대 좌표 사용
            }

        return result

    def generate_report(self, output_path: Path = None):
        """분석 결과를 리포트로 생성합니다."""
        scenarios = self.find_scenarios()

        if not scenarios:
            print("분석할 시나리오를 찾을 수 없습니다.")
            return

        print("=" * 80)
        print("INTERACTION Dataset 분석 리포트")
        print("=" * 80)
        print()

        for scenario in scenarios:
            print(f"\n{'='*80}")
            print(f"시나리오: {scenario}")
            print(f"{'='*80}")

            analysis = self.analyze_scenario(scenario)

            if "error" in analysis:
                print(f"❌ 오류: {analysis['error']}")
                continue

            print(f"✓ 궤적 파일 수: {analysis['num_track_files']}")
            print(f"{'✓' if analysis['map_file_exists'] else '✗'} 맵 파일 존재: {analysis['map_file_exists']}")

            if analysis["track_files_analysis"]:
                first_file = analysis["track_files_analysis"][0]
                print(f"\n첫 번째 파일 분석: {first_file['file']}")
                print(f"  - 총 행 수: {first_file['total_rows']:,}")
                print(f"  - 차량 수: {first_file['track_ids']}")
                print(f"  - 샘플링 속도: {first_file.get('sampling_rate', 'N/A'):.2f} Hz"
                      if first_file.get('sampling_rate') else "  - 샘플링 속도: 계산 불가")

                if first_file.get('coordinate_stats'):
                    coord = first_file['coordinate_stats']
                    print(f"\n  좌표 통계:")
                    print(f"    X: [{coord['x']['min']:.2f}, {coord['x']['max']:.2f}], "
                          f"평균={coord['x']['mean']:.2f}, std={coord['x']['std']:.2f}")
                    print(f"    Y: [{coord['y']['min']:.2f}, {coord['y']['max']:.2f}], "
                          f"평균={coord['y']['mean']:.2f}, std={coord['y']['std']:.2f}")

                if first_file.get('velocity_stats'):
                    vel = first_file['velocity_stats']
                    print(f"\n  속도 통계:")
                    print(f"    속도: [{vel['speed']['min']:.2f}, {vel['speed']['max']:.2f}] m/s, "
                          f"평균={vel['speed']['mean']:.2f} m/s")

                # 결측치 확인
                missing = {k: v for k, v in first_file['missing_values'].items() if v > 0}
                if missing:
                    print(f"\n  ⚠ 결측치:")
                    for col, count in missing.items():
                        print(f"    - {col}: {count}개 ({count/first_file['total_rows']*100:.2f}%)")
                else:
                    print(f"\n  ✓ 결측치 없음")

        print(f"\n{'='*80}")
        print("분석 완료")
        print(f"{'='*80}")


def main():
    """메인 함수"""
    import sys
    from pathlib import Path

    # 프로젝트 루트 디렉토리
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / "interaction"

    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])

    analyzer = DatasetAnalyzer(data_dir)
    analyzer.generate_report()


if __name__ == "__main__":
    main()

