"""
안전 분석 파이프라인
전체 시나리오에 대한 안전 지표 분석 및 리포트 생성
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from .safety_metrics import (
    SafetyMetricsCalculator,
    SpatialHasher,
    VehicleState,
    analyze_frame_safety
)


class SafetyAnalyzer:
    """안전 분석 클래스"""

    def __init__(self, vehicle_radius: float = 2.5):
        """
        Args:
            vehicle_radius: 차량 반경 (m)
        """
        self.calculator = SafetyMetricsCalculator(vehicle_radius)
        self.spatial_hasher = SpatialHasher(cell_size=10.0)
        self.results = []

    def analyze_scenario(
        self,
        trajectory_data: pd.DataFrame,
        output_dir: Optional[Path] = None
    ) -> Dict:
        """
        시나리오 전체를 분석합니다.

        Args:
            trajectory_data: 전체 궤적 데이터 (frame_id 컬럼 포함)
            output_dir: 결과 저장 디렉토리

        Returns:
            분석 결과 딕셔너리
        """
        frame_ids = sorted(trajectory_data['frame_id'].unique())

        all_results = []
        critical_events = []

        print(f"프레임 수: {len(frame_ids)}")

        for frame_id in tqdm(frame_ids, desc="안전 분석"):
            frame_data = trajectory_data[trajectory_data['frame_id'] == frame_id]

            if len(frame_data) < 2:
                continue

            # 프레임별 안전 분석
            frame_results = analyze_frame_safety(
                frame_data,
                self.calculator,
                self.spatial_hasher
            )

            frame_results['frame_id'] = frame_id
            all_results.append(frame_results)

            # 위험 상황 기록
            critical = frame_results[frame_results['is_critical']]
            if len(critical) > 0:
                critical_events.append({
                    'frame_id': frame_id,
                    'num_critical': len(critical),
                    'min_ttc': critical['ttc'].min(),
                    'max_drac': critical['drac'].max()
                })

        # 결과 통합
        if all_results:
            combined_results = pd.concat(all_results, ignore_index=True)
        else:
            combined_results = pd.DataFrame()

        # 통계 계산
        stats = self._calculate_statistics(combined_results, critical_events)

        # 결과 저장
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # CSV 저장
            combined_results.to_csv(
                output_dir / 'safety_metrics.csv',
                index=False
            )

            # 통계 저장
            stats_df = pd.DataFrame([stats])
            stats_df.to_csv(
                output_dir / 'safety_statistics.csv',
                index=False
            )

            # 시각화
            self._visualize_results(combined_results, output_dir)

        return {
            'results': combined_results,
            'statistics': stats,
            'critical_events': pd.DataFrame(critical_events) if critical_events else pd.DataFrame()
        }

    def _calculate_statistics(
        self,
        results: pd.DataFrame,
        critical_events: List[Dict]
    ) -> Dict:
        """통계 계산"""
        if len(results) == 0:
            return {}

        stats = {
            'total_pairs_analyzed': len(results),
            'critical_pairs': len(results[results['is_critical']]),
            'critical_ratio': len(results[results['is_critical']]) / len(results) if len(results) > 0 else 0
        }

        # TTC 통계
        valid_ttc = results[results['ttc'].notna()]['ttc']
        if len(valid_ttc) > 0:
            stats['ttc_mean'] = valid_ttc.mean()
            stats['ttc_min'] = valid_ttc.min()
            stats['ttc_max'] = valid_ttc.max()
            stats['ttc_std'] = valid_ttc.std()
            stats['ttc_below_3s'] = len(valid_ttc[valid_ttc < 3.0])
            stats['ttc_below_5s'] = len(valid_ttc[valid_ttc < 5.0])

        # DRAC 통계
        valid_drac = results[results['drac'].notna() & (results['drac'] != np.inf)]['drac']
        if len(valid_drac) > 0:
            stats['drac_mean'] = valid_drac.mean()
            stats['drac_max'] = valid_drac.max()
            stats['drac_above_5ms2'] = len(valid_drac[valid_drac > 5.0])  # 5 m/s² 이상

        # 거리 통계
        stats['distance_mean'] = results['distance'].mean()
        stats['distance_min'] = results['distance'].min()

        # 위험 이벤트 수
        stats['num_critical_events'] = len(critical_events)

        return stats

    def _visualize_results(self, results: pd.DataFrame, output_dir: Path):
        """결과 시각화"""
        sns.set_style("whitegrid")

        # TTC 분포
        valid_ttc = results[results['ttc'].notna()]['ttc']
        if len(valid_ttc) > 0:
            plt.figure(figsize=(10, 6))
            plt.hist(valid_ttc, bins=50, edgecolor='black')
            plt.xlabel('TTC (초)')
            plt.ylabel('빈도')
            plt.title('Time-to-Collision 분포')
            plt.axvline(x=3.0, color='r', linestyle='--', label='위험 임계값 (3초)')
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / 'ttc_distribution.png', dpi=300)
            plt.close()

        # DRAC 분포
        valid_drac = results[
            results['drac'].notna() & (results['drac'] != np.inf)
        ]['drac']
        if len(valid_drac) > 0:
            plt.figure(figsize=(10, 6))
            plt.hist(valid_drac, bins=50, edgecolor='black')
            plt.xlabel('DRAC (m/s²)')
            plt.ylabel('빈도')
            plt.title('Deceleration Rate to Avoid Collision 분포')
            plt.axvline(x=5.0, color='r', linestyle='--', label='임계값 (5 m/s²)')
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / 'drac_distribution.png', dpi=300)
            plt.close()

        # 위험 상황 히트맵 (프레임별)
        if 'frame_id' in results.columns:
            critical_by_frame = results.groupby('frame_id')['is_critical'].sum()
            if len(critical_by_frame) > 0:
                plt.figure(figsize=(12, 6))
                critical_by_frame.plot(kind='bar')
                plt.xlabel('Frame ID')
                plt.ylabel('위험 차량 쌍 수')
                plt.title('프레임별 위험 상황')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(output_dir / 'critical_by_frame.png', dpi=300)
                plt.close()


def main():
    """테스트용 메인 함수"""
    # 더미 데이터 생성
    data = []
    for frame_id in range(100):
        for track_id in range(3):
            data.append({
                'frame_id': frame_id,
                'track_id': track_id,
                'x': track_id * 5.0 + frame_id * 0.1,
                'y': 0.0,
                'vx': 5.0,
                'vy': 0.0,
                'width': 2.0,
                'length': 4.5
            })

    trajectory_data = pd.DataFrame(data)

    # 분석 실행
    analyzer = SafetyAnalyzer()
    results = analyzer.analyze_scenario(
        trajectory_data,
        output_dir=Path('results/safety_analysis')
    )

    print("\n분석 완료!")
    print(f"총 분석 쌍 수: {results['statistics'].get('total_pairs_analyzed', 0)}")
    print(f"위험 쌍 수: {results['statistics'].get('critical_pairs', 0)}")


if __name__ == "__main__":
    main()

