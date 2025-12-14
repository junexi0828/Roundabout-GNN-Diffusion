"""
시나리오별 성능 분석 모듈
Normal Merging, Dense Traffic, Aggressive Entry 등 시나리오별 분석
국가별 운전 문화 차이 분석 (미국/중국/독일)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from .metrics import TrajectoryEvaluator, calculate_ade, calculate_fde, calculate_miss_rate


class ScenarioAnalyzer:
    """시나리오별 성능 분석 클래스"""

    def __init__(
        self,
        vehicle_radius: float = 2.5,
        miss_threshold: float = 2.0
    ):
        """
        Args:
            vehicle_radius: 차량 반경
            miss_threshold: Miss Rate 임계값
        """
        self.evaluator = TrajectoryEvaluator(vehicle_radius, miss_threshold)
        self.results = defaultdict(list)

    def classify_scenario(
        self,
        trajectory_data: pd.DataFrame,
        frame_data: pd.DataFrame
    ) -> str:
        """
        시나리오를 분류합니다.

        Args:
            trajectory_data: 전체 궤적 데이터
            frame_data: 현재 프레임 데이터

        Returns:
            시나리오 타입: 'normal_merging', 'dense_traffic', 'aggressive_entry'
        """
        # 차량 밀도 계산
        num_vehicles = len(frame_data)

        # 평균 속도 계산
        if 'vx' in frame_data.columns and 'vy' in frame_data.columns:
            speeds = np.sqrt(frame_data['vx']**2 + frame_data['vy']**2)
            avg_speed = speeds.mean()
            speed_std = speeds.std()
        else:
            avg_speed = 0.0
            speed_std = 0.0

        # 차량 간 평균 거리
        if num_vehicles > 1:
            positions = frame_data[['x', 'y']].values
            distances = []
            for i in range(num_vehicles):
                for j in range(i + 1, num_vehicles):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    distances.append(dist)
            avg_distance = np.mean(distances) if distances else float('inf')
        else:
            avg_distance = float('inf')

        # 시나리오 분류
        if num_vehicles >= 8 and avg_distance < 10.0:
            return 'dense_traffic'
        elif speed_std > 3.0 or avg_speed > 8.0:
            # 속도 변화가 크거나 높은 속도 = 공격적 진입 가능성
            return 'aggressive_entry'
        else:
            return 'normal_merging'

    def get_country_from_scenario(self, scenario_name: str) -> Optional[str]:
        """
        시나리오 이름에서 국가 추출

        Args:
            scenario_name: 시나리오 이름 (예: DR_USA_Roundabout_FT)

        Returns:
            국가 코드: 'USA', 'CHN', 'DEU', None
        """
        scenario_upper = scenario_name.upper()

        if 'USA' in scenario_upper or 'US' in scenario_upper:
            return 'USA'
        elif 'CHN' in scenario_upper or 'CHINA' in scenario_upper:
            return 'CHN'
        elif 'DEU' in scenario_upper or 'GERMANY' in scenario_upper:
            return 'DEU'
        else:
            return None

    def analyze_scenario_type(
        self,
        predicted: np.ndarray,
        ground_truth: np.ndarray,
        scenario_type: str,
        scenario_name: Optional[str] = None
    ) -> Dict[str, float]:
        """
        특정 시나리오 타입에 대한 성능 분석

        Args:
            predicted: 예측 궤적
            ground_truth: 실제 궤적
            scenario_type: 시나리오 타입
            scenario_name: 시나리오 이름

        Returns:
            평가 지표 딕셔너리
        """
        metrics = self.evaluator.evaluate(predicted, ground_truth)
        metrics['scenario_type'] = scenario_type

        if scenario_name:
            metrics['scenario_name'] = scenario_name
            country = self.get_country_from_scenario(scenario_name)
            if country:
                metrics['country'] = country

        return metrics

    def analyze_by_scenario_type(
        self,
        results: List[Dict]
    ) -> pd.DataFrame:
        """
        시나리오 타입별로 결과를 그룹화하여 분석

        Args:
            results: 평가 결과 리스트

        Returns:
            시나리오 타입별 통계 DataFrame
        """
        df = pd.DataFrame(results)

        if 'scenario_type' not in df.columns:
            return pd.DataFrame()

        # 시나리오 타입별 통계
        scenario_stats = df.groupby('scenario_type').agg({
            'ADE': ['mean', 'std', 'min', 'max'],
            'FDE': ['mean', 'std', 'min', 'max'],
            'Miss_Rate': ['mean', 'std'],
            'Collision_Rate': ['mean', 'std']
        }).round(4)

        return scenario_stats

    def analyze_by_country(
        self,
        results: List[Dict]
    ) -> pd.DataFrame:
        """
        국가별로 결과를 그룹화하여 분석

        Args:
            results: 평가 결과 리스트

        Returns:
            국가별 통계 DataFrame
        """
        df = pd.DataFrame(results)

        if 'country' not in df.columns:
            return pd.DataFrame()

        # 국가별 통계
        country_stats = df.groupby('country').agg({
            'ADE': ['mean', 'std', 'min', 'max'],
            'FDE': ['mean', 'std', 'min', 'max'],
            'Miss_Rate': ['mean', 'std'],
            'Collision_Rate': ['mean', 'std']
        }).round(4)

        return country_stats

    def visualize_scenario_comparison(
        self,
        results: List[Dict],
        output_path: Optional[Path] = None
    ):
        """
        시나리오 타입별 성능 비교 시각화

        Args:
            results: 평가 결과 리스트
            output_path: 저장 경로
        """
        df = pd.DataFrame(results)

        if 'scenario_type' not in df.columns:
            print("시나리오 타입 정보가 없습니다.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # ADE 비교
        ax = axes[0, 0]
        scenario_ade = df.groupby('scenario_type')['ADE'].mean()
        scenario_ade.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
        ax.set_ylabel('ADE (m)')
        ax.set_title('Average Displacement Error by Scenario Type')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

        # FDE 비교
        ax = axes[0, 1]
        scenario_fde = df.groupby('scenario_type')['FDE'].mean()
        scenario_fde.plot(kind='bar', ax=ax, color='lightcoral', edgecolor='black')
        ax.set_ylabel('FDE (m)')
        ax.set_title('Final Displacement Error by Scenario Type')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

        # Miss Rate 비교
        ax = axes[1, 0]
        scenario_miss = df.groupby('scenario_type')['Miss_Rate'].mean()
        scenario_miss.plot(kind='bar', ax=ax, color='lightgreen', edgecolor='black')
        ax.set_ylabel('Miss Rate')
        ax.set_title('Miss Rate by Scenario Type')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

        # Collision Rate 비교
        ax = axes[1, 1]
        if 'Collision_Rate' in df.columns:
            scenario_collision = df.groupby('scenario_type')['Collision_Rate'].mean()
            scenario_collision.plot(kind='bar', ax=ax, color='orange', edgecolor='black')
            ax.set_ylabel('Collision Rate')
            ax.set_title('Collision Rate by Scenario Type')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"시나리오 비교 시각화 저장: {output_path}")
        else:
            plt.show()

        plt.close()

    def visualize_country_comparison(
        self,
        results: List[Dict],
        output_path: Optional[Path] = None
    ):
        """
        국가별 성능 비교 시각화

        Args:
            results: 평가 결과 리스트
            output_path: 저장 경로
        """
        df = pd.DataFrame(results)

        if 'country' not in df.columns:
            print("국가 정보가 없습니다.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 국가별 색상 매핑
        country_colors = {
            'USA': '#1f77b4',  # 파란색
            'CHN': '#ff7f0e',  # 주황색
            'DEU': '#2ca02c'   # 초록색
        }

        # ADE 비교
        ax = axes[0, 0]
        country_ade = df.groupby('country')['ADE'].mean()
        colors = [country_colors.get(c, 'gray') for c in country_ade.index]
        country_ade.plot(kind='bar', ax=ax, color=colors, edgecolor='black')
        ax.set_ylabel('ADE (m)')
        ax.set_title('Average Displacement Error by Country')
        ax.tick_params(axis='x', rotation=0)
        ax.grid(True, alpha=0.3, axis='y')

        # FDE 비교
        ax = axes[0, 1]
        country_fde = df.groupby('country')['FDE'].mean()
        colors = [country_colors.get(c, 'gray') for c in country_fde.index]
        country_fde.plot(kind='bar', ax=ax, color=colors, edgecolor='black')
        ax.set_ylabel('FDE (m)')
        ax.set_title('Final Displacement Error by Country')
        ax.tick_params(axis='x', rotation=0)
        ax.grid(True, alpha=0.3, axis='y')

        # Miss Rate 비교
        ax = axes[1, 0]
        country_miss = df.groupby('country')['Miss_Rate'].mean()
        colors = [country_colors.get(c, 'gray') for c in country_miss.index]
        country_miss.plot(kind='bar', ax=ax, color=colors, edgecolor='black')
        ax.set_ylabel('Miss Rate')
        ax.set_title('Miss Rate by Country')
        ax.tick_params(axis='x', rotation=0)
        ax.grid(True, alpha=0.3, axis='y')

        # 박스플롯: 국가별 ADE 분포
        ax = axes[1, 1]
        countries = df['country'].unique()
        data_to_plot = [df[df['country'] == c]['ADE'].values for c in countries]
        bp = ax.boxplot(data_to_plot, labels=countries, patch_artist=True)
        for patch, country in zip(bp['boxes'], countries):
            patch.set_facecolor(country_colors.get(country, 'gray'))
        ax.set_ylabel('ADE (m)')
        ax.set_title('ADE Distribution by Country')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"국가별 비교 시각화 저장: {output_path}")
        else:
            plt.show()

        plt.close()

    def create_comprehensive_report(
        self,
        results: List[Dict],
        output_dir: Path = Path('results')
    ):
        """
        종합 분석 리포트 생성

        Args:
            results: 평가 결과 리스트
            output_dir: 출력 디렉토리
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(results)

        print("=" * 80)
        print("시나리오별 성능 분석 리포트")
        print("=" * 80)

        # 시나리오 타입별 분석
        if 'scenario_type' in df.columns:
            print("\n[시나리오 타입별 성능]")
            scenario_stats = self.analyze_by_scenario_type(results)
            print(scenario_stats)
            scenario_stats.to_csv(output_dir / 'scenario_type_stats.csv')

            # 시각화
            self.visualize_scenario_comparison(
                results,
                output_path=output_dir / 'scenario_type_comparison.png'
            )

        # 국가별 분석
        if 'country' in df.columns:
            print("\n[국가별 성능]")
            country_stats = self.analyze_by_country(results)
            print(country_stats)
            country_stats.to_csv(output_dir / 'country_stats.csv')

            # 시각화
            self.visualize_country_comparison(
                results,
                output_path=output_dir / 'country_comparison.png'
            )

            # 국가별 운전 문화 차이 분석
            print("\n[국가별 운전 문화 차이 분석]")
            self._analyze_driving_culture_differences(df, output_dir)

        # 전체 결과 저장
        df.to_csv(output_dir / 'all_results.csv', index=False)

        print(f"\n리포트 생성 완료: {output_dir}")

    def _analyze_driving_culture_differences(
        self,
        df: pd.DataFrame,
        output_dir: Path
    ):
        """
        국가별 운전 문화 차이 분석

        Args:
            df: 결과 DataFrame
            output_dir: 출력 디렉토리
        """
        if 'country' not in df.columns:
            return

        # 국가별 특성 분석
        culture_analysis = {}

        for country in df['country'].unique():
            country_data = df[df['country'] == country]

            analysis = {
                'num_scenarios': len(country_data),
                'avg_ade': float(country_data['ADE'].mean()),
                'avg_fde': float(country_data['FDE'].mean()),
                'avg_miss_rate': float(country_data['Miss_Rate'].mean()),
                'avg_speed': 0.0,  # 실제 데이터에서 계산 필요
                'aggressiveness_score': 0.0  # Miss Rate와 Collision Rate 기반
            }

            if 'Collision_Rate' in country_data.columns:
                analysis['avg_collision_rate'] = float(country_data['Collision_Rate'].mean())
                # 공격성 점수: Miss Rate + Collision Rate
                analysis['aggressiveness_score'] = (
                    analysis['avg_miss_rate'] + analysis['avg_collision_rate']
                )

            culture_analysis[country] = analysis

        # 결과 출력
        print("\n국가별 운전 특성:")
        for country, analysis in culture_analysis.items():
            print(f"\n{country}:")
            print(f"  시나리오 수: {analysis['num_scenarios']}")
            print(f"  평균 ADE: {analysis['avg_ade']:.4f} m")
            print(f"  평균 FDE: {analysis['avg_fde']:.4f} m")
            print(f"  평균 Miss Rate: {analysis['avg_miss_rate']:.2%}")
            if 'avg_collision_rate' in analysis:
                print(f"  평균 Collision Rate: {analysis['avg_collision_rate']:.2%}")
            print(f"  공격성 점수: {analysis['aggressiveness_score']:.4f}")

        # CSV 저장
        culture_df = pd.DataFrame(culture_analysis).T
        culture_df.to_csv(output_dir / 'driving_culture_analysis.csv')

        # 시각화
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 공격성 점수 비교
        ax = axes[0]
        aggressiveness = {c: a['aggressiveness_score']
                          for c, a in culture_analysis.items()}
        countries = list(aggressiveness.keys())
        scores = list(aggressiveness.values())
        colors = ['#1f77b4' if c == 'USA' else '#ff7f0e' if c == 'CHN' else '#2ca02c'
                  for c in countries]
        ax.bar(countries, scores, color=colors, edgecolor='black')
        ax.set_ylabel('Aggressiveness Score')
        ax.set_title('Driving Aggressiveness by Country')
        ax.grid(True, alpha=0.3, axis='y')

        # ADE vs FDE 산점도 (국가별)
        ax = axes[1]
        for country in df['country'].unique():
            country_data = df[df['country'] == country]
            color = '#1f77b4' if country == 'USA' else '#ff7f0e' if country == 'CHN' else '#2ca02c'
            ax.scatter(
                country_data['ADE'],
                country_data['FDE'],
                label=country,
                color=color,
                alpha=0.6,
                s=100
            )
        ax.set_xlabel('ADE (m)')
        ax.set_ylabel('FDE (m)')
        ax.set_title('ADE vs FDE by Country')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'driving_culture_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()


def analyze_interaction_dataset_scenarios(
    data_dir: Path,
    model_predictions: Optional[Dict[str, np.ndarray]] = None
) -> List[Dict]:
    """
    INTERACTION Dataset의 시나리오들을 분석

    Args:
        data_dir: 데이터 디렉토리
        model_predictions: 모델 예측 결과 {scenario_name: predictions}

    Returns:
        분석 결과 리스트
    """
    analyzer = ScenarioAnalyzer()
    results = []

    # 시나리오 목록
    scenarios = [
        'DR_USA_Roundabout_FT',
        'DR_CHN_Roundabout_LN',
        'DR_DEU_Roundabout_OF'
    ]

    for scenario_name in scenarios:
        scenario_dir = data_dir / scenario_name

        if not scenario_dir.exists():
            continue

        print(f"\n시나리오 분석: {scenario_name}")

        # 실제 구현 시 데이터 로드 및 모델 예측 수행
        # 여기서는 예시로 더미 결과 생성
        if model_predictions and scenario_name in model_predictions:
            predicted = model_predictions[scenario_name]
            # 실제 타겟은 데이터에서 로드
            # ground_truth = load_ground_truth(scenario_dir)

            # 더미 타겟 (실제로는 데이터에서 로드)
            ground_truth = predicted + np.random.randn(*predicted.shape) * 0.1

            # 시나리오 타입 분류 (실제로는 프레임 데이터 기반)
            scenario_type = 'normal_merging'  # 기본값

            # 평가
            metrics = analyzer.analyze_scenario_type(
                predicted,
                ground_truth,
                scenario_type,
                scenario_name
            )

            results.append(metrics)

    return results


def main():
    """테스트용 메인 함수"""
    # 더미 결과 생성
    results = []

    scenarios = [
        ('DR_USA_Roundabout_FT', 'USA', 'normal_merging'),
        ('DR_USA_Roundabout_FT', 'USA', 'dense_traffic'),
        ('DR_CHN_Roundabout_LN', 'CHN', 'normal_merging'),
        ('DR_CHN_Roundabout_LN', 'CHN', 'aggressive_entry'),
        ('DR_DEU_Roundabout_OF', 'DEU', 'normal_merging'),
    ]

    for scenario_name, country, scenario_type in scenarios:
        # 더미 메트릭
        results.append({
            'scenario_name': scenario_name,
            'country': country,
            'scenario_type': scenario_type,
            'ADE': np.random.uniform(0.5, 2.0),
            'FDE': np.random.uniform(1.0, 3.0),
            'Miss_Rate': np.random.uniform(0.1, 0.3),
            'Collision_Rate': np.random.uniform(0.0, 0.1)
        })

    # 분석 실행
    analyzer = ScenarioAnalyzer()
    analyzer.create_comprehensive_report(
        results,
        output_dir=Path('results/scenario_analysis')
    )


if __name__ == "__main__":
    main()

