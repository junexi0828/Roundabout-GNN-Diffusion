"""
시나리오 분류 모듈
프레임 데이터를 기반으로 시나리오 타입을 자동 분류
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ScenarioCharacteristics:
    """시나리오 특성 데이터 클래스"""
    num_vehicles: int
    avg_speed: float
    speed_std: float
    avg_distance: float
    min_distance: float
    traffic_density: float
    speed_variance: float


class ScenarioClassifier:
    """시나리오 분류 클래스"""

    def __init__(
        self,
        dense_threshold: int = 8,  # 밀집 교통으로 간주할 최소 차량 수
        dense_distance: float = 10.0,  # 밀집 교통으로 간주할 최대 평균 거리 (m)
        aggressive_speed_std: float = 3.0,  # 공격적 진입으로 간주할 속도 표준편차
        aggressive_avg_speed: float = 8.0  # 공격적 진입으로 간주할 평균 속도 (m/s)
    ):
        """
        Args:
            dense_threshold: 밀집 교통 임계값
            dense_distance: 밀집 교통 거리 임계값
            aggressive_speed_std: 공격적 진입 속도 표준편차 임계값
            aggressive_avg_speed: 공격적 진입 평균 속도 임계값
        """
        self.dense_threshold = dense_threshold
        self.dense_distance = dense_distance
        self.aggressive_speed_std = aggressive_speed_std
        self.aggressive_avg_speed = aggressive_avg_speed

    def extract_characteristics(
        self,
        frame_data: pd.DataFrame
    ) -> ScenarioCharacteristics:
        """
        프레임 데이터에서 시나리오 특성 추출

        Args:
            frame_data: 프레임 데이터

        Returns:
            시나리오 특성
        """
        num_vehicles = len(frame_data)

        # 속도 계산
        if 'vx' in frame_data.columns and 'vy' in frame_data.columns:
            speeds = np.sqrt(frame_data['vx']**2 + frame_data['vy']**2)
            avg_speed = float(speeds.mean())
            speed_std = float(speeds.std())
            speed_variance = float(speeds.var())
        else:
            avg_speed = 0.0
            speed_std = 0.0
            speed_variance = 0.0

        # 거리 계산
        if num_vehicles > 1 and 'x' in frame_data.columns and 'y' in frame_data.columns:
            positions = frame_data[['x', 'y']].values
            distances = []
            for i in range(num_vehicles):
                for j in range(i + 1, num_vehicles):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    distances.append(dist)

            if distances:
                avg_distance = float(np.mean(distances))
                min_distance = float(np.min(distances))
            else:
                avg_distance = float('inf')
                min_distance = float('inf')
        else:
            avg_distance = float('inf')
            min_distance = float('inf')

        # 교통 밀도 (차량 수 / 면적, 간단히 차량 수로 근사)
        traffic_density = float(num_vehicles)

        return ScenarioCharacteristics(
            num_vehicles=num_vehicles,
            avg_speed=avg_speed,
            speed_std=speed_std,
            avg_distance=avg_distance,
            min_distance=min_distance,
            traffic_density=traffic_density,
            speed_variance=speed_variance
        )

    def classify(
        self,
        frame_data: pd.DataFrame
    ) -> str:
        """
        프레임 데이터를 기반으로 시나리오 타입 분류

        Args:
            frame_data: 프레임 데이터

        Returns:
            시나리오 타입: 'normal_merging', 'dense_traffic', 'aggressive_entry'
        """
        characteristics = self.extract_characteristics(frame_data)

        # 밀집 교통 판단
        if (characteristics.num_vehicles >= self.dense_threshold and
            characteristics.avg_distance < self.dense_distance):
            return 'dense_traffic'

        # 공격적 진입 판단
        if (characteristics.speed_std > self.aggressive_speed_std or
            characteristics.avg_speed > self.aggressive_avg_speed):
            return 'aggressive_entry'

        # 기본: 정상 합류
        return 'normal_merging'

    def classify_trajectory_data(
        self,
        trajectory_data: pd.DataFrame
    ) -> Dict[str, List[int]]:
        """
        전체 궤적 데이터를 프레임별로 분류

        Args:
            trajectory_data: 전체 궤적 데이터 (frame_id 컬럼 포함)

        Returns:
            시나리오 타입별 프레임 ID 리스트
        """
        frame_ids = sorted(trajectory_data['frame_id'].unique())

        classification = {
            'normal_merging': [],
            'dense_traffic': [],
            'aggressive_entry': []
        }

        for frame_id in frame_ids:
            frame_data = trajectory_data[trajectory_data['frame_id'] == frame_id]
            scenario_type = self.classify(frame_data)
            classification[scenario_type].append(frame_id)

        return classification

    def get_scenario_statistics(
        self,
        trajectory_data: pd.DataFrame
    ) -> Dict[str, Dict]:
        """
        시나리오 타입별 통계 계산

        Args:
            trajectory_data: 전체 궤적 데이터

        Returns:
            시나리오 타입별 통계
        """
        classification = self.classify_trajectory_data(trajectory_data)

        stats = {}

        for scenario_type, frame_ids in classification.items():
            if not frame_ids:
                continue

            # 해당 프레임들의 데이터
            scenario_frames = trajectory_data[
                trajectory_data['frame_id'].isin(frame_ids)
            ]

            # 통계 계산
            if 'vx' in scenario_frames.columns and 'vy' in scenario_frames.columns:
                speeds = np.sqrt(
                    scenario_frames['vx']**2 + scenario_frames['vy']**2
                )
                avg_speed = float(speeds.mean())
                max_speed = float(speeds.max())
            else:
                avg_speed = 0.0
                max_speed = 0.0

            stats[scenario_type] = {
                'num_frames': len(frame_ids),
                'num_vehicles': scenario_frames['track_id'].nunique(),
                'avg_speed': avg_speed,
                'max_speed': max_speed,
                'percentage': len(frame_ids) / len(trajectory_data['frame_id'].unique()) * 100
            }

        return stats


def main():
    """테스트용 메인 함수"""
    # 더미 데이터 생성
    data = []
    for frame_id in range(100):
        num_vehicles = np.random.randint(3, 10)
        for track_id in range(num_vehicles):
            data.append({
                'frame_id': frame_id,
                'track_id': track_id,
                'x': np.random.randn() * 10,
                'y': np.random.randn() * 10,
                'vx': np.random.randn() * 3,
                'vy': np.random.randn() * 3
            })

    trajectory_data = pd.DataFrame(data)

    # 분류 실행
    classifier = ScenarioClassifier()
    classification = classifier.classify_trajectory_data(trajectory_data)

    print("시나리오 분류 결과:")
    for scenario_type, frame_ids in classification.items():
        print(f"  {scenario_type}: {len(frame_ids)}개 프레임 ({len(frame_ids)/100*100:.1f}%)")

    # 통계
    stats = classifier.get_scenario_statistics(trajectory_data)
    print("\n시나리오별 통계:")
    for scenario_type, stat in stats.items():
        print(f"\n{scenario_type}:")
        for key, value in stat.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

