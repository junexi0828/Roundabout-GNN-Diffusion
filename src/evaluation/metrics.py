"""
궤적 예측 평가 지표 계산
ADE, FDE, Miss Rate, Collision Rate, Lane Violation Rate
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
from pathlib import Path


def calculate_ade(
    predicted: np.ndarray,
    ground_truth: np.ndarray
) -> float:
    """
    Average Displacement Error (ADE) 계산

    예측 경로와 실제 경로 간의 평균 거리 오차

    Args:
        predicted: 예측 궤적 [N, T, 2] 또는 [T, 2]
        ground_truth: 실제 궤적 [N, T, 2] 또는 [T, 2]

    Returns:
        ADE 값 (미터)
    """
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.detach().cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()

    predicted = np.asarray(predicted)
    ground_truth = np.asarray(ground_truth)

    # 차원 확장
    if predicted.ndim == 2:
        predicted = predicted[np.newaxis, :, :]
    if ground_truth.ndim == 2:
        ground_truth = ground_truth[np.newaxis, :, :]

    # 거리 계산: ||predicted - ground_truth||
    diff = predicted - ground_truth
    distances = np.linalg.norm(diff, axis=2)  # [N, T]

    # 평균
    ade = np.mean(distances)

    return float(ade)


def calculate_fde(
    predicted: np.ndarray,
    ground_truth: np.ndarray
) -> float:
    """
    Final Displacement Error (FDE) 계산

    예측 마지막 시점에서의 위치 오차

    Args:
        predicted: 예측 궤적 [N, T, 2] 또는 [T, 2]
        ground_truth: 실제 궤적 [N, T, 2] 또는 [T, 2]

    Returns:
        FDE 값 (미터)
    """
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.detach().cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()

    predicted = np.asarray(predicted)
    ground_truth = np.asarray(ground_truth)

    # 차원 확장
    if predicted.ndim == 2:
        predicted = predicted[np.newaxis, :, :]
    if ground_truth.ndim == 2:
        ground_truth = ground_truth[np.newaxis, :, :]

    # 마지막 시점만 선택
    pred_final = predicted[:, -1, :]  # [N, 2]
    gt_final = ground_truth[:, -1, :]  # [N, 2]

    # 거리 계산
    diff = pred_final - gt_final
    distances = np.linalg.norm(diff, axis=1)  # [N]

    # 평균
    fde = np.mean(distances)

    return float(fde)


def calculate_miss_rate(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    threshold: float = 2.0
) -> float:
    """
    Miss Rate 계산

    예측 경로가 실제 경로와 특정 임계값 이상 벗어난 비율

    Args:
        predicted: 예측 궤적 [N, T, 2] 또는 [T, 2]
        ground_truth: 실제 궤적 [N, T, 2] 또는 [T, 2]
        threshold: 임계값 (미터)

    Returns:
        Miss Rate (0~1)
    """
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.detach().cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()

    predicted = np.asarray(predicted)
    ground_truth = np.asarray(ground_truth)

    # 차원 확장
    if predicted.ndim == 2:
        predicted = predicted[np.newaxis, :, :]
    if ground_truth.ndim == 2:
        ground_truth = ground_truth[np.newaxis, :, :]

    # 거리 계산
    diff = predicted - ground_truth
    distances = np.linalg.norm(diff, axis=2)  # [N, T]

    # 임계값 초과 여부 (어느 시점이든 한 번이라도 초과하면 miss)
    max_distances = np.max(distances, axis=1)  # [N]
    misses = (max_distances > threshold).astype(float)

    miss_rate = np.mean(misses)

    return float(miss_rate)


def calculate_collision_rate(
    predicted_trajectories: List[np.ndarray],
    vehicle_radius: float = 2.5,
    time_threshold: float = 0.1
) -> float:
    """
    Collision Rate 계산

    예측된 궤적 간의 충돌 발생 비율

    Args:
        predicted_trajectories: 예측 궤적 리스트 [traj1, traj2, ...]
                                각 traj는 [T, 2] 형태
        vehicle_radius: 차량 반경 (미터)
        time_threshold: 충돌로 간주할 최소 시간 (초)

    Returns:
        Collision Rate (0~1)
    """
    if len(predicted_trajectories) < 2:
        return 0.0

    num_collisions = 0
    num_pairs = 0

    # 모든 차량 쌍에 대해 검사
    for i in range(len(predicted_trajectories)):
        for j in range(i + 1, len(predicted_trajectories)):
            traj1 = np.asarray(predicted_trajectories[i])
            traj2 = np.asarray(predicted_trajectories[j])

            num_pairs += 1

            # 시간별 거리 계산
            if traj1.shape[0] != traj2.shape[0]:
                min_len = min(traj1.shape[0], traj2.shape[0])
                traj1 = traj1[:min_len]
                traj2 = traj2[:min_len]

            distances = np.linalg.norm(traj1 - traj2, axis=1)

            # 충돌 여부 (거리가 2*radius 이하인 시점이 있으면 충돌)
            collision_mask = distances <= (2 * vehicle_radius)
            collision_duration = np.sum(collision_mask) * time_threshold

            if collision_duration > 0:
                num_collisions += 1

    if num_pairs == 0:
        return 0.0

    collision_rate = num_collisions / num_pairs
    return float(collision_rate)


def calculate_lane_violation_rate(
    predicted: np.ndarray,
    lane_boundaries: Optional[List[np.ndarray]] = None,
    map_data: Optional[Dict] = None
) -> float:
    """
    Lane Violation Rate 계산

    예측 궤적이 도로 경계를 침범하는 비율

    Args:
        predicted: 예측 궤적 [N, T, 2] 또는 [T, 2]
        lane_boundaries: 차선 경계 리스트 (각 경계는 [M, 2] 형태)
        map_data: 맵 데이터 (선택사항)

    Returns:
        Lane Violation Rate (0~1)
    """
    if lane_boundaries is None and map_data is None:
        # 맵 정보가 없으면 계산 불가
        return 0.0

    predicted = np.asarray(predicted)

    # 차원 확장
    if predicted.ndim == 2:
        predicted = predicted[np.newaxis, :, :]

    num_trajectories, num_timesteps, _ = predicted.shape

    violations = 0
    total_points = 0

    # 간단한 구현: 각 궤적 포인트가 차선 경계 내부에 있는지 확인
    # 실제로는 더 정교한 기하학적 검사가 필요
    for traj_idx in range(num_trajectories):
        traj = predicted[traj_idx]  # [T, 2]

        for t in range(num_timesteps):
            point = traj[t]
            total_points += 1

            # 차선 경계 검사 (간단한 구현)
            # 실제로는 shapely 등을 사용하여 정교한 검사 필요
            is_violation = False

            if lane_boundaries:
                # 각 차선 경계와의 거리 계산
                for boundary in lane_boundaries:
                    # 간단한 거리 기반 검사
                    distances = np.linalg.norm(boundary - point, axis=1)
                    min_dist = np.min(distances)

                    # 임계값 이하이면 경계 위에 있음 (위반 아님)
                    # 임계값 초과이면 차선 밖 (위반)
                    if min_dist > 1.0:  # 임계값 (미터)
                        is_violation = True
                        break

            if is_violation:
                violations += 1

    if total_points == 0:
        return 0.0

    violation_rate = violations / total_points
    return float(violation_rate)


class TrajectoryEvaluator:
    """궤적 예측 평가 클래스"""

    def __init__(
        self,
        vehicle_radius: float = 2.5,
        miss_threshold: float = 2.0
    ):
        """
        Args:
            vehicle_radius: 차량 반경 (충돌 검사용)
            miss_threshold: Miss Rate 계산용 임계값 (미터)
        """
        self.vehicle_radius = vehicle_radius
        self.miss_threshold = miss_threshold

    def evaluate(
        self,
        predicted: Union[np.ndarray, torch.Tensor],
        ground_truth: Union[np.ndarray, torch.Tensor],
        predicted_trajectories: Optional[List[np.ndarray]] = None,
        lane_boundaries: Optional[List[np.ndarray]] = None
    ) -> Dict[str, float]:
        """
        모든 평가 지표를 계산합니다.

        Args:
            predicted: 예측 궤적 [N, T, 2] 또는 [T, 2]
            ground_truth: 실제 궤적 [N, T, 2] 또는 [T, 2]
            predicted_trajectories: 개별 예측 궤적 리스트 (충돌 검사용)
            lane_boundaries: 차선 경계 (위반 검사용)

        Returns:
            평가 지표 딕셔너리
        """
        metrics = {}

        # ADE
        metrics['ADE'] = calculate_ade(predicted, ground_truth)

        # FDE
        metrics['FDE'] = calculate_fde(predicted, ground_truth)

        # Miss Rate
        metrics['Miss_Rate'] = calculate_miss_rate(
            predicted, ground_truth, self.miss_threshold
        )

        # Collision Rate
        if predicted_trajectories is not None:
            metrics['Collision_Rate'] = calculate_collision_rate(
                predicted_trajectories, self.vehicle_radius
            )
        else:
            metrics['Collision_Rate'] = 0.0

        # Lane Violation Rate
        metrics['Lane_Violation_Rate'] = calculate_lane_violation_rate(
            predicted, lane_boundaries
        )

        return metrics

    def evaluate_batch(
        self,
        predictions: List[np.ndarray],
        ground_truths: List[np.ndarray],
        lane_boundaries: Optional[List[np.ndarray]] = None
    ) -> Dict[str, float]:
        """
        배치 단위 평가

        Args:
            predictions: 예측 궤적 리스트
            ground_truths: 실제 궤적 리스트
            lane_boundaries: 차선 경계

        Returns:
            평균 평가 지표
        """
        all_metrics = []

        for pred, gt in zip(predictions, ground_truths):
            metrics = self.evaluate(
                pred, gt,
                predicted_trajectories=[pred] if len(predictions) > 1 else None,
                lane_boundaries=lane_boundaries
            )
            all_metrics.append(metrics)

        # 평균 계산
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

        return avg_metrics


def print_metrics(metrics: Dict[str, float]):
    """평가 지표를 보기 좋게 출력"""
    print("=" * 60)
    print("평가 지표")
    print("=" * 60)

    for key, value in metrics.items():
        if 'Rate' in key:
            print(f"{key:25s}: {value:.4f} ({value*100:.2f}%)")
        else:
            print(f"{key:25s}: {value:.4f} m")

    print("=" * 60)


def save_metrics(
    metrics: Dict[str, float],
    output_path: Path,
    scenario_name: Optional[str] = None
):
    """평가 지표를 CSV로 저장"""
    df = pd.DataFrame([metrics])

    if scenario_name:
        df['scenario'] = scenario_name

    # 기존 파일이 있으면 추가
    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        df = pd.concat([existing_df, df], ignore_index=True)

    df.to_csv(output_path, index=False)
    print(f"평가 지표 저장 완료: {output_path}")


def main():
    """테스트용 메인 함수"""
    # 더미 데이터 생성
    np.random.seed(42)

    num_trajectories = 10
    num_timesteps = 50

    # 예측 궤적 (약간의 노이즈 추가)
    predicted = np.random.randn(num_trajectories, num_timesteps, 2) * 0.5
    ground_truth = predicted + np.random.randn(num_trajectories, num_timesteps, 2) * 0.1

    # 평가
    evaluator = TrajectoryEvaluator()
    metrics = evaluator.evaluate(predicted, ground_truth)

    print_metrics(metrics)

    # 배치 평가
    predictions = [predicted[i] for i in range(num_trajectories)]
    ground_truths = [ground_truth[i] for i in range(num_trajectories)]

    batch_metrics = evaluator.evaluate_batch(predictions, ground_truths)
    print("\n배치 평균 지표:")
    print_metrics(batch_metrics)


if __name__ == "__main__":
    main()

