"""
Diffusion 모델 평가 지표
다중 모달리티 궤적 예측을 위한 특화 지표
"""

import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from scipy.spatial.distance import cdist


def calculate_diversity(samples: np.ndarray, method: str = "mean_pairwise") -> float:
    """
    다중 모달리티 다양성 계산

    생성된 여러 궤적 샘플 간의 다양성을 측정합니다.

    Args:
        samples: 생성된 궤적 샘플 [num_samples, batch, pred_steps, 2]
        method: 다양성 계산 방법
            - 'mean_pairwise': 평균 쌍별 거리
            - 'final_distance': 최종 위치 거리
            - 'path_diversity': 경로 다양성

    Returns:
        다양성 점수 (높을수록 다양함)
    """
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()

    samples = np.asarray(samples)

    if samples.ndim != 4:
        raise ValueError(
            f"samples must be 4D [num_samples, batch, pred_steps, 2], got {samples.shape}"
        )

    num_samples, batch_size, pred_steps, _ = samples.shape

    if num_samples < 2:
        return 0.0

    if method == "mean_pairwise":
        # 모든 샘플 쌍 간의 평균 거리
        total_distance = 0.0
        num_pairs = 0

        for b in range(batch_size):
            batch_samples = samples[:, b, :, :]  # [num_samples, pred_steps, 2]

            # 모든 쌍에 대해 거리 계산
            for i in range(num_samples):
                for j in range(i + 1, num_samples):
                    # 각 시점별 거리
                    distances = np.linalg.norm(
                        batch_samples[i] - batch_samples[j], axis=1
                    )
                    # 평균 거리
                    mean_dist = np.mean(distances)
                    total_distance += mean_dist
                    num_pairs += 1

        diversity = total_distance / num_pairs if num_pairs > 0 else 0.0

    elif method == "final_distance":
        # 최종 위치만 사용
        final_positions = samples[:, :, -1, :]  # [num_samples, batch, 2]

        total_distance = 0.0
        num_pairs = 0

        for b in range(batch_size):
            batch_finals = final_positions[:, b, :]  # [num_samples, 2]

            for i in range(num_samples):
                for j in range(i + 1, num_samples):
                    dist = np.linalg.norm(batch_finals[i] - batch_finals[j])
                    total_distance += dist
                    num_pairs += 1

        diversity = total_distance / num_pairs if num_pairs > 0 else 0.0

    elif method == "path_diversity":
        # 경로 다양성 (전체 경로의 차이)
        total_diversity = 0.0
        num_pairs = 0

        for b in range(batch_size):
            batch_samples = samples[:, b, :, :]  # [num_samples, pred_steps, 2]

            for i in range(num_samples):
                for j in range(i + 1, num_samples):
                    # DTW 또는 단순 거리
                    traj_i = batch_samples[i]  # [pred_steps, 2]
                    traj_j = batch_samples[j]  # [pred_steps, 2]

                    # 전체 경로 거리
                    path_dist = np.sum(np.linalg.norm(traj_i - traj_j, axis=1))
                    total_diversity += path_dist
                    num_pairs += 1

        diversity = total_diversity / num_pairs if num_pairs > 0 else 0.0

    else:
        raise ValueError(f"Unknown method: {method}")

    return float(diversity)


def calculate_coverage(
    samples: np.ndarray, ground_truth: np.ndarray, k: int = 20
) -> float:
    """
    Coverage 계산 (실제 궤적 커버리지)

    K개의 샘플 중 실제 궤적과 가장 가까운 샘플의 거리

    Args:
        samples: 생성된 궤적 샘플 [num_samples, batch, pred_steps, 2]
        ground_truth: 실제 궤적 [batch, pred_steps, 2]
        k: 사용할 샘플 수 (K=20)

    Returns:
        Coverage 점수 (낮을수록 좋음)
    """
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()

    samples = np.asarray(samples)
    ground_truth = np.asarray(ground_truth)

    num_samples, batch_size, pred_steps, _ = samples.shape

    # K개 샘플만 사용
    if num_samples > k:
        samples = samples[:k]
        num_samples = k

    total_coverage = 0.0

    for b in range(batch_size):
        gt_traj = ground_truth[b]  # [pred_steps, 2]

        # 모든 샘플과의 ADE 계산
        min_ade = float("inf")

        for s in range(num_samples):
            sample_traj = samples[s, b]  # [pred_steps, 2]

            # ADE 계산
            distances = np.linalg.norm(sample_traj - gt_traj, axis=1)
            ade = np.mean(distances)

            if ade < min_ade:
                min_ade = ade

        total_coverage += min_ade

    coverage = total_coverage / batch_size if batch_size > 0 else 0.0
    return float(coverage)


def calculate_min_ade_fde(
    samples: np.ndarray, ground_truth: np.ndarray
) -> Dict[str, float]:
    """
    최소 ADE/FDE 계산 (K=20 샘플 중 최선)

    Args:
        samples: 생성된 궤적 샘플 [num_samples, batch, pred_steps, 2]
        ground_truth: 실제 궤적 [batch, pred_steps, 2]

    Returns:
        {'min_ade': float, 'min_fde': float}
    """
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()

    samples = np.asarray(samples)
    ground_truth = np.asarray(ground_truth)

    num_samples, batch_size, pred_steps, _ = samples.shape

    total_min_ade = 0.0
    total_min_fde = 0.0

    for b in range(batch_size):
        gt_traj = ground_truth[b]  # [pred_steps, 2]

        min_ade = float("inf")
        min_fde = float("inf")

        for s in range(num_samples):
            sample_traj = samples[s, b]  # [pred_steps, 2]

            # ADE
            distances = np.linalg.norm(sample_traj - gt_traj, axis=1)
            ade = np.mean(distances)

            # FDE
            final_dist = np.linalg.norm(sample_traj[-1] - gt_traj[-1])

            if ade < min_ade:
                min_ade = ade
            if final_dist < min_fde:
                min_fde = final_dist

        total_min_ade += min_ade
        total_min_fde += min_fde

    return {
        "min_ade": float(total_min_ade / batch_size) if batch_size > 0 else 0.0,
        "min_fde": float(total_min_fde / batch_size) if batch_size > 0 else 0.0,
    }


def calculate_multimodal_metrics(
    samples: np.ndarray, ground_truth: np.ndarray, k: int = 20
) -> Dict[str, float]:
    """
    다중 모달리티 평가 지표 통합 계산

    Args:
        samples: 생성된 궤적 샘플 [num_samples, batch, pred_steps, 2]
        ground_truth: 실제 궤적 [batch, pred_steps, 2]
        k: 사용할 샘플 수

    Returns:
        평가 지표 딕셔너리
    """
    metrics = {}

    # Diversity
    metrics["diversity"] = calculate_diversity(samples, method="mean_pairwise")
    metrics["diversity_final"] = calculate_diversity(samples, method="final_distance")
    metrics["diversity_path"] = calculate_diversity(samples, method="path_diversity")

    # Coverage
    metrics["coverage"] = calculate_coverage(samples, ground_truth, k=k)

    # Min ADE/FDE
    min_metrics = calculate_min_ade_fde(samples, ground_truth)
    metrics.update(min_metrics)

    return metrics


class DiffusionEvaluator:
    """Diffusion 모델 평가 클래스"""

    def __init__(self, k: int = 20, vehicle_radius: float = 2.5):
        """
        Args:
            k: 평가에 사용할 샘플 수
            vehicle_radius: 차량 반경 (충돌 검사용)
        """
        self.k = k
        self.vehicle_radius = vehicle_radius

    def evaluate(
        self, samples: np.ndarray, ground_truth: np.ndarray
    ) -> Dict[str, float]:
        """
        Diffusion 모델 평가

        Args:
            samples: 생성된 궤적 샘플 [num_samples, batch, pred_steps, 2]
            ground_truth: 실제 궤적 [batch, pred_steps, 2]

        Returns:
            평가 지표 딕셔너리
        """
        # 다중 모달리티 지표
        metrics = calculate_multimodal_metrics(samples, ground_truth, k=self.k)

        # Collision Rate (샘플 간)
        collision_rate = self._calculate_sample_collision_rate(samples)
        metrics["collision_rate"] = collision_rate

        return metrics

    def _calculate_sample_collision_rate(self, samples: np.ndarray) -> float:
        """
        샘플 간 충돌 비율 계산

        Args:
            samples: 생성된 궤적 샘플 [num_samples, batch, pred_steps, 2]

        Returns:
            충돌 비율
        """
        num_samples, batch_size, pred_steps, _ = samples.shape

        if num_samples < 2 or batch_size < 2:
            return 0.0

        total_collisions = 0
        total_pairs = 0

        for b in range(batch_size):
            batch_samples = samples[:, b, :, :]  # [num_samples, pred_steps, 2]

            # 모든 샘플 쌍에 대해
            for i in range(num_samples):
                for j in range(i + 1, num_samples):
                    traj_i = batch_samples[i]  # [pred_steps, 2]
                    traj_j = batch_samples[j]  # [pred_steps, 2]

                    # 거리 계산
                    distances = np.linalg.norm(traj_i - traj_j, axis=1)

                    # 충돌 여부 (2 * radius 이하)
                    collision_mask = distances <= (2 * self.vehicle_radius)

                    if np.any(collision_mask):
                        total_collisions += 1

                    total_pairs += 1

        return float(total_collisions / total_pairs) if total_pairs > 0 else 0.0
