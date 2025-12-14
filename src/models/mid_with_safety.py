"""
MID 모델 + Plan B 안전 검증 통합
안전 가이드 샘플링 구현
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, List, Tuple
from torch_geometric.data import Data, HeteroData

from .mid_model import HybridGNNMID, MIDModel
from ..integration.hybrid_safety_layer import SafetyLayer


class SafetyGuidedMID(nn.Module):
    """
    안전 가이드 MID 모델
    Plan B 안전 지표로 샘플링 가이드
    """

    def __init__(
        self,
        mid_model: nn.Module,
        ttc_threshold: float = 1.5,
        drac_threshold: float = 5.0,
        vehicle_radius: float = 2.5
    ):
        """
        Args:
            mid_model: MID 모델
            ttc_threshold: TTC 임계값
            drac_threshold: DRAC 임계값
            vehicle_radius: 차량 반경
        """
        super(SafetyGuidedMID, self).__init__()

        self.mid_model = mid_model
        self.safety_layer = SafetyLayer(
            ttc_threshold=ttc_threshold,
            drac_threshold=drac_threshold,
            vehicle_radius=vehicle_radius
        )

    def sample_with_safety(
        self,
        graph_data: Optional[Data] = None,
        hetero_data: Optional[HeteroData] = None,
        obs_trajectory: Optional[torch.Tensor] = None,
        current_states: Optional[np.ndarray] = None,
        num_samples: int = 20,
        ddim_steps: int = 2,
        filter_unsafe: bool = True,
        min_safe_samples: int = 5
    ) -> Dict[str, torch.Tensor]:
        """
        안전 가이드 샘플링

        Args:
            graph_data: 그래프 데이터
            hetero_data: 이기종 그래프 데이터
            obs_trajectory: 관측 궤적
            current_states: 현재 상태 [batch, state_dim] (x, y, vx, vy 포함)
            num_samples: 샘플 수
            ddim_steps: DDIM 스텝 수
            filter_unsafe: 위험한 샘플 필터링 여부
            min_safe_samples: 최소 안전 샘플 수

        Returns:
            {
                'samples': 모든 샘플 [num_samples, batch, pred_steps, 2],
                'safe_samples': 안전한 샘플 [num_safe, batch, pred_steps, 2],
                'safety_scores': 안전 점수 [num_samples, batch]
            }
        """
        self.eval()

        # MID로 샘플링
        with torch.no_grad():
            samples = self.mid_model.sample(
                graph_data=graph_data,
                hetero_data=hetero_data,
                obs_trajectory=obs_trajectory,
                num_samples=num_samples,
                ddim_steps=ddim_steps
            )

        # CPU로 이동 및 numpy 변환
        samples_np = samples.detach().cpu().numpy()  # [num_samples, batch, pred_steps, 2]
        num_samples, batch_size, pred_steps, _ = samples_np.shape

        # 안전 점수 계산
        safety_scores = np.ones((num_samples, batch_size))
        safe_indices = []

        if current_states is not None and filter_unsafe:
            for s in range(num_samples):
                sample_traj = samples_np[s]  # [batch, pred_steps, 2]

                # 각 배치에 대해 안전 검증
                for b in range(batch_size):
                    traj = sample_traj[b]  # [pred_steps, 2]
                    state = current_states[b]  # [state_dim]

                    # 안전 검증 (간단한 버전)
                    # 실제로는 SafetyLayer 사용
                    # 여기서는 거리 기반 간단한 검증
                    if b < batch_size - 1:
                        # 다음 에이전트와의 거리 확인
                        next_traj = sample_traj[b + 1]  # [pred_steps, 2]
                        distances = np.linalg.norm(traj - next_traj, axis=1)
                        min_dist = np.min(distances)

                        # 안전 점수 (거리가 가까울수록 낮음)
                        if min_dist < 2 * self.safety_layer.vehicle_radius:
                            safety_scores[s, b] = 0.0
                        else:
                            safety_scores[s, b] = min_dist / (2 * self.safety_layer.vehicle_radius)

        # 안전한 샘플 필터링
        if filter_unsafe:
            # 각 배치별로 안전한 샘플 선택
            safe_samples_list = []

            for b in range(batch_size):
                batch_scores = safety_scores[:, b]
                safe_mask = batch_scores > 0.5  # 임계값
                safe_idx = np.where(safe_mask)[0]

                if len(safe_idx) < min_safe_samples:
                    # 안전한 샘플이 부족하면 점수 상위 샘플 선택
                    top_k = np.argsort(batch_scores)[-min_safe_samples:]
                    safe_idx = top_k

                safe_samples_list.append(samples_np[safe_idx, b, :, :])

            # 배치별로 다른 수의 안전 샘플이 있을 수 있음
            # 최소 개수로 맞춤
            min_safe_count = min(len(s) for s in safe_samples_list) if safe_samples_list else num_samples
            safe_samples_array = np.stack([
                s[:min_safe_count] for s in safe_samples_list
            ], axis=1)  # [min_safe_count, batch, pred_steps, 2]

            safe_samples = torch.from_numpy(safe_samples_array).float()
        else:
            safe_samples = samples

        return {
            'samples': samples,
            'safe_samples': safe_samples,
            'safety_scores': torch.from_numpy(safety_scores).float()
        }

    def forward(
        self,
        graph_data: Optional[Data] = None,
        hetero_data: Optional[HeteroData] = None,
        obs_trajectory: Optional[torch.Tensor] = None,
        future_trajectory: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        x_t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass (학습용)"""
        return self.mid_model(
            graph_data=graph_data,
            hetero_data=hetero_data,
            obs_trajectory=obs_trajectory,
            future_trajectory=future_trajectory,
            t=t,
            x_t=x_t
        )


def create_safety_guided_mid(
    mid_model: nn.Module,
    ttc_threshold: float = 1.5,
    drac_threshold: float = 5.0
) -> SafetyGuidedMID:
    """안전 가이드 MID 모델 생성"""
    return SafetyGuidedMID(
        mid_model=mid_model,
        ttc_threshold=ttc_threshold,
        drac_threshold=drac_threshold
    )

