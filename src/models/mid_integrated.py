"""
MID 통합 모델
HeteroGAT + 씬 그래프 + Plan B 안전 검증 통합
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict
from torch_geometric.data import Data, HeteroData

from .mid_model import HybridGNNMID, create_mid_model
from .mid_with_safety import SafetyGuidedMID, create_safety_guided_mid
from ..integration.mid_scene_graph import MIDSceneGraphIntegrator, create_mid_with_scene_graph


class FullyIntegratedMID(nn.Module):
    """
    완전 통합 MID 모델
    - HeteroGAT (이기종 에이전트)
    - 씬 그래프 (상호작용 모델링)
    - Plan B (안전 검증)
    """

    def __init__(
        self,
        obs_steps: int = 30,
        pred_steps: int = 50,
        hidden_dim: int = 128,
        num_diffusion_steps: int = 100,
        node_features: int = 9,
        node_types: Optional[List[str]] = None,
        edge_types: Optional[List[Tuple[str, str, str]]] = None,
        use_safety: bool = True,
        ttc_threshold: float = 1.5
    ):
        super(FullyIntegratedMID, self).__init__()

        # MID 모델 생성 (HeteroGAT 포함)
        self.mid_model = create_mid_model(
            obs_steps=obs_steps,
            pred_steps=pred_steps,
            hidden_dim=hidden_dim,
            num_diffusion_steps=num_diffusion_steps,
            use_gnn=True,
            node_features=node_features,
            use_hetero_gnn=True,
            node_types=node_types or ['car', 'pedestrian', 'biker', 'skater', 'cart', 'bus'],
            edge_types=edge_types or [
                ('car', 'spatial', 'car'),
                ('car', 'spatial', 'pedestrian'),
                ('pedestrian', 'spatial', 'biker'),
                ('biker', 'spatial', 'car'),
            ]
        )

        # 안전 검증 레이어
        self.use_safety = use_safety
        if use_safety:
            self.safety_guided = create_safety_guided_mid(
                self.mid_model,
                ttc_threshold=ttc_threshold
            )
        else:
            self.safety_guided = None

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

    def sample(
        self,
        graph_data: Optional[Data] = None,
        hetero_data: Optional[HeteroData] = None,
        obs_trajectory: Optional[torch.Tensor] = None,
        current_states: Optional[torch.Tensor] = None,
        num_samples: int = 20,
        ddim_steps: int = 2,
        use_safety_filter: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        샘플링 (안전 검증 포함)

        Args:
            graph_data: 그래프 데이터
            hetero_data: 이기종 그래프 데이터
            obs_trajectory: 관측 궤적
            current_states: 현재 상태 (안전 검증용)
            num_samples: 샘플 수
            ddim_steps: DDIM 스텝 수
            use_safety_filter: 안전 필터링 사용 여부

        Returns:
            {
                'samples': 모든 샘플,
                'safe_samples': 안전한 샘플,
                'safety_scores': 안전 점수
            }
        """
        if self.use_safety and use_safety_filter and current_states is not None:
            # 안전 가이드 샘플링
            if isinstance(current_states, torch.Tensor):
                current_states_np = current_states.detach().cpu().numpy()
            else:
                current_states_np = current_states

            return self.safety_guided.sample_with_safety(
                graph_data=graph_data,
                hetero_data=hetero_data,
                obs_trajectory=obs_trajectory,
                current_states=current_states_np,
                num_samples=num_samples,
                ddim_steps=ddim_steps,
                filter_unsafe=True
            )
        else:
            # 일반 샘플링
            samples = self.mid_model.sample(
                graph_data=graph_data,
                hetero_data=hetero_data,
                obs_trajectory=obs_trajectory,
                num_samples=num_samples,
                ddim_steps=ddim_steps
            )
            return {
                'samples': samples,
                'safe_samples': samples,
                'safety_scores': torch.ones(samples.shape[0], samples.shape[1])
            }


def create_fully_integrated_mid(
    obs_steps: int = 30,
    pred_steps: int = 50,
    hidden_dim: int = 128,
    num_diffusion_steps: int = 100,
    node_features: int = 9,
    node_types: Optional[List[str]] = None,
    edge_types: Optional[List[Tuple[str, str, str]]] = None,
    use_safety: bool = True
) -> FullyIntegratedMID:
    """완전 통합 MID 모델 생성"""
    return FullyIntegratedMID(
        obs_steps=obs_steps,
        pred_steps=pred_steps,
        hidden_dim=hidden_dim,
        num_diffusion_steps=num_diffusion_steps,
        node_features=node_features,
        node_types=node_types,
        edge_types=edge_types,
        use_safety=use_safety
    )

