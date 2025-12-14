"""
이기종 그래프 신경망 모델
SDD Death Circle의 이기종 에이전트(차량, 보행자, 자전거 등) 처리
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv
from torch_geometric.data import HeteroData
from typing import Dict, List, Optional, Tuple


class HeteroGAT(nn.Module):
    """이기종 Graph Attention Network"""

    def __init__(
        self,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        in_channels: int = 9,
        hidden_channels: int = 64,
        out_channels: int = 64,
        num_heads: int = 4,
        num_layers: int = 2
    ):
        """
        Args:
            node_types: 노드 타입 리스트 ['car', 'pedestrian', 'biker', ...]
            edge_types: 엣지 타입 리스트 [('car', 'yield', 'pedestrian'), ...]
            in_channels: 입력 특징 차원
            hidden_channels: 은닉층 차원
            out_channels: 출력 차원
            num_heads: 어텐션 헤드 수
            num_layers: 레이어 수
        """
        super(HeteroGAT, self).__init__()

        self.node_types = node_types
        self.edge_types = edge_types
        self.num_layers = num_layers

        # 노드 타입별 입력 변환
        self.node_encoders = nn.ModuleDict({
            node_type: nn.Linear(in_channels, hidden_channels)
            for node_type in node_types
        })

        # 이기종 컨볼루션 레이어
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for src_type, relation, dst_type in edge_types:
                edge_key = (src_type, relation, dst_type)
                conv_dict[edge_key] = GATConv(
                    (-1, -1),  # 동적 입력 크기
                    hidden_channels,
                    heads=num_heads,
                    concat=False,
                    add_self_loops=False
                )
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))

        # 출력 변환
        self.node_decoders = nn.ModuleDict({
            node_type: nn.Linear(hidden_channels, out_channels)
            for node_type in node_types
        })

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            x_dict: 노드 타입별 특징 딕셔너리 {node_type: [num_nodes, features]}
            edge_index_dict: 엣지 타입별 인덱스 딕셔너리 {(src, relation, dst): [2, num_edges]}

        Returns:
            노드 타입별 임베딩 딕셔너리
        """
        # 입력 인코딩
        h_dict = {}
        for node_type in self.node_types:
            if node_type in x_dict:
                h_dict[node_type] = self.node_encoders[node_type](x_dict[node_type])
                h_dict[node_type] = F.relu(h_dict[node_type])

        # 이기종 컨볼루션
        for conv in self.convs:
            h_dict = conv(h_dict, edge_index_dict)
            h_dict = {k: F.relu(v) for k, v in h_dict.items()}

        # 출력 디코딩
        out_dict = {}
        for node_type in self.node_types:
            if node_type in h_dict:
                out_dict[node_type] = self.node_decoders[node_type](h_dict[node_type])

        return out_dict


class HybridGNNMID(nn.Module):
    """
    HeteroGAT + MID 결합 모델

    이질적 그래프 구조와 Diffusion 기반 궤적 예측 결합
    """

    def __init__(
        self,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        node_features: int = 9,
        hidden_channels: int = 128,
        num_heads: int = 4,
        obs_steps: int = 30,
        pred_steps: int = 50,
        num_diffusion_steps: int = 100,
        dropout: float = 0.1
    ):
        super().__init__()

        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_channels = hidden_channels
        self.obs_steps = obs_steps
        self.pred_steps = pred_steps
        self.num_diffusion_steps = num_diffusion_steps

        # HeteroGAT 인코더
        self.spatial_encoder = HeteroGAT(
            node_types=node_types,
            edge_types=edge_types,
            in_channels=node_features,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_heads=num_heads
        )

        # 간단한 디코더 (실제로는 MID 모델 사용)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, 2 * pred_steps)
        )

    def forward(self, x_dict, edge_index_dict):
        """Forward pass"""
        # HeteroGAT 인코딩
        h_dict = self.spatial_encoder(x_dict, edge_index_dict)

        # 모든 노드 타입 결합
        all_h = torch.cat(list(h_dict.values()), dim=0)

        # 디코딩
        pred = self.decoder(all_h)
        pred = pred.view(-1, self.pred_steps, 2)

        return pred


def create_heterogeneous_model(
    node_types: List[str],
    edge_types: List[Tuple[str, str, str]],
    node_features: int = 9,
    hidden_channels: int = 64,
    pred_steps: int = 50
) -> HybridGNNMID:
    """
    이기종 모델 생성 헬퍼 함수

    Args:
        node_types: 노드 타입 리스트
        edge_types: 엣지 타입 리스트
        node_features: 노드 특징 차원
        hidden_channels: 은닉층 차원
        pred_steps: 예측 스텝 수

    Returns:
        HybridGNNMID 모델 인스턴스
    """
    return HybridGNNMID(
        node_types=node_types,
        edge_types=edge_types,
        node_features=node_features,
        hidden_channels=hidden_channels,
        pred_steps=pred_steps
    )


if __name__ == "__main__":
    # 테스트 코드
    node_types = ['car', 'pedestrian', 'biker']
    edge_types = [
        ('car', 'yield', 'pedestrian'),
        ('biker', 'overtake', 'car'),
    ]

    model = create_heterogeneous_model(
        node_types=node_types,
        edge_types=edge_types
    )

    print(f"모델 생성 완료: {model.__class__.__name__}")
