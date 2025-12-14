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


class STHGNN(nn.Module):
    """
    Spatio-Temporal Heterogeneous Graph Neural Network
    HeteroGAT + A3TGCN 결합 모델
    """

    def __init__(
        self,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        node_features: int = 9,
        hidden_channels: int = 64,
        pred_steps: int = 50,
        periods: int = 30
    ):
        """
        Args:
            node_types: 노드 타입 리스트
            edge_types: 엣지 타입 리스트
            node_features: 노드 특징 차원
            hidden_channels: 은닉층 차원
            pred_steps: 예측 스텝 수
            periods: 시간 윈도우 길이
        """
        super(STHGNN, self).__init__()

        # 공간 인코더 (HeteroGAT)
        self.spatial_encoder = HeteroGAT(
            node_types=node_types,
            edge_types=edge_types,
            in_channels=node_features,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels
        )

        # 시간 인코더 (A3TGCN 대신 간단한 GRU 사용)
        # 실제로는 PyTorch Geometric Temporal의 A3TGCN 사용 가능
        self.temporal_encoder = nn.GRU(
            input_size=hidden_channels,
            hidden_size=hidden_channels,
            num_layers=2,
            batch_first=True
        )

        # 디코더
        self.decoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels, 2 * pred_steps)  # (x, y) * pred_steps
        )

        self.pred_steps = pred_steps

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict,
        temporal_sequence: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            x_dict: 노드 타입별 특징
            edge_index_dict: 엣지 인덱스
            temporal_sequence: 시간적 시퀀스 (선택사항)

        Returns:
            노드 타입별 예측 궤적
        """
        # 공간 인코딩
        spatial_embeddings = self.spatial_encoder(x_dict, edge_index_dict)

        # 시간 인코딩 (간단한 구현)
        # 실제로는 A3TGCN 사용
        if temporal_sequence is not None:
            # 시퀀스 형태로 변환
            h, _ = self.temporal_encoder(temporal_sequence)
            temporal_embedding = h[:, -1, :]  # 마지막 시점
        else:
            # 공간 임베딩을 시간 임베딩으로 사용
            # 모든 노드 타입의 임베딩을 결합
            all_embeddings = torch.cat(list(spatial_embeddings.values()), dim=0)
            temporal_embedding = all_embeddings

        # 디코딩
        pred = self.decoder(temporal_embedding)
        pred = pred.view(-1, self.pred_steps, 2)

        # 노드 타입별로 분할 (간단한 구현)
        # 실제로는 더 정교한 분할 로직 필요
        predictions = {}
        start_idx = 0
        for node_type in spatial_embeddings.keys():
            num_nodes = spatial_embeddings[node_type].size(0)
            predictions[node_type] = pred[start_idx:start_idx + num_nodes]
            start_idx += num_nodes

        return predictions


def create_heterogeneous_model(
    node_types: List[str],
    edge_types: List[Tuple[str, str, str]],
    node_features: int = 9,
    hidden_channels: int = 64,
    pred_steps: int = 50
) -> STHGNN:
    """
    이기종 모델 생성 헬퍼 함수

    Args:
        node_types: 노드 타입 리스트
        edge_types: 엣지 타입 리스트
        node_features: 노드 특징 차원
        hidden_channels: 은닉층 차원
        pred_steps: 예측 스텝 수

    Returns:
        STHGNN 모델 인스턴스
    """
    return STHGNN(
        node_types=node_types,
        edge_types=edge_types,
        node_features=node_features,
        hidden_channels=hidden_channels,
        pred_steps=pred_steps
    )


def main():
    """테스트용 메인 함수"""
    # SDD Death Circle의 이기종 에이전트 타입
    node_types = ['car', 'pedestrian', 'biker', 'skater', 'cart', 'bus']

    # 관계 타입
    edge_types = [
        ('car', 'yield', 'pedestrian'),
        ('biker', 'overtake', 'car'),
        ('pedestrian', 'avoid', 'pedestrian'),
        ('biker', 'filter', 'car'),
        ('car', 'follow', 'car')
    ]

    # 모델 생성
    model = create_heterogeneous_model(
        node_types=node_types,
        edge_types=edge_types,
        node_features=9,
        hidden_channels=64,
        pred_steps=50
    )

    print(f"모델 구조:")
    print(model)

    # 더미 데이터로 테스트
    x_dict = {
        'car': torch.randn(3, 9),
        'pedestrian': torch.randn(2, 9),
        'biker': torch.randn(2, 9)
    }

    edge_index_dict = {
        ('car', 'yield', 'pedestrian'): torch.randint(0, 3, (2, 4)),
        ('biker', 'overtake', 'car'): torch.randint(0, 2, (2, 3))
    }

    # Forward pass
    predictions = model(x_dict, edge_index_dict)

    print(f"\n예측 결과:")
    for node_type, pred in predictions.items():
        print(f"  {node_type}: {pred.shape}")


if __name__ == "__main__":
    main()

