"""
Custom Spatio-Temporal Encoder
GAT (Spatial) + LSTM (Temporal) 조합으로 torch-geometric-temporal 대체

이 구현은 A3TGCN과 동일한 기능을 PyTorch 기본 기능만으로 제공합니다.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from typing import Optional


class SpatioTemporalEncoder(nn.Module):
    """
    Spatio-Temporal Encoder using GAT + LSTM

    torch-geometric-temporal의 A3TGCN을 대체하는 커스텀 구현
    - Spatial: GATConv (주변 에이전트 관계 분석)
    - Temporal: LSTM (과거 궤적 흐름 분석)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: 입력 특징 차원
            hidden_dim: 은닉 차원
            num_heads: GAT attention heads 수
            num_layers: LSTM 레이어 수
            dropout: Dropout 비율
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # 1. Spatial Encoder (GAT)
        self.spatial_gnn = GATConv(
            in_channels=input_dim,
            out_channels=hidden_dim // num_heads,  # 멀티헤드 고려
            heads=num_heads,
            dropout=dropout,
            concat=True  # 헤드 concat
        )

        # 2. Temporal Encoder (LSTM)
        self.temporal_rnn = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )

        # 3. Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # 4. Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Node features [Batch, Time, Nodes, Features]
            edge_index: Edge indices [2, Edges]
            edge_attr: Edge attributes (optional)

        Returns:
            Context vector [Batch, Nodes, Hidden]
        """
        batch_size, time_steps, num_nodes, _ = x.size()

        # 1. Spatial Feature Extraction (매 시간마다 GAT 적용)
        spatial_features = []

        for t in range(time_steps):
            # t 시점의 데이터 [Batch * Nodes, Features]
            x_t = x[:, t, :, :].reshape(batch_size * num_nodes, -1)

            # GAT 통과 [Batch * Nodes, Hidden]
            h_t = self.spatial_gnn(x_t, edge_index, edge_attr)
            h_t = torch.relu(h_t)
            h_t = self.dropout(h_t)

            # [Batch, Nodes, Hidden]로 reshape
            h_t = h_t.reshape(batch_size, num_nodes, -1)
            spatial_features.append(h_t)

        # [Batch, Time, Nodes, Hidden]
        spatial_seq = torch.stack(spatial_features, dim=1)

        # 2. Temporal Feature Extraction (노드별로 LSTM 적용)
        temporal_features = []

        for n in range(num_nodes):
            # n번째 노드의 시간 시퀀스 [Batch, Time, Hidden]
            node_seq = spatial_seq[:, :, n, :]

            # LSTM 통과
            # output: [Batch, Time, Hidden]
            # h_n: [Num_layers, Batch, Hidden]
            _, (h_n, _) = self.temporal_rnn(node_seq)

            # 마지막 레이어의 hidden state [Batch, Hidden]
            context = h_n[-1]
            temporal_features.append(context)

        # [Batch, Nodes, Hidden]
        output = torch.stack(temporal_features, dim=1)

        # 3. Layer Normalization
        output = self.layer_norm(output)

        return output

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """
        GAT attention weights 반환 (시각화용)
        """
        # GATConv는 forward 중에만 attention 계산
        # 별도로 저장하려면 forward hook 필요
        return None


class SimplifiedSTEncoder(nn.Module):
    """
    간소화된 Spatio-Temporal Encoder

    노드별 LSTM 대신 전체 시퀀스에 대해 한 번에 처리 (더 빠름)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        # Spatial
        self.spatial_gnn = GATConv(
            input_dim,
            hidden_dim // num_heads,
            heads=num_heads,
            dropout=dropout,
            concat=True
        )

        # Temporal
        self.temporal_rnn = nn.LSTM(
            hidden_dim,
            hidden_dim,
            batch_first=True
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: [Batch, Time, Nodes, Features]
            edge_index: [2, Edges]

        Returns:
            [Batch, Nodes, Hidden]
        """
        batch_size, time_steps, num_nodes, _ = x.size()

        # 1. Spatial (모든 시간 한번에 처리)
        # [Batch * Time * Nodes, Features]
        x_flat = x.reshape(-1, x.size(-1))

        # GAT
        h = self.spatial_gnn(x_flat, edge_index)
        h = torch.relu(h)
        h = self.dropout(h)

        # [Batch, Time, Nodes, Hidden]
        h = h.reshape(batch_size, time_steps, num_nodes, -1)

        # 2. Temporal (평균 풀링 후 LSTM)
        # [Batch, Time, Hidden] - 노드 평균
        h_pooled = h.mean(dim=2)

        # LSTM
        _, (context, _) = self.temporal_rnn(h_pooled)

        # [Batch, Hidden]
        context = context.squeeze(0)

        # 모든 노드에 브로드캐스트
        # [Batch, Nodes, Hidden]
        output = context.unsqueeze(1).expand(-1, num_nodes, -1)

        output = self.layer_norm(output)

        return output


def create_st_encoder(
    input_dim: int,
    hidden_dim: int,
    simplified: bool = False,
    **kwargs
) -> nn.Module:
    """
    Spatio-Temporal Encoder 생성 헬퍼 함수

    Args:
        input_dim: 입력 차원
        hidden_dim: 은닉 차원
        simplified: 간소화 버전 사용 여부
        **kwargs: 추가 파라미터

    Returns:
        STEncoder 모델
    """
    if simplified:
        return SimplifiedSTEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            **kwargs
        )
    else:
        return SpatioTemporalEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            **kwargs
        )
