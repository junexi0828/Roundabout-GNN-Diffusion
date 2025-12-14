"""
Custom A3TGCN Implementation
torch-geometric-temporal 대체 (Mac MPS 호환성 개선)

GAT (Spatial) + LSTM (Temporal) 조합으로 동일한 기능 제공
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from typing import Optional, Tuple

# 커스텀 ST Encoder 사용
from src.models.st_encoder import SpatioTemporalEncoder


class InteractionAwarePredictor(nn.Module):
    """
    Custom Spatio-Temporal GNN 기반 궤적 예측 모델

    torch-geometric-temporal의 A3TGCN을 GAT+LSTM으로 대체
    """

    def __init__(
        self,
        node_features: int = 9,  # x, y, vx, vy, ax, ay, psi, width, length
        hidden_channels: int = 32,
        num_layers: int = 2,
        periods: int = 30,  # 관측 윈도우 길이
        pred_steps: int = 50,  # 예측 스텝 수 (5초, 10Hz 기준)
        batch_size: int = 1,
        dropout: float = 0.1
    ):
        """
        Args:
            node_features: 노드 특징 차원
            hidden_channels: 은닉층 채널 수
            num_layers: ST Encoder 레이어 수
            periods: 시간 윈도우 길이 (look-back)
            pred_steps: 예측할 미래 스텝 수
            batch_size: 배치 크기
            dropout: 드롭아웃 비율
        """
        super(InteractionAwarePredictor, self).__init__()

        self.node_features = node_features
        self.hidden_channels = hidden_channels
        self.pred_steps = pred_steps
        self.periods = periods

        # Custom Spatio-Temporal Encoder (GAT + LSTM)
        self.st_encoder = SpatioTemporalEncoder(
            input_dim=node_features,
            hidden_dim=hidden_channels,
            num_heads=4,
            num_layers=num_layers,
            dropout=dropout
        )

        # Decoder: Hidden state -> Trajectory (x, y)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 2 * pred_steps)  # (x, y) * pred_steps
        )

        # 어텐션 가중치 저장 (시각화용)
        self.attention_weights = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: 노드 특징 [Batch, Time, Nodes, Features] 또는 [Nodes, Features]
            edge_index: 엣지 인덱스 [2, num_edges]
            edge_weight: 엣지 가중치 (선택사항)

        Returns:
            예측된 궤적 [Batch, Nodes, pred_steps, 2]
        """
        # 입력 형태 확인 및 변환
        if x.dim() == 2:
            # [Nodes, Features] -> [1, 1, Nodes, Features]
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            # [Batch, Nodes, Features] -> [Batch, 1, Nodes, Features]
            x = x.unsqueeze(1)

        # ST Encoder 통과
        # x: [Batch, Time, Nodes, Features]
        # output: [Batch, Nodes, Hidden]
        h = self.st_encoder(x, edge_index, edge_weight)

        # Decoder
        pred = self.decoder(h)  # [Batch, Nodes, 2 * pred_steps]

        # Reshape: [Batch, Nodes, 2 * pred_steps] -> [Batch, Nodes, pred_steps, 2]
        batch_size, num_nodes, _ = pred.size()
        pred = pred.view(batch_size, num_nodes, self.pred_steps, 2)

        return pred

    def predict_trajectory(
        self,
        data: Data,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        단일 그래프에 대한 궤적 예측

        Args:
            data: PyTorch Geometric Data 객체
            return_attention: 어텐션 가중치 반환 여부

        Returns:
            예측 궤적, (선택적) 어텐션 가중치
        """
        self.eval()
        with torch.no_grad():
            pred = self.forward(data.x, data.edge_index, data.edge_weight)

        attention = None
        if return_attention:
            # A3TGCN의 어텐션 가중치 추출 (구현에 따라 다를 수 있음)
            attention = self.attention_weights

        return pred, attention


class A3TGCNWithMap(nn.Module):
    """
    맵 정보를 통합한 Custom ST-GNN 모델
    맵 노드를 추가하여 회전교차로의 기하학적 제약을 반영
    """

    def __init__(
        self,
        agent_features: int = 9,
        map_features: int = 6,  # x_mid, y_mid, curvature, speed_limit, is_entry, is_exit
        hidden_channels: int = 32,
        periods: int = 30,
        pred_steps: int = 50,
        batch_size: int = 1
    ):
        super(A3TGCNWithMap, self).__init__()

        # Agent 특징 인코더
        self.agent_encoder = nn.Linear(agent_features, hidden_channels)

        # Map 특징 인코더
        self.map_encoder = nn.Linear(map_features, hidden_channels)

        # Custom ST Encoder
        self.st_encoder = SpatioTemporalEncoder(
            input_dim=hidden_channels,
            hidden_dim=hidden_channels,
            num_heads=4,
            num_layers=2,
            dropout=0.1
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.ReLU(),
            nn.Linear(hidden_channels * 2, 2 * pred_steps)
        )

        self.pred_steps = pred_steps

    def forward(
        self,
        agent_x: torch.Tensor,
        map_x: Optional[torch.Tensor],
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            agent_x: 에이전트 노드 특징 [Batch, Time, Agents, Features]
            map_x: 맵 노드 특징 [Map_nodes, Features] (선택사항)
            edge_index: 엣지 인덱스 [2, num_edges]
        """
        # 입력 형태 확인
        if agent_x.dim() == 2:
            agent_x = agent_x.unsqueeze(0).unsqueeze(0)
        elif agent_x.dim() == 3:
            agent_x = agent_x.unsqueeze(1)

        batch_size, time_steps, num_agents, _ = agent_x.size()

        # 특징 인코딩
        agent_h = self.agent_encoder(agent_x)  # [Batch, Time, Agents, Hidden]

        if map_x is not None:
            map_h = self.map_encoder(map_x)  # [Map_nodes, Hidden]
            # 맵 특징을 시간/배치 차원으로 확장
            map_h = map_h.unsqueeze(0).unsqueeze(0).expand(batch_size, time_steps, -1, -1)
            # Agent와 Map 특징 결합
            x = torch.cat([agent_h, map_h], dim=2)  # [Batch, Time, Agents+Map, Hidden]
        else:
            x = agent_h

        # ST Encoder
        h = self.st_encoder(x, edge_index)  # [Batch, Agents+Map, Hidden]
        h = F.relu(h)

        # Agent 노드만 선택 (맵 노드는 예측 대상이 아님)
        if map_x is not None:
            h = h[:, :num_agents, :]  # [Batch, Agents, Hidden]

        # Decoder
        pred = self.decoder(h)  # [Batch, Agents, 2 * pred_steps]
        pred = pred.view(batch_size, -1, self.pred_steps, 2)

        return pred


def create_a3tgcn_model(
    node_features: int = 9,
    hidden_channels: int = 32,
    pred_steps: int = 50,
    use_map: bool = False
) -> nn.Module:
    """
    A3TGCN 모델 생성 헬퍼 함수

    Args:
        node_features: 노드 특징 차원
        hidden_channels: 은닉층 채널 수
        pred_steps: 예측 스텝 수
        use_map: 맵 정보 사용 여부

    Returns:
        모델 인스턴스
    """
    if use_map:
        return A3TGCNWithMap(
            agent_features=node_features,
            hidden_channels=hidden_channels,
            pred_steps=pred_steps
        )
    else:
        return InteractionAwarePredictor(
            node_features=node_features,
            hidden_channels=hidden_channels,
            pred_steps=pred_steps
        )


def main():
    """테스트용 메인 함수"""
    # 모델 생성
    model = create_a3tgcn_model(
        node_features=9,
        hidden_channels=32,
        pred_steps=50,
        use_map=False
    )

    print(f"모델 구조:")
    print(model)

    # 더미 데이터로 테스트
    num_nodes = 5
    num_edges = 10

    x = torch.randn(num_nodes, 9)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))

    # Forward pass
    pred = model(x, edge_index)

    print(f"\n입력:")
    print(f"  노드 특징: {x.shape}")
    print(f"  엣지 인덱스: {edge_index.shape}")

    print(f"\n출력:")
    print(f"  예측 궤적: {pred.shape}")  # [num_nodes, pred_steps, 2]

    # 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n파라미터:")
    print(f"  총 파라미터 수: {total_params:,}")
    print(f"  학습 가능 파라미터: {trainable_params:,}")


if __name__ == "__main__":
    main()

