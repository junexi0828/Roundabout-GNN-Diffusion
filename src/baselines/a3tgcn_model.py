"""
A3TGCN (Attention Temporal Graph Convolutional Network) 모델 구현
PyTorch Geometric Temporal을 활용한 시공간 그래프 신경망
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN2
from torch_geometric.data import Data, Batch
from typing import Optional, Tuple


class InteractionAwarePredictor(nn.Module):
    """
    A3TGCN 기반 상호작용 인식 궤적 예측 모델
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
            num_layers: A3TGCN 레이어 수
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

        # A3TGCN 레이어
        self.tgnn = A3TGCN2(
            in_channels=node_features,
            out_channels=hidden_channels,
            periods=periods,
            batch_size=batch_size
        )

        # 추가 GNN 레이어 (선택사항)
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.gnn_layers.append(
                A3TGCN2(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    periods=periods,
                    batch_size=batch_size
                )
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
            x: 노드 특징 [num_nodes, node_features]
            edge_index: 엣지 인덱스 [2, num_edges]
            edge_weight: 엣지 가중치 (선택사항)

        Returns:
            예측된 궤적 [num_nodes, pred_steps, 2] 또는 [num_nodes, 2 * pred_steps]
        """
        # A3TGCN 통과
        h = self.tgnn(x, edge_index)
        h = F.relu(h)

        # 추가 GNN 레이어
        for gnn_layer in self.gnn_layers:
            h = gnn_layer(h, edge_index)
            h = F.relu(h)

        # Decoder
        pred = self.decoder(h)  # [num_nodes, 2 * pred_steps]

        # Reshape: [num_nodes, 2 * pred_steps] -> [num_nodes, pred_steps, 2]
        pred = pred.view(-1, self.pred_steps, 2)

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
    맵 정보를 통합한 A3TGCN 모델
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

        # A3TGCN (통합된 특징 사용)
        self.tgnn = A3TGCN2(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            periods=periods,
            batch_size=batch_size
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
            agent_x: 에이전트 노드 특징 [num_agents, agent_features]
            map_x: 맵 노드 특징 [num_map_nodes, map_features] (선택사항)
            edge_index: 엣지 인덱스 [2, num_edges]
        """
        # 특징 인코딩
        agent_h = self.agent_encoder(agent_x)

        if map_x is not None:
            map_h = self.map_encoder(map_x)
            # Agent와 Map 특징 결합
            x = torch.cat([agent_h, map_h], dim=0)
        else:
            x = agent_h

        # A3TGCN
        h = self.tgnn(x, edge_index)
        h = F.relu(h)

        # Agent 노드만 선택 (맵 노드는 예측 대상이 아님)
        if map_x is not None:
            h = h[:agent_h.size(0)]

        # Decoder
        pred = self.decoder(h)
        pred = pred.view(-1, self.pred_steps, 2)

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

