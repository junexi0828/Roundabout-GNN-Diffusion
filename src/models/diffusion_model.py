"""
Diffusion 기반 궤적 예측 모델
MID (Motion Indeterminacy Diffusion) 아키텍처 기반
이기종 에이전트 및 씬 그래프 조건 통합
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict
import math

from .a3tgcn_model import InteractionAwarePredictor


class SinusoidalPositionalEmbedding(nn.Module):
    """사인 코사인 위치 임베딩 (Diffusion timestep용)"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time: 타임스텝 [batch_size]

        Returns:
            임베딩 [batch_size, dim]
        """
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class DiffusionTrajectoryPredictor(nn.Module):
    """
    Diffusion 기반 궤적 예측 모델
    MID 아키텍처를 이기종 환경에 맞게 확장
    """

    def __init__(
        self,
        node_features: int = 9,
        hidden_dim: int = 128,
        num_diffusion_steps: int = 100,
        pred_steps: int = 50,
        use_scene_graph: bool = True,
    ):
        """
        Args:
            node_features: 노드 특징 차원
            hidden_dim: 은닉층 차원
            num_diffusion_steps: Diffusion 스텝 수
            pred_steps: 예측 스텝 수
            use_scene_graph: 씬 그래프 사용 여부
        """
        super(DiffusionTrajectoryPredictor, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_diffusion_steps = num_diffusion_steps
        self.pred_steps = pred_steps
        self.use_scene_graph = use_scene_graph

        # 타임스텝 임베딩
        self.time_embedding = SinusoidalPositionalEmbedding(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 관측 인코더 (GNN 또는 LSTM)
        if use_scene_graph:
            # GNN 기반 인코더 (씬 그래프 사용)
            from torch_geometric.nn import GATConv

            self.graph_encoder = nn.ModuleList(
                [
                    GATConv(node_features, hidden_dim, heads=4, concat=False),
                    GATConv(hidden_dim, hidden_dim, heads=4, concat=False),
                ]
            )
        else:
            # LSTM 기반 인코더
            self.lstm_encoder = nn.LSTM(
                node_features, hidden_dim, num_layers=2, batch_first=True
            )

        # 조건 임베딩 (관측 정보)
        self.condition_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # Diffusion Denoiser (Transformer 기반)
        self.denoiser = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=4,
        )

        # 출력 레이어
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, pred_steps * 2),  # (x, y) * pred_steps
        )

        # Beta 스케줄 (선형 또는 코사인)
        self.register_buffer("betas", self._linear_beta_schedule(num_diffusion_steps))
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))
        self.register_buffer(
            "alphas_cumprod_prev", F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        )

        # DDIM용 (빠른 샘플링)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - self.alphas_cumprod)
        )

    def _linear_beta_schedule(
        self, num_steps: int, beta_start: float = 0.0001, beta_end: float = 0.02
    ) -> torch.Tensor:
        """선형 Beta 스케줄"""
        return torch.linspace(beta_start, beta_end, num_steps)

    def _cosine_beta_schedule(self, num_steps: int, s: float = 0.008) -> torch.Tensor:
        """코사인 Beta 스케줄 (선택사항)"""
        steps = num_steps + 1
        x = torch.linspace(0, num_steps, steps)
        alphas_cumprod = torch.cos(((x / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def encode_observation(
        self,
        obs_data: torch.Tensor,
        graph_data: Optional[torch_geometric.data.Data] = None,
    ) -> torch.Tensor:
        """
        관측 데이터 인코딩

        Args:
            obs_data: 관측 데이터 [batch, obs_steps, features]
            graph_data: 그래프 데이터 (선택사항)

        Returns:
            인코딩된 특징 [batch, hidden_dim]
        """
        if self.use_scene_graph and graph_data is not None:
            # GNN 인코딩
            x = graph_data.x
            edge_index = graph_data.edge_index

            for layer in self.graph_encoder:
                x = layer(x, edge_index)
                x = F.relu(x)

            # 그래프 레벨 풀링 (평균)
            graph_embedding = torch.mean(x, dim=0, keepdim=True)
            return graph_embedding
        else:
            # LSTM 인코딩
            lstm_out, (h_n, c_n) = self.lstm_encoder(obs_data)
            # 마지막 시점의 은닉 상태
            return h_n[-1]  # [batch, hidden_dim]

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward Diffusion Process (노이즈 추가)

        Args:
            x_start: 시작 궤적 [batch, pred_steps, 2]
            t: 타임스텝 [batch]
            noise: 노이즈 (None이면 생성)

        Returns:
            노이즈가 추가된 궤적
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(
            -1, 1, 1
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        return_pred_xstart: bool = False,
    ) -> torch.Tensor:
        """
        Reverse Diffusion Process (노이즈 제거)

        Args:
            x_t: 현재 노이즈 궤적 [batch, pred_steps, 2]
            t: 타임스텝 [batch]
            condition: 조건 (관측 인코딩) [batch, hidden_dim]
            return_pred_xstart: 예측된 시작점 반환 여부

        Returns:
            디노이즈된 궤적
        """
        # 타임스텝 임베딩
        t_emb = self.time_embedding(t)
        t_emb = self.time_mlp(t_emb)  # [batch, hidden_dim]

        # 조건 결합
        condition_emb = self.condition_mlp(condition)  # [batch, hidden_dim]
        combined_emb = t_emb + condition_emb  # [batch, hidden_dim]

        # 궤적을 임베딩 공간으로 변환
        x_emb = x_t.view(x_t.size(0), -1)  # [batch, pred_steps * 2]
        x_emb = F.linear(
            x_emb,
            weight=torch.randn(self.hidden_dim, x_emb.size(1), device=x_emb.device),
        )

        # Denoiser (간단한 구현)
        # 실제로는 Transformer Decoder 사용
        denoised_emb = combined_emb.unsqueeze(1).expand(-1, self.pred_steps, -1)
        denoised_emb = self.denoiser(denoised_emb, denoised_emb)
        denoised_emb = denoised_emb.mean(dim=1)  # [batch, hidden_dim]

        # 출력
        pred_xstart = self.output_mlp(denoised_emb)  # [batch, pred_steps * 2]
        pred_xstart = pred_xstart.view(-1, self.pred_steps, 2)

        if return_pred_xstart:
            return pred_xstart

        # 다음 스텝 계산
        pred_noise = (
            x_t - self.sqrt_alphas_cumprod[t].view(-1, 1, 1) * pred_xstart
        ) / self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)

        # 샘플링
        alpha_t = self.alphas[t].view(-1, 1, 1)
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1)
        alpha_cumprod_t_prev = self.alphas_cumprod_prev[t].view(-1, 1, 1)

        pred_xprev = (1.0 / torch.sqrt(alpha_t)) * (
            x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * pred_noise
        )

        return pred_xprev

    def forward(
        self,
        obs_data: torch.Tensor,
        graph_data: Optional[torch_geometric.data.Data] = None,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass (학습용)

        Args:
            obs_data: 관측 데이터
            graph_data: 그래프 데이터
            t: 타임스텝 (None이면 랜덤 샘플링)

        Returns:
            예측된 노이즈
        """
        batch_size = obs_data.size(0)

        # 타임스텝 샘플링
        if t is None:
            t = torch.randint(
                0, self.num_diffusion_steps, (batch_size,), device=obs_data.device
            )

        # 관측 인코딩
        condition = self.encode_observation(obs_data, graph_data)

        # 실제 궤적은 입력에서 추출 (실제 구현 시)
        # 여기서는 더미 사용
        x_start = torch.randn(batch_size, self.pred_steps, 2, device=obs_data.device)

        # Forward diffusion
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)

        # 예측
        pred_noise = self.p_sample(x_t, t, condition, return_pred_xstart=False)

        return pred_noise

    def sample(
        self,
        obs_data: torch.Tensor,
        graph_data: Optional[torch_geometric.data.Data] = None,
        num_samples: int = 1,
        ddim_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        샘플링 (추론용)

        Args:
            obs_data: 관측 데이터
            graph_data: 그래프 데이터
            num_samples: 샘플 수
            ddim_steps: DDIM 스텝 수 (None이면 전체 스텝)

        Returns:
            생성된 궤적 [num_samples, batch, pred_steps, 2]
        """
        self.eval()
        batch_size = obs_data.size(0)
        device = obs_data.device

        # 관측 인코딩
        condition = self.encode_observation(obs_data, graph_data)

        # 샘플링
        samples = []

        for _ in range(num_samples):
            # 노이즈에서 시작
            x_t = torch.randn(batch_size, self.pred_steps, 2, device=device)

            # DDIM 샘플링 (빠른 버전)
            if ddim_steps is not None:
                step_size = self.num_diffusion_steps // ddim_steps
                timesteps = list(range(0, self.num_diffusion_steps, step_size))
            else:
                timesteps = list(range(self.num_diffusion_steps - 1, -1, -1))

            for i, t_val in enumerate(timesteps):
                t = torch.full((batch_size,), t_val, device=device, dtype=torch.long)
                x_t = self.p_sample(x_t, t, condition)

            samples.append(x_t)

        return torch.stack(samples, dim=0)  # [num_samples, batch, pred_steps, 2]


class HybridGNNDiffusion(nn.Module):
    """
    GNN + Diffusion 하이브리드 모델
    GNN으로 상호작용 모델링, Diffusion으로 다중 모달리티 생성
    """

    def __init__(
        self,
        node_features: int = 9,
        hidden_dim: int = 128,
        num_diffusion_steps: int = 100,
        pred_steps: int = 50,
    ):
        super(HybridGNNDiffusion, self).__init__()

        # GNN 인코더 (상호작용 모델링)
        from torch_geometric.nn import GATConv

        self.gnn_encoder = nn.ModuleList(
            [
                GATConv(node_features, hidden_dim, heads=4, concat=False),
                GATConv(hidden_dim, hidden_dim, heads=4, concat=False),
            ]
        )

        # Diffusion 모델
        self.diffusion = DiffusionTrajectoryPredictor(
            node_features=hidden_dim,
            hidden_dim=hidden_dim,
            num_diffusion_steps=num_diffusion_steps,
            pred_steps=pred_steps,
            use_scene_graph=False,  # 이미 GNN으로 인코딩됨
        )

    def forward(
        self, graph_data: torch_geometric.data.Data, t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            graph_data: 그래프 데이터
            t: 타임스텝

        Returns:
            예측된 노이즈
        """
        # GNN 인코딩
        x = graph_data.x
        edge_index = graph_data.edge_index

        for layer in self.gnn_encoder:
            x = layer(x, edge_index)
            x = F.relu(x)

        # 그래프 레벨 특징
        graph_embedding = torch.mean(x, dim=0, keepdim=True)

        # Diffusion (조건부)
        # 관측 데이터는 그래프에서 추출
        obs_data = graph_embedding.unsqueeze(0)  # [1, 1, hidden_dim]

        return self.diffusion(obs_data, graph_data=None, t=t)

    def sample(
        self,
        graph_data: torch_geometric.data.Data,
        num_samples: int = 20,
        ddim_steps: int = 2,
    ) -> torch.Tensor:
        """
        다중 궤적 샘플링

        Args:
            graph_data: 그래프 데이터
            num_samples: 샘플 수
            ddim_steps: DDIM 스텝 수

        Returns:
            생성된 궤적 [num_samples, batch, pred_steps, 2]
        """
        # GNN 인코딩
        x = graph_data.x
        edge_index = graph_data.edge_index

        for layer in self.gnn_encoder:
            x = layer(x, edge_index)
            x = F.relu(x)

        graph_embedding = torch.mean(x, dim=0, keepdim=True)
        obs_data = graph_embedding.unsqueeze(0)

        return self.diffusion.sample(
            obs_data, graph_data=None, num_samples=num_samples, ddim_steps=ddim_steps
        )


def create_diffusion_model(
    node_features: int = 9,
    hidden_dim: int = 128,
    num_diffusion_steps: int = 100,
    pred_steps: int = 50,
    use_gnn: bool = True,
) -> nn.Module:
    """
    Diffusion 모델 생성 헬퍼 함수

    Args:
        node_features: 노드 특징 차원
        hidden_dim: 은닉층 차원
        num_diffusion_steps: Diffusion 스텝 수
        pred_steps: 예측 스텝 수
        use_gnn: GNN 사용 여부 (하이브리드 모델)

    Returns:
        Diffusion 모델
    """
    if use_gnn:
        return HybridGNNDiffusion(
            node_features=node_features,
            hidden_dim=hidden_dim,
            num_diffusion_steps=num_diffusion_steps,
            pred_steps=pred_steps,
        )
    else:
        return DiffusionTrajectoryPredictor(
            node_features=node_features,
            hidden_dim=hidden_dim,
            num_diffusion_steps=num_diffusion_steps,
            pred_steps=pred_steps,
            use_scene_graph=False,
        )


def main():
    """테스트용 메인 함수"""
    # 모델 생성
    model = create_diffusion_model(
        node_features=9,
        hidden_dim=128,
        num_diffusion_steps=100,
        pred_steps=50,
        use_gnn=True,
    )

    print("Diffusion 모델 구조:")
    print(model)

    # 더미 데이터로 테스트
    batch_size = 4
    num_nodes = 10

    # 그래프 데이터
    from torch_geometric.data import Data

    graph_data = Data(
        x=torch.randn(num_nodes, 9), edge_index=torch.randint(0, num_nodes, (2, 20))
    )

    # Forward pass
    pred_noise = model(graph_data)
    print(f"\n예측 노이즈 형태: {pred_noise.shape}")

    # 샘플링
    samples = model.sample(graph_data, num_samples=5, ddim_steps=2)
    print(f"샘플링 결과 형태: {samples.shape}")
    print(f"  샘플 수: {samples.size(0)}")
    print(f"  배치 크기: {samples.size(1)}")
    print(f"  예측 스텝: {samples.size(2)}")


if __name__ == "__main__":
    main()
