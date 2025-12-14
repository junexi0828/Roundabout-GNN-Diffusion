"""
MID (Motion Indeterminacy Diffusion) 모델 구현
CVPR 2022: "Stochastic Trajectory Prediction via Motion Indeterminacy Diffusion"

핵심 개념:
- Motion Indeterminacy를 명시적으로 모델링
- 모든 가능한 보행 영역에서 시작하여 점진적으로 불확정성 제거
- Transformer 기반 Denoiser
- DDIM 지원으로 빠른 샘플링
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List
import math

from torch_geometric.data import Data, HeteroData


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


class TransformerDenoiser(nn.Module):
    """
    Transformer 기반 Denoiser
    MID 논문의 핵심 구성요소
    """

    def __init__(
        self,
        input_dim: int = 2,  # (x, y)
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        pred_steps: int = 50
    ):
        super(TransformerDenoiser, self).__init__()

        self.hidden_dim = hidden_dim
        self.pred_steps = pred_steps

        # 입력 프로젝션
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # 타임스텝 임베딩
        self.time_embedding = SinusoidalPositionalEmbedding(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 조건 임베딩 (관측 정보)
        self.condition_proj = nn.Linear(hidden_dim, hidden_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 출력 레이어
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)  # (x, y)
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Denoiser forward pass

        Args:
            x_t: 노이즈 궤적 [batch, pred_steps, 2]
            t: 타임스텝 [batch]
            condition: 조건 (관측 인코딩) [batch, hidden_dim]

        Returns:
            예측된 노이즈 [batch, pred_steps, 2]
        """
        batch_size = x_t.size(0)

        # 입력 프로젝션
        x_emb = self.input_proj(x_t)  # [batch, pred_steps, hidden_dim]

        # 타임스텝 임베딩
        t_emb = self.time_embedding(t)  # [batch, hidden_dim]
        t_emb = self.time_mlp(t_emb)  # [batch, hidden_dim]
        t_emb = t_emb.unsqueeze(1).expand(-1, self.pred_steps, -1)  # [batch, pred_steps, hidden_dim]

        # 조건 임베딩
        cond_emb = self.condition_proj(condition)  # [batch, hidden_dim]
        cond_emb = cond_emb.unsqueeze(1).expand(-1, self.pred_steps, -1)  # [batch, pred_steps, hidden_dim]

        # 결합
        x_emb = x_emb + t_emb + cond_emb  # [batch, pred_steps, hidden_dim]

        # Transformer 통과
        x_emb = self.transformer(x_emb)  # [batch, pred_steps, hidden_dim]

        # 출력
        pred_noise = self.output_mlp(x_emb)  # [batch, pred_steps, 2]

        return pred_noise


class ObservationEncoder(nn.Module):
    """
    관측 궤적 인코더
    LSTM 또는 Transformer 기반
    """

    def __init__(
        self,
        input_dim: int = 2,  # (x, y)
        hidden_dim: int = 128,
        num_layers: int = 2,
        use_transformer: bool = False
    ):
        super(ObservationEncoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.use_transformer = use_transformer

        if use_transformer:
            # Transformer 기반 인코더
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.input_proj = nn.Linear(input_dim, hidden_dim)
        else:
            # LSTM 기반 인코더 (MID 논문 기본)
            self.encoder = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True
            )
            # 양방향 LSTM이므로 hidden_dim * 2
            self.hidden_dim = hidden_dim * 2

        # 출력 프로젝션
        self.output_proj = nn.Linear(self.hidden_dim, hidden_dim)

    def forward(self, obs_trajectory: torch.Tensor) -> torch.Tensor:
        """
        관측 궤적 인코딩

        Args:
            obs_trajectory: 관측 궤적 [batch, obs_steps, 2]

        Returns:
            인코딩된 특징 [batch, hidden_dim]
        """
        if self.use_transformer:
            # Transformer
            x_emb = self.input_proj(obs_trajectory)  # [batch, obs_steps, hidden_dim]
            encoded = self.encoder(x_emb)  # [batch, obs_steps, hidden_dim]
            # 마지막 시점 사용
            encoded = encoded[:, -1, :]  # [batch, hidden_dim]
        else:
            # LSTM
            encoded, (h_n, c_n) = self.encoder(obs_trajectory)
            # 양방향이므로 마지막 레이어의 forward/backward 결합
            h_n = h_n.view(self.encoder.num_layers, 2, encoded.size(0), -1)
            h_forward = h_n[-1, 0]  # [batch, hidden_dim]
            h_backward = h_n[-1, 1]  # [batch, hidden_dim]
            encoded = torch.cat([h_forward, h_backward], dim=1)  # [batch, hidden_dim * 2]

        # 출력 프로젝션
        output = self.output_proj(encoded)  # [batch, hidden_dim]

        return output


class MIDModel(nn.Module):
    """
    MID (Motion Indeterminacy Diffusion) 메인 모델

    아키텍처:
    1. Observation Encoder: 관측 궤적 인코딩
    2. Diffusion Process: Forward/Reverse diffusion
    3. Transformer Denoiser: 노이즈 제거
    """

    def __init__(
        self,
        obs_steps: int = 30,  # 관측 스텝 수 (3초, 10Hz)
        pred_steps: int = 50,  # 예측 스텝 수 (5초, 10Hz)
        hidden_dim: int = 128,
        num_diffusion_steps: int = 100,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        use_transformer_encoder: bool = False
    ):
        super(MIDModel, self).__init__()

        self.obs_steps = obs_steps
        self.pred_steps = pred_steps
        self.hidden_dim = hidden_dim
        self.num_diffusion_steps = num_diffusion_steps

        # 관측 인코더
        self.obs_encoder = ObservationEncoder(
            input_dim=2,  # (x, y)
            hidden_dim=hidden_dim,
            num_layers=2,
            use_transformer=use_transformer_encoder
        )

        # Denoiser
        self.denoiser = TransformerDenoiser(
            input_dim=2,
            hidden_dim=hidden_dim,
            num_layers=4,
            num_heads=8,
            dropout=0.1,
            pred_steps=pred_steps
        )

        # Beta 스케줄 (선형)
        betas = torch.linspace(beta_start, beta_end, num_diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # 등록 (버퍼로 저장)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # DDIM용
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
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
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        return_pred_xstart: bool = False
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
        # Denoiser로 노이즈 예측
        pred_noise = self.denoiser(x_t, t, condition)  # [batch, pred_steps, 2]

        # 예측된 시작점 계산
        pred_xstart = (
            self.sqrt_recip_alphas_cumprod[t].view(-1, 1, 1) * x_t -
            self.sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1) * pred_noise
        )

        if return_pred_xstart:
            return pred_xstart

        # 다음 스텝 계산
        alpha_t = self.alphas[t].view(-1, 1, 1)
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1)
        alpha_cumprod_t_prev = self.alphas_cumprod_prev[t].view(-1, 1, 1)

        # 노이즈 샘플링
        noise = torch.randn_like(x_t)
        pred_xprev = (
            torch.sqrt(alpha_cumprod_t_prev) * pred_xstart +
            torch.sqrt(1.0 - alpha_cumprod_t_prev) * noise
        )

        return pred_xprev

    def p_sample_ddim(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        condition: torch.Tensor,
        eta: float = 0.0
    ) -> torch.Tensor:
        """
        DDIM 샘플링 (빠른 샘플링)

        Args:
            x_t: 현재 노이즈 궤적
            t: 현재 타임스텝
            t_prev: 이전 타임스텝
            condition: 조건
            eta: DDIM 파라미터 (0이면 deterministic)

        Returns:
            다음 스텝 궤적
        """
        # 노이즈 예측
        pred_noise = self.denoiser(x_t, t, condition)

        # 예측된 시작점
        pred_xstart = (
            self.sqrt_recip_alphas_cumprod[t].view(-1, 1, 1) * x_t -
            self.sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1) * pred_noise
        )

        # DDIM 샘플링
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1)
        alpha_cumprod_t_prev = self.alphas_cumprod_prev[t_prev].view(-1, 1, 1)

        pred_dir = torch.sqrt(1.0 - alpha_cumprod_t_prev) * pred_noise

        if eta > 0:
            noise = torch.randn_like(x_t)
            random_noise = torch.sqrt(1.0 - alpha_cumprod_t_prev) * noise
            pred_dir = pred_dir + eta * random_noise

        x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_xstart + pred_dir

        return x_prev

    def forward(
        self,
        obs_trajectory: torch.Tensor,
        future_trajectory: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        x_t: Optional[torch.Tensor] = None,
        graph_data: Optional[Data] = None,  # GNN 미사용 시 무시
        hetero_data: Optional[HeteroData] = None,  # GNN 미사용 시 무시
        **kwargs  # 추가 인수 무시
    ) -> torch.Tensor:
        """
        Forward pass (학습용)

        Args:
            obs_trajectory: 관측 궤적 [batch, obs_steps, 2]
            future_trajectory: 미래 궤적 (학습용) [batch, pred_steps, 2]
            t: 타임스텝 (None이면 랜덤 샘플링)

        Returns:
            예측된 노이즈 [batch, pred_steps, 2]
        """
        batch_size = obs_trajectory.size(0)
        device = obs_trajectory.device

        # 관측 인코딩
        condition = self.obs_encoder(obs_trajectory)  # [batch, hidden_dim]

        # 타임스텝 샘플링
        if t is None:
            t = torch.randint(0, self.num_diffusion_steps, (batch_size,), device=device)

        # Forward diffusion
        if x_t is not None:
            # 이미 노이즈가 추가된 궤적이 제공된 경우
            pass
        elif future_trajectory is not None:
            # Forward diffusion 수행
            noise = torch.randn_like(future_trajectory)
            x_t = self.q_sample(future_trajectory, t, noise)
        else:
            # 더미 데이터 (실제로는 학습 시 future_trajectory 또는 x_t 필요)
            x_t = torch.randn(batch_size, self.pred_steps, 2, device=device)

        # 노이즈 예측
        pred_noise = self.denoiser(x_t, t, condition)

        return pred_noise

    def sample(
        self,
        obs_trajectory: torch.Tensor,
        num_samples: int = 20,
        ddim_steps: Optional[int] = None,
        ddim_eta: float = 0.0
    ) -> torch.Tensor:
        """
        샘플링 (추론용)

        Args:
            obs_trajectory: 관측 궤적 [batch, obs_steps, 2]
            num_samples: 샘플 수
            ddim_steps: DDIM 스텝 수 (None이면 전체 스텝)
            ddim_eta: DDIM 파라미터

        Returns:
            생성된 궤적 [num_samples, batch, pred_steps, 2]
        """
        self.eval()
        batch_size = obs_trajectory.size(0)
        device = obs_trajectory.device

        # 관측 인코딩
        condition = self.obs_encoder(obs_trajectory)  # [batch, hidden_dim]

        samples = []

        for _ in range(num_samples):
            # 노이즈에서 시작
            x_t = torch.randn(batch_size, self.pred_steps, 2, device=device)

            # DDIM 샘플링 (빠른 버전)
            if ddim_steps is not None:
                step_size = self.num_diffusion_steps // ddim_steps
                timesteps = list(range(0, self.num_diffusion_steps, step_size))
                if timesteps[-1] != self.num_diffusion_steps - 1:
                    timesteps.append(self.num_diffusion_steps - 1)

                for i in range(len(timesteps) - 1):
                    t_val = timesteps[-(i+1)]
                    t_prev_val = timesteps[-(i+2)] if i+2 < len(timesteps) else 0

                    t = torch.full((batch_size,), t_val, device=device, dtype=torch.long)
                    t_prev = torch.full((batch_size,), t_prev_val, device=device, dtype=torch.long)

                    x_t = self.p_sample_ddim(x_t, t, t_prev, condition, ddim_eta)
            else:
                # 전체 스텝 샘플링
                for t_val in range(self.num_diffusion_steps - 1, -1, -1):
                    t = torch.full((batch_size,), t_val, device=device, dtype=torch.long)
                    x_t = self.p_sample(x_t, t, condition)

            samples.append(x_t)

        return torch.stack(samples, dim=0)  # [num_samples, batch, pred_steps, 2]


class HybridGNNMID(nn.Module):
    """
    GNN + MID 하이브리드 모델
    GNN으로 상호작용 모델링, MID로 다중 모달리티 생성

    개선사항:
    - HeteroGAT 통합 (이기종 에이전트 처리)
    - 씬 그래프 통합
    - Plan B 안전 검증
    """

    def __init__(
        self,
        node_features: int = 9,
        hidden_dim: int = 128,
        obs_steps: int = 30,
        pred_steps: int = 50,
        num_diffusion_steps: int = 100,
        use_gnn: bool = True,
        use_hetero_gnn: bool = True,
        node_types: Optional[List[str]] = None,
        edge_types: Optional[List[Tuple[str, str, str]]] = None
    ):
        super(HybridGNNMID, self).__init__()

        self.use_gnn = use_gnn
        self.use_hetero_gnn = use_hetero_gnn and use_gnn

        if use_gnn:
            if self.use_hetero_gnn and node_types and edge_types:
                # HeteroGAT 사용 (이기종 에이전트 처리)
                from .heterogeneous_gnn import HeteroGAT

                self.gnn_encoder = HeteroGAT(
                    node_types=node_types,
                    edge_types=edge_types,
                    in_channels=node_features,
                    hidden_channels=hidden_dim,
                    out_channels=hidden_dim,
                    num_heads=4,
                    num_layers=2
                )
                self.is_hetero = True
            else:
                # 기본 GAT 사용
                from torch_geometric.nn import GATConv

                self.gnn_encoder = nn.ModuleList([
                    GATConv(node_features, hidden_dim, heads=4, concat=False),
                    GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
                ])
                self.is_hetero = False
        else:
            # 기본 인코더 (노드 특징만 사용)
            self.node_encoder = nn.Linear(node_features, hidden_dim)
            self.is_hetero = False

        # MID 모델
        self.mid = MIDModel(
            obs_steps=obs_steps,
            pred_steps=pred_steps,
            hidden_dim=hidden_dim,
            num_diffusion_steps=num_diffusion_steps
        )

        # GNN 특징을 관측 궤적 형태로 변환 (MID 입력용)
        self.gnn_to_obs = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, obs_steps * 2)  # (x, y) * obs_steps
        )

    def forward(
        self,
        graph_data: Optional[Data] = None,
        hetero_data: Optional[HeteroData] = None,
        obs_trajectory: Optional[torch.Tensor] = None,
        future_trajectory: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        x_t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            graph_data: 그래프 데이터 (일반 GNN 사용 시)
            hetero_data: 이기종 그래프 데이터 (HeteroGAT 사용 시)
            obs_trajectory: 관측 궤적 (GNN 미사용 시)
            future_trajectory: 미래 궤적 (학습용)
            t: 타임스텝
            x_t: 이미 노이즈가 추가된 궤적 (학습용)

        Returns:
            예측된 노이즈
        """
        if self.use_gnn and (graph_data is not None or hetero_data is not None):
            # GNN 인코딩
            if self.is_hetero and hetero_data is not None:
                # HeteroGAT 사용
                x_dict = {node_type: hetero_data[node_type].x
                         for node_type in hetero_data.node_types}
                edge_index_dict = {
                    edge_type: hetero_data[edge_type].edge_index
                    for edge_type in hetero_data.edge_types
                }

                # HeteroGAT forward
                out_dict = self.gnn_encoder(x_dict, edge_index_dict)

                # 모든 노드 타입의 임베딩 결합
                all_embeddings = torch.cat(list(out_dict.values()), dim=0)
                graph_embedding = torch.mean(all_embeddings, dim=0, keepdim=True)  # [1, hidden_dim]
            elif graph_data is not None:
                # 일반 GAT 사용
                x = graph_data.x
                edge_index = graph_data.edge_index

                for layer in self.gnn_encoder:
                    x = layer(x, edge_index)
                    x = F.relu(x)

                # 그래프 레벨 특징 (평균 풀링)
                graph_embedding = torch.mean(x, dim=0, keepdim=True)  # [1, hidden_dim]
            else:
                raise ValueError("graph_data or hetero_data required when use_gnn=True")

            # 관측 궤적 형태로 변환
            obs_emb = self.gnn_to_obs(graph_embedding)  # [1, obs_steps * 2]
            obs_emb = obs_emb.view(1, -1, 2)  # [1, obs_steps, 2]

            # 배치 차원 확장
            if self.is_hetero and hetero_data is not None:
                batch_size = sum(hetero_data[node_type].x.size(0)
                               for node_type in hetero_data.node_types)
            elif graph_data is not None:
                batch_size = graph_data.x.size(0) if hasattr(graph_data, 'batch') else 1
            else:
                batch_size = 1

            obs_emb = obs_emb.expand(batch_size, -1, -1)

            # MID forward
            return self.mid(obs_emb, future_trajectory, t, x_t)
        else:
            # MID만 사용
            if obs_trajectory is None:
                raise ValueError("obs_trajectory required when use_gnn=False")
            return self.mid(obs_trajectory, future_trajectory, t, x_t)

    def sample(
        self,
        graph_data: Optional[Data] = None,
        hetero_data: Optional[HeteroData] = None,
        obs_trajectory: Optional[torch.Tensor] = None,
        num_samples: int = 20,
        ddim_steps: int = 2
    ) -> torch.Tensor:
        """
        다중 궤적 샘플링

        Args:
            graph_data: 그래프 데이터 (일반 GNN)
            hetero_data: 이기종 그래프 데이터 (HeteroGAT)
            obs_trajectory: 관측 궤적
            num_samples: 샘플 수
            ddim_steps: DDIM 스텝 수

        Returns:
            생성된 궤적 [num_samples, batch, pred_steps, 2]
        """
        if self.use_gnn and (graph_data is not None or hetero_data is not None):
            # GNN 인코딩
            if self.is_hetero and hetero_data is not None:
                # HeteroGAT 사용
                x_dict = {node_type: hetero_data[node_type].x
                         for node_type in hetero_data.node_types}
                edge_index_dict = {
                    edge_type: hetero_data[edge_type].edge_index
                    for edge_type in hetero_data.edge_types
                }

                out_dict = self.gnn_encoder(x_dict, edge_index_dict)
                all_embeddings = torch.cat(list(out_dict.values()), dim=0)
                graph_embedding = torch.mean(all_embeddings, dim=0, keepdim=True)

                batch_size = sum(hetero_data[node_type].x.size(0)
                               for node_type in hetero_data.node_types)
            elif graph_data is not None:
                # 일반 GAT 사용
                x = graph_data.x
                edge_index = graph_data.edge_index

                for layer in self.gnn_encoder:
                    x = layer(x, edge_index)
                    x = F.relu(x)

                graph_embedding = torch.mean(x, dim=0, keepdim=True)
                batch_size = graph_data.x.size(0) if hasattr(graph_data, 'batch') else 1
            else:
                raise ValueError("graph_data or hetero_data required when use_gnn=True")

            obs_emb = self.gnn_to_obs(graph_embedding)
            obs_emb = obs_emb.view(1, -1, 2)
            obs_emb = obs_emb.expand(batch_size, -1, -1)

            return self.mid.sample(obs_emb, num_samples=num_samples, ddim_steps=ddim_steps)
        else:
            if obs_trajectory is None:
                raise ValueError("obs_trajectory required when use_gnn=False")
            return self.mid.sample(obs_trajectory, num_samples=num_samples, ddim_steps=ddim_steps)


def create_mid_model(
    obs_steps: int = 30,
    pred_steps: int = 50,
    hidden_dim: int = 128,
    num_diffusion_steps: int = 100,
    use_gnn: bool = True,
    node_features: int = 9,
    use_hetero_gnn: bool = True,
    node_types: Optional[List[str]] = None,
    edge_types: Optional[List[Tuple[str, str, str]]] = None
) -> nn.Module:
    """
    MID 모델 생성 헬퍼 함수

    Args:
        obs_steps: 관측 스텝 수
        pred_steps: 예측 스텝 수
        hidden_dim: 은닉층 차원
        num_diffusion_steps: Diffusion 스텝 수
        use_gnn: GNN 사용 여부
        node_features: 노드 특징 차원 (GNN 사용 시)

    Returns:
        MID 모델
    """
    if use_gnn:
        # 기본 node_types와 edge_types 설정 (제공되지 않은 경우)
        if node_types is None:
            node_types = ['car', 'pedestrian', 'biker', 'skater', 'cart', 'bus']
        if edge_types is None:
            edge_types = [
                ('car', 'spatial', 'car'),
                ('car', 'spatial', 'pedestrian'),
                ('pedestrian', 'spatial', 'biker'),
                ('biker', 'spatial', 'car'),
            ]

        return HybridGNNMID(
            node_features=node_features,
            hidden_dim=hidden_dim,
            obs_steps=obs_steps,
            pred_steps=pred_steps,
            num_diffusion_steps=num_diffusion_steps,
            use_gnn=True,
            use_hetero_gnn=use_hetero_gnn,
            node_types=node_types,
            edge_types=edge_types
        )
    else:
        return MIDModel(
            obs_steps=obs_steps,
            pred_steps=pred_steps,
            hidden_dim=hidden_dim,
            num_diffusion_steps=num_diffusion_steps
        )


def main():
    """테스트용 메인 함수"""
    print("=" * 80)
    print("MID 모델 테스트")
    print("=" * 80)

    # 모델 생성
    model = create_mid_model(
        obs_steps=30,
        pred_steps=50,
        hidden_dim=128,
        num_diffusion_steps=100,
        use_gnn=True,
        node_features=9
    )

    print(f"\n모델 구조:")
    print(model)

    # 더미 데이터
    batch_size = 4
    num_nodes = 10

    # 그래프 데이터
    from torch_geometric.data import Data
    graph_data = Data(
        x=torch.randn(num_nodes, 9),
        edge_index=torch.randint(0, num_nodes, (2, 20))
    )

    # Forward pass
    print(f"\n[Forward Pass 테스트]")
    pred_noise = model(graph_data=graph_data)
    print(f"✓ 예측 노이즈 형태: {pred_noise.shape}")

    # 샘플링
    print(f"\n[샘플링 테스트]")
    samples = model.sample(graph_data=graph_data, num_samples=5, ddim_steps=2)
    print(f"✓ 샘플링 결과 형태: {samples.shape}")
    print(f"  샘플 수: {samples.size(0)}")
    print(f"  배치 크기: {samples.size(1)}")
    print(f"  예측 스텝: {samples.size(2)}")

    # MID만 사용 (GNN 없이)
    print(f"\n[MID Only 테스트]")
    mid_only = create_mid_model(use_gnn=False)
    obs_traj = torch.randn(batch_size, 30, 2)
    pred_noise = mid_only(obs_trajectory=obs_traj)
    print(f"✓ MID only 예측 형태: {pred_noise.shape}")

    samples = mid_only.sample(obs_trajectory=obs_traj, num_samples=3, ddim_steps=2)
    print(f"✓ MID only 샘플링 형태: {samples.shape}")

    print("\n" + "=" * 80)
    print("✓ MID 모델 테스트 완료!")
    print("=" * 80)


if __name__ == "__main__":
    main()

