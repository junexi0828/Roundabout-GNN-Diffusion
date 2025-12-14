"""
MID + 씬 그래프 통합 모듈
SceneGraphBuilder와 MID 모델 연결
"""

import torch
import numpy as np
import pandas as pd
from typing import Optional, Dict, List
from pathlib import Path

from ..scene_graph.scene_graph_builder import SceneGraphBuilder
from ..models.mid_model import HybridGNNMID, create_mid_model
from torch_geometric.data import Data, HeteroData


class MIDSceneGraphIntegrator:
    """
    MID 모델과 씬 그래프 통합 클래스
    """

    def __init__(
        self,
        mid_model: HybridGNNMID,
        scene_graph_builder: Optional[SceneGraphBuilder] = None,
        spatial_threshold: float = 20.0,
    ):
        """
        Args:
            mid_model: MID 모델
            scene_graph_builder: 씬 그래프 빌더
            spatial_threshold: 공간적 엣지 임계값
        """
        self.mid_model = mid_model
        self.scene_graph_builder = scene_graph_builder or SceneGraphBuilder(
            spatial_threshold=spatial_threshold
        )

    def build_scene_graph_from_frame(self, frame_data: pd.DataFrame) -> Data:
        """
        프레임 데이터로부터 씬 그래프 생성

        Args:
            frame_data: 프레임 데이터 (columns: track_id, x, y, vx, vy, agent_type, ...)

        Returns:
            PyTorch Geometric Data 객체
        """
        # NetworkX 그래프 생성
        nx_graph = self.scene_graph_builder.build_from_frame(frame_data)

        # PyTorch Geometric 변환
        pyg_data = self.scene_graph_builder.to_pytorch_geometric()

        return pyg_data

    def build_hetero_scene_graph(self, frame_data: pd.DataFrame) -> HeteroData:
        """
        이기종 씬 그래프 생성 (HeteroGAT용)

        Args:
            frame_data: 프레임 데이터

        Returns:
            HeteroData 객체
        """
        from torch_geometric.data import HeteroData

        # 에이전트 타입별로 분류
        agent_types = (
            frame_data["agent_type"].unique()
            if "agent_type" in frame_data.columns
            else ["vehicle"]
        )

        hetero_data = HeteroData()

        # 노드 타입별 데이터 준비
        x_dict = {}
        edge_index_dict = {}

        for agent_type in agent_types:
            type_data = frame_data[frame_data["agent_type"] == agent_type]

            if len(type_data) == 0:
                continue

            # 특징 벡터 생성
            features = []
            for _, row in type_data.iterrows():
                feat = [
                    row.get("x", 0.0),
                    row.get("y", 0.0),
                    row.get("vx", 0.0),
                    row.get("vy", 0.0),
                    row.get("ax", 0.0) if "ax" in row else 0.0,
                    row.get("ay", 0.0) if "ay" in row else 0.0,
                    row.get("psi_rad", 0.0) if "psi_rad" in row else 0.0,
                    row.get("width", 0.0) if "width" in row else 0.0,
                    row.get("length", 0.0) if "length" in row else 0.0,
                ]
                features.append(feat)

            x_dict[agent_type] = torch.tensor(features, dtype=torch.float)
            hetero_data[agent_type].x = x_dict[agent_type]

        # 엣지 생성 (간단한 버전)
        # 실제로는 SceneGraphBuilder의 엣지 로직 활용
        positions = frame_data[["x", "y"]].values

        for i, agent_type_i in enumerate(agent_types):
            for j, agent_type_j in enumerate(agent_types):
                if i == j:
                    # 같은 타입 내 엣지
                    type_i_data = frame_data[frame_data["agent_type"] == agent_type_i]
                    if len(type_i_data) < 2:
                        continue

                    # 공간적 엣지 (거리 기반)
                    pos_i = type_i_data[["x", "y"]].values
                    from scipy.spatial.distance import cdist

                    distances = cdist(pos_i, pos_i)

                    # 임계값 내 엣지
                    threshold = 20.0
                    edge_indices = np.where((distances < threshold) & (distances > 0))

                    if len(edge_indices[0]) > 0:
                        edge_index = np.stack(
                            [edge_indices[0], edge_indices[1]], axis=0
                        )
                        edge_type = (agent_type_i, "spatial", agent_type_j)
                        hetero_data[edge_type].edge_index = torch.tensor(
                            edge_index, dtype=torch.long
                        )
                else:
                    # 다른 타입 간 엣지
                    type_i_data = frame_data[frame_data["agent_type"] == agent_type_i]
                    type_j_data = frame_data[frame_data["agent_type"] == agent_type_j]

                    if len(type_i_data) == 0 or len(type_j_data) == 0:
                        continue

                    pos_i = type_i_data[["x", "y"]].values
                    pos_j = type_j_data[["x", "y"]].values

                    distances = cdist(pos_i, pos_j)
                    threshold = 20.0
                    edge_indices = np.where(distances < threshold)

                    if len(edge_indices[0]) > 0:
                        # HeteroData는 인덱스를 타입별로 관리
                        edge_index = np.stack(
                            [edge_indices[0], edge_indices[1]], axis=0
                        )
                        edge_type = (agent_type_i, "spatial", agent_type_j)
                        hetero_data[edge_type].edge_index = torch.tensor(
                            edge_index, dtype=torch.long
                        )

        return hetero_data

    def predict_with_scene_graph(
        self,
        frame_data: pd.DataFrame,
        use_hetero: bool = True,
        num_samples: int = 20,
        ddim_steps: int = 2,
    ) -> torch.Tensor:
        """
        씬 그래프를 사용한 예측

        Args:
            frame_data: 프레임 데이터
            use_hetero: HeteroGAT 사용 여부
            num_samples: 샘플 수
            ddim_steps: DDIM 스텝 수

        Returns:
            생성된 궤적 [num_samples, batch, pred_steps, 2]
        """
        if use_hetero and self.mid_model.use_hetero_gnn:
            # 이기종 그래프 생성
            hetero_data = self.build_hetero_scene_graph(frame_data)
            return self.mid_model.sample(
                hetero_data=hetero_data, num_samples=num_samples, ddim_steps=ddim_steps
            )
        else:
            # 일반 그래프 생성
            graph_data = self.build_scene_graph_from_frame(frame_data)
            return self.mid_model.sample(
                graph_data=graph_data, num_samples=num_samples, ddim_steps=ddim_steps
            )


def create_mid_with_scene_graph(
    mid_model: HybridGNNMID, spatial_threshold: float = 20.0
) -> MIDSceneGraphIntegrator:
    """MID + 씬 그래프 통합 모델 생성"""
    return MIDSceneGraphIntegrator(
        mid_model=mid_model, spatial_threshold=spatial_threshold
    )
