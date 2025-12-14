"""
시공간 씬 그래프 생성 모듈
여러 프레임에 걸친 동적 씬 그래프 생성 및 시계열 처리
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path
from .scene_graph_builder import SceneGraphBuilder
import torch
from torch_geometric.data import Data, Batch
from torch_geometric_temporal.signal import StaticGraphTemporalSignal


class TemporalSceneGraphBuilder:
    """시공간 씬 그래프 생성 클래스"""

    def __init__(
        self,
        spatial_threshold: float = 20.0,
        use_kdtree: bool = True
    ):
        """
        Args:
            spatial_threshold: 공간적 엣지 생성 거리 임계값
            use_kdtree: KD-Tree 사용 여부
        """
        self.builder = SceneGraphBuilder(
            spatial_threshold=spatial_threshold,
            use_kdtree=use_kdtree
        )

    def build_temporal_graphs(
        self,
        trajectory_data: pd.DataFrame,
        map_data: Optional[pd.DataFrame] = None,
        frame_ids: Optional[List[int]] = None
    ) -> List[Data]:
        """
        여러 프레임에 걸쳐 시공간 그래프 시퀀스를 생성합니다.

        Args:
            trajectory_data: 전체 궤적 데이터 (frame_id 컬럼 포함)
            map_data: 맵 데이터 (선택사항)
            frame_ids: 처리할 프레임 ID 목록 (None이면 모든 프레임)

        Returns:
            PyTorch Geometric Data 객체 리스트 (프레임 순서대로)
        """
        if frame_ids is None:
            frame_ids = sorted(trajectory_data['frame_id'].unique())

        temporal_graphs = []

        for frame_id in frame_ids:
            frame_data = trajectory_data[trajectory_data['frame_id'] == frame_id]

            if len(frame_data) == 0:
                continue

            # 씬 그래프 생성
            nx_graph = self.builder.build_from_frame(frame_data, map_data)

            # PyTorch Geometric 변환
            pyg_data = self.builder.to_pytorch_geometric()

            # 프레임 ID 추가
            pyg_data.frame_id = frame_id

            temporal_graphs.append(pyg_data)

        return temporal_graphs

    def build_static_temporal_signal(
        self,
        trajectory_data: pd.DataFrame,
        map_data: Optional[pd.DataFrame] = None,
        obs_window: int = 30,
        pred_window: int = 50
    ) -> StaticGraphTemporalSignal:
        """
        PyTorch Geometric Temporal의 StaticGraphTemporalSignal 형식으로 변환합니다.

        Args:
            trajectory_data: 전체 궤적 데이터
            map_data: 맵 데이터
            obs_window: 관측 윈도우 길이
            pred_window: 예측 윈도우 길이

        Returns:
            StaticGraphTemporalSignal 객체
        """
        frame_ids = sorted(trajectory_data['frame_id'].unique())

        # 첫 프레임으로 정적 그래프 구조 생성
        first_frame = trajectory_data[trajectory_data['frame_id'] == frame_ids[0]]
        nx_graph = self.builder.build_from_frame(first_frame, map_data)
        static_data = self.builder.to_pytorch_geometric()

        # 엣지 인덱스는 정적 (모든 프레임에서 동일)
        edge_index = static_data.edge_index

        # 각 프레임의 노드 특징 추출
        features = []
        targets = []

        for i, frame_id in enumerate(frame_ids):
            frame_data = trajectory_data[trajectory_data['frame_id'] == frame_id]

            if len(frame_data) == 0:
                continue

            # 노드 특징 (위치, 속도 등)
            frame_features = []
            for _, row in frame_data.iterrows():
                feature = np.array([
                    row['x'], row['y'],
                    row.get('vx', 0.0), row.get('vy', 0.0),
                    row.get('psi_rad', 0.0)
                ])
                frame_features.append(feature)

            features.append(torch.tensor(np.array(frame_features), dtype=torch.float))

            # 타겟 (다음 프레임의 위치) - 예측용
            if i < len(frame_ids) - 1:
                next_frame = trajectory_data[trajectory_data['frame_id'] == frame_ids[i + 1]]
                next_positions = []
                for _, row in frame_data.iterrows():
                    track_id = row['track_id']
                    next_row = next_frame[next_frame['track_id'] == track_id]
                    if len(next_row) > 0:
                        next_positions.append([next_row.iloc[0]['x'], next_row.iloc[0]['y']])
                    else:
                        next_positions.append([row['x'], row['y']])  # 없으면 현재 위치
                targets.append(torch.tensor(np.array(next_positions), dtype=torch.float))

        # StaticGraphTemporalSignal 생성
        # 주의: 실제 사용 시 노드 수가 프레임마다 다를 수 있으므로 동적 그래프 고려 필요
        return StaticGraphTemporalSignal(
            edge_index=edge_index,
            edge_weight=static_data.edge_attr[:, 0] if static_data.edge_attr is not None else None,
            features=features,
            targets=targets
        )


def main():
    """테스트용 메인 함수"""
    # 예시 시계열 데이터 생성
    num_frames = 10
    num_vehicles = 3

    data_list = []
    for frame_id in range(num_frames):
        for track_id in range(num_vehicles):
            data_list.append({
                'frame_id': frame_id,
                'track_id': track_id,
                'x': track_id * 5.0 + frame_id * 0.5,
                'y': 0.0,
                'vx': 5.0,
                'vy': 0.0,
                'psi_rad': 0.0,
                'width': 2.0,
                'length': 4.5
            })

    trajectory_data = pd.DataFrame(data_list)

    builder = TemporalSceneGraphBuilder(spatial_threshold=15.0)
    graphs = builder.build_temporal_graphs(trajectory_data)

    print(f"생성된 시공간 그래프 수: {len(graphs)}")
    if graphs:
        print(f"첫 번째 그래프:")
        print(f"  노드 수: {graphs[0].num_nodes}")
        print(f"  엣지 수: {graphs[0].edge_index.size(1)}")


if __name__ == "__main__":
    main()

