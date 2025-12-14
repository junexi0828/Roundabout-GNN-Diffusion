"""
씬 그래프 생성 모듈
- 노드 정의 (Agent Node, Map Node)
- 엣지 정의 (Spatial Edge, Semantic Edge - Conflict, Yielding, Following)
- KD-Tree를 활용한 공간 쿼리 최적화
- NetworkX/PyTorch Geometric 변환
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set
from scipy.spatial import cKDTree
from dataclasses import dataclass
import torch
from torch_geometric.data import Data
from pathlib import Path


@dataclass
class AgentNode:
    """에이전트 노드 데이터 클래스"""
    track_id: int
    x: float
    y: float
    vx: float
    vy: float
    ax: float = 0.0
    ay: float = 0.0
    psi: float = 0.0  # 헤딩 각도 (rad)
    width: float = 0.0
    length: float = 0.0
    agent_type: str = "vehicle"

    def to_feature_vector(self) -> np.ndarray:
        """노드 특징 벡터로 변환"""
        return np.array([
            self.x, self.y,
            self.vx, self.vy,
            self.ax, self.ay,
            self.psi,
            self.width, self.length
        ])


@dataclass
class MapNode:
    """맵 노드 데이터 클래스 (차선 중심선 세그먼트)"""
    lanelet_id: int
    x_mid: float
    y_mid: float
    curvature: float = 0.0
    speed_limit: float = 0.0
    is_entry: bool = False
    is_exit: bool = False

    def to_feature_vector(self) -> np.ndarray:
        """노드 특징 벡터로 변환"""
        return np.array([
            self.x_mid, self.y_mid,
            self.curvature,
            self.speed_limit,
            float(self.is_entry),
            float(self.is_exit)
        ])


class SceneGraphBuilder:
    """씬 그래프 생성 클래스"""

    def __init__(
        self,
        spatial_threshold: float = 20.0,  # 공간적 엣지 생성 거리 임계값 (m)
        use_kdtree: bool = True  # KD-Tree 사용 여부
    ):
        """
        Args:
            spatial_threshold: 공간적 엣지를 생성할 최대 거리 (미터)
            use_kdtree: KD-Tree를 사용하여 공간 쿼리 최적화 여부
        """
        self.spatial_threshold = spatial_threshold
        self.use_kdtree = use_kdtree
        self.graph = None

    def build_from_frame(
        self,
        frame_data: pd.DataFrame,
        map_data: Optional[pd.DataFrame] = None
    ) -> nx.MultiDiGraph:
        """
        특정 프레임의 데이터로부터 씬 그래프를 생성합니다.

        Args:
            frame_data: 프레임별 차량 데이터 (columns: track_id, x, y, vx, vy, ...)
            map_data: 맵 데이터 (선택사항, columns: lanelet_id, x_mid, y_mid, ...)

        Returns:
            NetworkX MultiDiGraph 객체
        """
        self.graph = nx.MultiDiGraph()

        # 1. Agent 노드 추가
        agent_nodes = self._create_agent_nodes(frame_data)
        for node_id, agent in agent_nodes.items():
            self.graph.add_node(
                node_id,
                type='agent',
                features=agent.to_feature_vector(),
                **agent.__dict__
            )

        # 2. Map 노드 추가 (있는 경우)
        if map_data is not None:
            map_nodes = self._create_map_nodes(map_data)
            for node_id, map_node in map_nodes.items():
                self.graph.add_node(
                    node_id,
                    type='map',
                    features=map_node.to_feature_vector(),
                    **map_node.__dict__
                )

        # 3. Spatial Edge 추가
        self._add_spatial_edges(agent_nodes)

        # 4. Semantic Edge 추가
        self._add_semantic_edges(agent_nodes, map_data)

        return self.graph

    def _create_agent_nodes(self, frame_data: pd.DataFrame) -> Dict[int, AgentNode]:
        """에이전트 노드 생성"""
        agent_nodes = {}

        for _, row in frame_data.iterrows():
            track_id = int(row['track_id'])

            # 가속도 계산 (이전 프레임과 비교)
            ax, ay = 0.0, 0.0
            if 'ax' in row:
                ax = row['ax']
            if 'ay' in row:
                ay = row['ay']

            agent = AgentNode(
                track_id=track_id,
                x=float(row['x']),
                y=float(row['y']),
                vx=float(row.get('vx', 0.0)),
                vy=float(row.get('vy', 0.0)),
                ax=ax,
                ay=ay,
                psi=float(row.get('psi_rad', row.get('psi', 0.0))),
                width=float(row.get('width', 2.0)),
                length=float(row.get('length', 4.5)),
                agent_type=str(row.get('agent_type', 'vehicle'))
            )

            agent_nodes[track_id] = agent

        return agent_nodes

    def _create_map_nodes(self, map_data: pd.DataFrame) -> Dict[int, MapNode]:
        """맵 노드 생성"""
        map_nodes = {}

        for _, row in map_data.iterrows():
            lanelet_id = int(row['lanelet_id'])

            map_node = MapNode(
                lanelet_id=lanelet_id,
                x_mid=float(row['x_mid']),
                y_mid=float(row['y_mid']),
                curvature=float(row.get('curvature', 0.0)),
                speed_limit=float(row.get('speed_limit', 0.0)),
                is_entry=bool(row.get('is_entry', False)),
                is_exit=bool(row.get('is_exit', False))
            )

            map_nodes[lanelet_id] = map_node

        return map_nodes

    def _add_spatial_edges(self, agent_nodes: Dict[int, AgentNode]):
        """공간적 엣지 추가 (물리적 거리 기반)"""
        if len(agent_nodes) < 2:
            return

        # 위치 배열 생성
        positions = np.array([[agent.x, agent.y] for agent in agent_nodes.values()])
        track_ids = list(agent_nodes.keys())

        if self.use_kdtree:
            # KD-Tree를 사용한 효율적인 공간 쿼리
            tree = cKDTree(positions)
            pairs = tree.query_pairs(r=self.spatial_threshold)

            for i, j in pairs:
                u, v = track_ids[i], track_ids[j]
                self._add_spatial_edge(u, v, agent_nodes[u], agent_nodes[v])
        else:
            # 브루트 포스 방식 (O(N^2))
            for i, (u, agent_u) in enumerate(agent_nodes.items()):
                for j, (v, agent_v) in enumerate(agent_nodes.items()):
                    if i >= j:
                        continue

                    dist = np.sqrt(
                        (agent_u.x - agent_v.x)**2 +
                        (agent_u.y - agent_v.y)**2
                    )

                    if dist <= self.spatial_threshold:
                        self._add_spatial_edge(u, v, agent_u, agent_v)

    def _add_spatial_edge(
        self,
        u: int,
        v: int,
        agent_u: AgentNode,
        agent_v: AgentNode
    ):
        """공간적 엣지 추가 (상대 거리, 상대 속도 포함)"""
        dx = agent_v.x - agent_u.x
        dy = agent_v.y - agent_u.y
        dvx = agent_v.vx - agent_u.vx
        dvy = agent_v.vy - agent_u.vy

        self.graph.add_edge(
            u, v,
            type='spatial',
            distance=np.sqrt(dx**2 + dy**2),
            dx=dx,
            dy=dy,
            dvx=dvx,
            dvy=dvy
        )

    def _add_semantic_edges(
        self,
        agent_nodes: Dict[int, AgentNode],
        map_data: Optional[pd.DataFrame]
    ):
        """의미론적 엣지 추가 (교통 규칙 기반)"""
        # Conflict Edge: 경로가 겹칠 가능성이 있는 차량 간 연결
        self._add_conflict_edges(agent_nodes)

        # Yielding Edge: 양보 관계
        if map_data is not None:
            self._add_yielding_edges(agent_nodes, map_data)

        # Following Edge: 동일 차선 상의 선행-후행 관계
        if map_data is not None:
            self._add_following_edges(agent_nodes, map_data)

    def _add_conflict_edges(self, agent_nodes: Dict[int, AgentNode]):
        """Conflict Edge 추가: 서로 다른 진입로에서 진입하여 경로가 겹칠 가능성"""
        # 간단한 구현: 속도 벡터가 교차하는 차량 쌍 찾기
        for u, agent_u in agent_nodes.items():
            for v, agent_v in agent_nodes.items():
                if u >= v:
                    continue

                # 두 차량의 속도 벡터
                v1 = np.array([agent_u.vx, agent_u.vy])
                v2 = np.array([agent_v.vx, agent_v.vy])

                # 위치 벡터
                p1 = np.array([agent_u.x, agent_u.y])
                p2 = np.array([agent_v.x, agent_v.y])

                # 충돌 가능성 간단 판단: 속도 방향이 교차하고 거리가 가까움
                dist = np.linalg.norm(p1 - p2)
                if dist > self.spatial_threshold * 2:
                    continue

                # 속도 벡터의 각도 차이
                angle_diff = np.abs(
                    np.arctan2(v1[1], v1[0]) - np.arctan2(v2[1], v2[0])
                )
                angle_diff = min(angle_diff, 2 * np.pi - angle_diff)

                # 각도 차이가 30도 이상 150도 이하이면 충돌 가능성
                if 30 * np.pi / 180 < angle_diff < 150 * np.pi / 180:
                    self.graph.add_edge(
                        u, v,
                        type='conflict',
                        angle_diff=angle_diff,
                        distance=dist
                    )

    def _add_yielding_edges(
        self,
        agent_nodes: Dict[int, AgentNode],
        map_data: pd.DataFrame
    ):
        """Yielding Edge 추가: 양보 표지판이 있는 차선의 차량과 우선권을 가진 차량"""
        # 간단한 구현: 진입 차량과 회전 중인 차량 간의 관계
        # 실제로는 맵 데이터의 yield sign 정보를 활용해야 함

        for u, agent_u in agent_nodes.items():
            for v, agent_v in agent_nodes.items():
                if u >= v:
                    continue

                # 회전교차로 중심으로부터의 거리로 판단
                dist_u = np.sqrt(agent_u.x**2 + agent_u.y**2)
                dist_v = np.sqrt(agent_v.x**2 + agent_v.y**2)

                # 한 차량은 중심에 가깝고(회전 중), 다른 차량은 멀리 있음(진입 대기)
                if abs(dist_u - dist_v) > 5.0:  # 5m 이상 차이
                    if dist_u < dist_v:
                        # agent_u가 회전 중, agent_v가 진입 대기
                        self.graph.add_edge(
                            v, u,  # 진입 차량 -> 회전 차량 (양보)
                            type='yielding',
                            yielding_agent=v,
                            priority_agent=u
                        )
                    else:
                        self.graph.add_edge(
                            u, v,
                            type='yielding',
                            yielding_agent=u,
                            priority_agent=v
                        )

    def _add_following_edges(
        self,
        agent_nodes: Dict[int, AgentNode],
        map_data: pd.DataFrame
    ):
        """Following Edge 추가: 동일 차선 상의 선행-후행 관계"""
        # 간단한 구현: 속도 방향이 유사하고 거리가 가까운 차량 쌍
        for u, agent_u in agent_nodes.items():
            for v, agent_v in agent_nodes.items():
                if u >= v:
                    continue

                # 속도 벡터
                v1 = np.array([agent_u.vx, agent_u.vy])
                v2 = np.array([agent_v.vx, agent_v.vy])

                # 위치 벡터
                p1 = np.array([agent_u.x, agent_u.y])
                p2 = np.array([agent_v.x, agent_v.y])

                # 거리
                dist = np.linalg.norm(p1 - p2)
                if dist > self.spatial_threshold:
                    continue

                # 속도 방향 유사도 (내적)
                v1_norm = v1 / (np.linalg.norm(v1) + 1e-6)
                v2_norm = v2 / (np.linalg.norm(v2) + 1e-6)
                similarity = np.dot(v1_norm, v2_norm)

                # 속도 방향이 유사하고 (similarity > 0.7), 한 차량이 앞에 있음
                if similarity > 0.7:
                    # 앞뒤 관계 판단
                    direction = (p2 - p1) / (dist + 1e-6)
                    forward_projection = np.dot(direction, v1_norm)

                    if forward_projection > 0.5:  # agent_v가 agent_u의 앞
                        self.graph.add_edge(
                            u, v,
                            type='following',
                            leader=v,
                            follower=u,
                            distance=dist
                        )
                    elif forward_projection < -0.5:  # agent_u가 agent_v의 앞
                        self.graph.add_edge(
                            v, u,
                            type='following',
                            leader=u,
                            follower=v,
                            distance=dist
                        )

    def to_pytorch_geometric(self) -> Data:
        """
        NetworkX 그래프를 PyTorch Geometric Data 객체로 변환합니다.

        Returns:
            PyTorch Geometric Data 객체
        """
        if self.graph is None:
            raise ValueError("그래프가 생성되지 않았습니다. build_from_frame()을 먼저 호출하세요.")

        # 노드 특징 추출
        node_features = []
        node_mapping = {}  # 원본 노드 ID -> 인덱스 매핑

        for idx, (node_id, data) in enumerate(self.graph.nodes(data=True)):
            node_mapping[node_id] = idx
            node_features.append(data['features'])

        x = torch.tensor(np.array(node_features), dtype=torch.float)

        # 엣지 인덱스 추출
        edge_index = []
        edge_attr = []

        for u, v, data in self.graph.edges(data=True):
            u_idx = node_mapping[u]
            v_idx = node_mapping[v]
            edge_index.append([u_idx, v_idx])

            # 엣지 속성 (거리, 상대 속도 등)
            edge_attr_vec = [
                data.get('distance', 0.0),
                data.get('dx', 0.0),
                data.get('dy', 0.0),
                data.get('dvx', 0.0),
                data.get('dvy', 0.0)
            ]
            edge_attr.append(edge_attr_vec)

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        # 엣지 타입 정보
        edge_type = [data.get('type', 'unknown') for _, _, data in self.graph.edges(data=True)]

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_type=edge_type,
            num_nodes=x.size(0)
        )

    def get_statistics(self) -> Dict:
        """그래프 통계 정보 반환"""
        if self.graph is None:
            return {}

        edge_types = {}
        for _, _, data in self.graph.edges(data=True):
            edge_type = data.get('type', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'edge_types': edge_types,
            'is_connected': nx.is_weakly_connected(self.graph.to_undirected())
        }


def main():
    """테스트용 메인 함수"""
    # 예시 데이터 생성
    frame_data = pd.DataFrame({
        'track_id': [1, 2, 3],
        'x': [0.0, 5.0, 10.0],
        'y': [0.0, 0.0, 0.0],
        'vx': [5.0, 5.0, 5.0],
        'vy': [0.0, 0.0, 0.0],
        'psi_rad': [0.0, 0.0, 0.0],
        'width': [2.0, 2.0, 2.0],
        'length': [4.5, 4.5, 4.5]
    })

    builder = SceneGraphBuilder(spatial_threshold=15.0)
    graph = builder.build_from_frame(frame_data)

    print("그래프 통계:")
    stats = builder.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # PyTorch Geometric 변환
    data = builder.to_pytorch_geometric()
    print(f"\nPyTorch Geometric Data:")
    print(f"  노드 수: {data.num_nodes}")
    print(f"  엣지 수: {data.edge_index.size(1)}")
    print(f"  노드 특징 차원: {data.x.size(1)}")
    print(f"  엣지 특징 차원: {data.edge_attr.size(1)}")


if __name__ == "__main__":
    main()

