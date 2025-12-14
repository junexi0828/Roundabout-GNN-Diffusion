"""
어텐션 가중치 시각화 모듈
GAT 레이어의 어텐션 가중치를 추출하고 회전교차로 BEV 맵 위에 시각화
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
from typing import Dict, List, Optional, Tuple, Union
import torch
from pathlib import Path
import pandas as pd
from scipy.spatial import cKDTree

# from ..scene_graph.scene_graph_builder import SceneGraphBuilder, AgentNode


class AttentionVisualizer:
    """어텐션 가중치 시각화 클래스"""

    def __init__(self, figsize: Tuple[int, int] = (12, 12), dpi: int = 300):
        """
        Args:
            figsize: 그림 크기
            dpi: 해상도
        """
        self.figsize = figsize
        self.dpi = dpi

    def extract_attention_weights(
        self,
        model: torch.nn.Module,
        graph_data: torch_geometric.data.Data,
        layer_idx: int = 0,
    ) -> Optional[np.ndarray]:
        """
        모델에서 어텐션 가중치를 추출합니다.

        Args:
            model: GAT 모델
            graph_data: 그래프 데이터
            layer_idx: 어텐션 레이어 인덱스

        Returns:
            어텐션 가중치 행렬 [num_nodes, num_nodes]
        """
        # 모델을 eval 모드로 설정
        model.eval()

        # Forward hook을 사용하여 어텐션 가중치 추출
        attention_weights = None

        def attention_hook(module, input, output):
            nonlocal attention_weights
            # GAT 레이어의 어텐션 가중치 추출
            if hasattr(module, "alpha"):
                # alpha는 [num_edges, num_heads] 형태
                alpha = module.alpha
                # 단일 헤드인 경우
                if alpha.dim() == 1:
                    attention_weights = alpha.detach().cpu().numpy()
                else:
                    # 멀티 헤드인 경우 평균
                    attention_weights = alpha.mean(dim=1).detach().cpu().numpy()

        # GAT 레이어에 hook 등록
        hooks = []
        for name, module in model.named_modules():
            if "gat" in name.lower() or "attention" in name.lower():
                hook = module.register_forward_hook(attention_hook)
                hooks.append(hook)

        # Forward pass
        with torch.no_grad():
            _ = model(graph_data.x, graph_data.edge_index)

        # Hook 제거
        for hook in hooks:
            hook.remove()

        # 엣지 기반 어텐션을 노드 쌍 행렬로 변환
        if attention_weights is not None and graph_data.edge_index is not None:
            num_nodes = graph_data.x.size(0)
            attention_matrix = np.zeros((num_nodes, num_nodes))

            edge_index = graph_data.edge_index.cpu().numpy()
            for idx, (src, dst) in enumerate(edge_index.T):
                attention_matrix[src, dst] = attention_weights[idx]

            return attention_matrix

        return None

    def compute_distance_based_attention(
        self, positions: np.ndarray, threshold: float = 20.0, decay_factor: float = 0.1
    ) -> np.ndarray:
        """
        거리 기반 어텐션 가중치 계산

        Args:
            positions: 노드 위치 [num_nodes, 2]
            threshold: 최대 거리 임계값 (m)
            decay_factor: 거리 감쇠 계수

        Returns:
            어텐션 행렬 [num_nodes, num_nodes]
        """
        num_nodes = positions.shape[0]
        attention_matrix = np.zeros((num_nodes, num_nodes))

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    attention_matrix[i, j] = 1.0
                else:
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist <= threshold:
                        # 거리가 가까울수록 높은 어텐션
                        attention_matrix[i, j] = np.exp(-decay_factor * dist)
                    else:
                        attention_matrix[i, j] = 0.0

        # 정규화
        row_sums = attention_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # 0으로 나누기 방지
        attention_matrix = attention_matrix / row_sums

        return attention_matrix

    def compute_velocity_based_attention(
        self, positions: np.ndarray, velocities: np.ndarray, threshold: float = 20.0
    ) -> np.ndarray:
        """
        속도 기반 어텐션 가중치 계산

        충돌 위험이 높은 차량 쌍에 높은 어텐션 부여

        Args:
            positions: 노드 위치 [num_nodes, 2]
            velocities: 노드 속도 [num_nodes, 2]
            threshold: 최대 거리 임계값 (m)

        Returns:
            어텐션 행렬 [num_nodes, num_nodes]
        """
        num_nodes = positions.shape[0]
        attention_matrix = np.zeros((num_nodes, num_nodes))

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    attention_matrix[i, j] = 1.0
                else:
                    # 상대 위치와 속도
                    delta_p = positions[j] - positions[i]
                    delta_v = velocities[j] - velocities[i]

                    dist = np.linalg.norm(delta_p)

                    if dist <= threshold:
                        # 상대 속도가 가까워지는 방향이면 높은 어텐션
                        # 내적이 음수면 접근 중
                        approach_rate = -np.dot(delta_p, delta_v) / (dist + 1e-6)

                        if approach_rate > 0:
                            # 접근 중이고 거리가 가까울수록 높은 어텐션
                            attention_matrix[i, j] = approach_rate / (dist + 1.0)
                        else:
                            # 멀어지는 중
                            attention_matrix[i, j] = 0.1 / (dist + 1.0)
                    else:
                        attention_matrix[i, j] = 0.0

        # 정규화
        row_sums = attention_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        attention_matrix = attention_matrix / row_sums

        return attention_matrix

    def visualize_attention_heatmap(
        self,
        attention_matrix: np.ndarray,
        node_labels: Optional[List[str]] = None,
        output_path: Optional[Path] = None,
        title: str = "Attention Weights Heatmap",
    ):
        """
        어텐션 가중치 히트맵 시각화

        Args:
            attention_matrix: 어텐션 행렬 [num_nodes, num_nodes]
            node_labels: 노드 레이블 리스트
            output_path: 저장 경로
            title: 제목
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        im = ax.imshow(
            attention_matrix, cmap="YlOrRd", aspect="auto", interpolation="nearest"
        )

        # 컬러바
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Attention Weight", rotation=270, labelpad=20)

        # 레이블 설정
        if node_labels is None:
            node_labels = [f"Node {i}" for i in range(attention_matrix.shape[0])]

        ax.set_xticks(range(len(node_labels)))
        ax.set_yticks(range(len(node_labels)))
        ax.set_xticklabels(node_labels, rotation=45, ha="right")
        ax.set_yticklabels(node_labels)

        ax.set_xlabel("Target Node")
        ax.set_ylabel("Source Node")
        ax.set_title(title)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
            print(f"히트맵 저장: {output_path}")
        else:
            plt.show()

        plt.close()

    def visualize_attention_on_map(
        self,
        positions: np.ndarray,
        attention_matrix: np.ndarray,
        velocities: Optional[np.ndarray] = None,
        map_image: Optional[np.ndarray] = None,
        output_path: Optional[Path] = None,
        title: str = "Attention Weights on Roundabout Map",
        min_attention: float = 0.1,
        max_edge_width: float = 5.0,
    ):
        """
        회전교차로 맵 위에 어텐션 가중치를 그래프로 오버레이

        Args:
            positions: 노드 위치 [num_nodes, 2]
            attention_matrix: 어텐션 행렬 [num_nodes, num_nodes]
            velocities: 노드 속도 [num_nodes, 2] (선택사항, 화살표 표시용)
            map_image: 맵 이미지 (선택사항)
            output_path: 저장 경로
            title: 제목
            min_attention: 표시할 최소 어텐션 값
            max_edge_width: 최대 엣지 두께
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # 맵 이미지 표시 (있는 경우)
        if map_image is not None:
            ax.imshow(map_image, extent=[-50, 50, -50, 50], alpha=0.3, origin="lower")
        else:
            # 기본 회전교차로 원형 표시
            circle = plt.Circle((0, 0), 15, color="lightgray", fill=True, alpha=0.3)
            ax.add_patch(circle)
            ax.set_xlim(-30, 30)
            ax.set_ylim(-30, 30)

        num_nodes = positions.shape[0]

        # 엣지 그리기 (어텐션 가중치에 비례)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and attention_matrix[i, j] >= min_attention:
                    # 어텐션 가중치에 비례한 엣지 두께 및 투명도
                    attention = attention_matrix[i, j]
                    edge_width = (attention / attention_matrix.max()) * max_edge_width
                    edge_alpha = min(attention * 2, 1.0)

                    # 색상: 어텐션이 높을수록 빨간색
                    color_intensity = attention / attention_matrix.max()
                    color = plt.cm.Reds(color_intensity)

                    ax.plot(
                        [positions[i, 0], positions[j, 0]],
                        [positions[i, 1], positions[j, 1]],
                        color=color,
                        linewidth=edge_width,
                        alpha=edge_alpha,
                        zorder=1,
                    )

        # 노드 표시
        for i in range(num_nodes):
            # 노드 위치
            ax.scatter(
                positions[i, 0],
                positions[i, 1],
                s=200,
                c="blue",
                edgecolors="black",
                linewidths=2,
                zorder=3,
                label="Vehicle" if i == 0 else "",
            )

            # 노드 번호
            ax.annotate(
                f"{i}",
                (positions[i, 0], positions[i, 1]),
                fontsize=10,
                ha="center",
                va="center",
                color="white",
                weight="bold",
                zorder=4,
            )

            # 속도 벡터 표시 (있는 경우)
            if velocities is not None:
                ax.arrow(
                    positions[i, 0],
                    positions[i, 1],
                    velocities[i, 0] * 0.5,
                    velocities[i, 1] * 0.5,
                    head_width=1.0,
                    head_length=0.8,
                    fc="green",
                    ec="green",
                    alpha=0.7,
                    zorder=2,
                )

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
            print(f"맵 시각화 저장: {output_path}")
        else:
            plt.show()

        plt.close()

    def visualize_yielding_situation(
        self,
        positions: np.ndarray,
        attention_matrix: np.ndarray,
        yielding_pairs: List[Tuple[int, int]],
        velocities: Optional[np.ndarray] = None,
        output_path: Optional[Path] = None,
    ):
        """
        양보 상황 특화 시각화

        Args:
            positions: 노드 위치
            attention_matrix: 어텐션 행렬
            yielding_pairs: 양보 관계 리스트 [(yielding_agent, priority_agent), ...]
            velocities: 노드 속도
            output_path: 저장 경로
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # 회전교차로 배경
        circle = plt.Circle((0, 0), 15, color="lightgray", fill=True, alpha=0.3)
        ax.add_patch(circle)
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)

        num_nodes = positions.shape[0]

        # 일반 엣지 (회색, 얇게)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and attention_matrix[i, j] > 0.1:
                    if (i, j) not in yielding_pairs and (j, i) not in yielding_pairs:
                        ax.plot(
                            [positions[i, 0], positions[j, 0]],
                            [positions[i, 1], positions[j, 1]],
                            color="gray",
                            linewidth=0.5,
                            alpha=0.3,
                            zorder=1,
                        )

        # 양보 관계 엣지 (빨간색, 굵게)
        for yielding_agent, priority_agent in yielding_pairs:
            attention = attention_matrix[yielding_agent, priority_agent]

            ax.plot(
                [positions[yielding_agent, 0], positions[priority_agent, 0]],
                [positions[yielding_agent, 1], positions[priority_agent, 1]],
                color="red",
                linewidth=attention * 5,
                alpha=0.8,
                zorder=2,
                label=(
                    "Yielding Relationship"
                    if yielding_pairs.index((yielding_agent, priority_agent)) == 0
                    else ""
                ),
            )

            # 화살표 표시 (양보 방향)
            mid_x = (positions[yielding_agent, 0] + positions[priority_agent, 0]) / 2
            mid_y = (positions[yielding_agent, 1] + positions[priority_agent, 1]) / 2

            dx = positions[priority_agent, 0] - positions[yielding_agent, 0]
            dy = positions[priority_agent, 1] - positions[yielding_agent, 1]
            length = np.sqrt(dx**2 + dy**2)
            dx_norm = dx / length
            dy_norm = dy / length

            ax.arrow(
                mid_x - dx_norm * 2,
                mid_y - dy_norm * 2,
                dx_norm * 2,
                dy_norm * 2,
                head_width=1.5,
                head_length=1.2,
                fc="red",
                ec="red",
                alpha=0.8,
                zorder=3,
            )

        # 노드 표시
        for i in range(num_nodes):
            # 양보 차량은 노란색, 우선권 차량은 빨간색
            is_yielding = any(i == y for y, _ in yielding_pairs)
            is_priority = any(i == p for _, p in yielding_pairs)

            if is_yielding:
                color = "yellow"
                edgecolor = "orange"
            elif is_priority:
                color = "red"
                edgecolor = "darkred"
            else:
                color = "blue"
                edgecolor = "black"

            ax.scatter(
                positions[i, 0],
                positions[i, 1],
                s=300,
                c=color,
                edgecolors=edgecolor,
                linewidths=3,
                zorder=4,
            )

            ax.annotate(
                f"{i}",
                (positions[i, 0], positions[i, 1]),
                fontsize=12,
                ha="center",
                va="center",
                color="black",
                weight="bold",
                zorder=5,
            )

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Yielding Situation Visualization")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
            print(f"양보 상황 시각화 저장: {output_path}")
        else:
            plt.show()

        plt.close()

    def create_attention_report(
        self,
        attention_matrix: np.ndarray,
        positions: np.ndarray,
        velocities: Optional[np.ndarray] = None,
        output_dir: Path = Path("results"),
    ):
        """
        어텐션 가중치 종합 리포트 생성

        Args:
            attention_matrix: 어텐션 행렬
            positions: 노드 위치
            velocities: 노드 속도
            output_dir: 출력 디렉토리
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. 히트맵
        self.visualize_attention_heatmap(
            attention_matrix,
            output_path=output_dir / "attention_heatmap.png",
            title="Attention Weights Heatmap",
        )

        # 2. 맵 위 시각화
        self.visualize_attention_on_map(
            positions,
            attention_matrix,
            velocities,
            output_path=output_dir / "attention_on_map.png",
            title="Attention Weights on Roundabout Map",
        )

        # 3. 통계 정보 저장
        stats = {
            "mean_attention": float(np.mean(attention_matrix)),
            "max_attention": float(np.max(attention_matrix)),
            "min_attention": float(np.min(attention_matrix[attention_matrix > 0])),
            "num_edges": int(np.sum(attention_matrix > 0.1)),
            "top_attention_pairs": [],
        }

        # 상위 어텐션 쌍 찾기
        flat_indices = np.argsort(attention_matrix.flatten())[::-1]
        for idx in flat_indices[:10]:
            i, j = np.unravel_index(idx, attention_matrix.shape)
            if i != j and attention_matrix[i, j] > 0:
                stats["top_attention_pairs"].append(
                    {
                        "source": int(i),
                        "target": int(j),
                        "attention": float(attention_matrix[i, j]),
                        "distance": float(np.linalg.norm(positions[i] - positions[j])),
                    }
                )

        # 통계 저장
        import json

        with open(output_dir / "attention_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        print(f"\n어텐션 리포트 생성 완료:")
        print(f"  평균 어텐션: {stats['mean_attention']:.4f}")
        print(f"  최대 어텐션: {stats['max_attention']:.4f}")
        print(f"  엣지 수: {stats['num_edges']}")
        print(f"  상위 어텐션 쌍: {len(stats['top_attention_pairs'])}개")


def main():
    """테스트용 메인 함수"""
    # 더미 데이터 생성
    num_nodes = 5
    positions = np.random.randn(num_nodes, 2) * 10
    velocities = np.random.randn(num_nodes, 2) * 2

    visualizer = AttentionVisualizer()

    # 거리 기반 어텐션 계산
    dist_attention = visualizer.compute_distance_based_attention(positions)

    # 속도 기반 어텐션 계산
    vel_attention = visualizer.compute_velocity_based_attention(positions, velocities)

    # 시각화
    visualizer.visualize_attention_heatmap(
        dist_attention, output_path=Path("results/test_heatmap.png")
    )

    visualizer.visualize_attention_on_map(
        positions, dist_attention, velocities, output_path=Path("results/test_map.png")
    )

    print("시각화 완료!")


if __name__ == "__main__":
    main()
