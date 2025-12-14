"""
어텐션 가중치 분석 및 통계
테스트 결과를 기반으로 한 어텐션 분석 도구
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

from .attention_visualizer import AttentionVisualizer


class AttentionAnalyzer:
    """어텐션 가중치 분석 클래스"""

    def __init__(self):
        self.visualizer = AttentionVisualizer()

    def load_attention_results(self, npz_path: Path) -> Dict:
        """
        저장된 어텐션 결과 로드

        Args:
            npz_path: .npz 파일 경로

        Returns:
            로드된 데이터 딕셔너리
        """
        data = np.load(npz_path, allow_pickle=True)
        return dict(data)

    def analyze_attention_statistics(
        self,
        attention_matrix: np.ndarray,
        positions: Optional[np.ndarray] = None
    ) -> Dict:
        """
        어텐션 통계 분석

        Args:
            attention_matrix: 어텐션 행렬
            positions: 노드 위치 (거리 분석용)

        Returns:
            통계 딕셔너리
        """
        stats = {
            'mean': float(np.mean(attention_matrix)),
            'std': float(np.std(attention_matrix)),
            'max': float(np.max(attention_matrix)),
            'min': float(np.min(attention_matrix[attention_matrix > 0])),
            'median': float(np.median(attention_matrix[attention_matrix > 0])),
            'num_edges': int(np.sum(attention_matrix > 0.1)),
            'sparsity': float(1.0 - np.sum(attention_matrix > 0.1) / (attention_matrix.size - attention_matrix.shape[0]))
        }

        # 거리 기반 분석 (위치가 있는 경우)
        if positions is not None:
            distances = []
            attentions = []

            for i in range(attention_matrix.shape[0]):
                for j in range(attention_matrix.shape[1]):
                    if i != j and attention_matrix[i, j] > 0:
                        dist = np.linalg.norm(positions[i] - positions[j])
                        distances.append(dist)
                        attentions.append(attention_matrix[i, j])

            if distances:
                stats['distance_analysis'] = {
                    'mean_distance': float(np.mean(distances)),
                    'mean_attention_by_distance': float(np.mean(attentions)),
                    'correlation': float(np.corrcoef(distances, attentions)[0, 1]) if len(distances) > 1 else 0.0
                }

        return stats

    def find_top_attention_pairs(
        self,
        attention_matrix: np.ndarray,
        positions: Optional[np.ndarray] = None,
        top_k: int = 10
    ) -> List[Dict]:
        """
        상위 어텐션 쌍 찾기

        Args:
            attention_matrix: 어텐션 행렬
            positions: 노드 위치
            top_k: 상위 k개

        Returns:
            상위 어텐션 쌍 리스트
        """
        # 대각선 제외하고 플래튼
        mask = ~np.eye(attention_matrix.shape[0], dtype=bool)
        flat_indices = np.argsort(attention_matrix[mask])[::-1]

        top_pairs = []
        row_indices, col_indices = np.where(mask)

        for idx in flat_indices[:top_k]:
            i = row_indices[idx]
            j = col_indices[idx]
            attention = attention_matrix[i, j]

            pair_info = {
                'source': int(i),
                'target': int(j),
                'attention': float(attention)
            }

            if positions is not None:
                pair_info['distance'] = float(np.linalg.norm(positions[i] - positions[j]))

            top_pairs.append(pair_info)

        return top_pairs

    def compare_attention_types(
        self,
        distance_attention: np.ndarray,
        velocity_attention: np.ndarray,
        output_path: Optional[Path] = None
    ):
        """
        거리 기반과 속도 기반 어텐션 비교

        Args:
            distance_attention: 거리 기반 어텐션
            velocity_attention: 속도 기반 어텐션
            output_path: 저장 경로
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 거리 기반
        im1 = axes[0].imshow(distance_attention, cmap='YlOrRd', aspect='auto')
        axes[0].set_title('Distance-based Attention')
        axes[0].set_xlabel('Target Node')
        axes[0].set_ylabel('Source Node')
        plt.colorbar(im1, ax=axes[0])

        # 속도 기반
        im2 = axes[1].imshow(velocity_attention, cmap='YlOrRd', aspect='auto')
        axes[1].set_title('Velocity-based Attention')
        axes[1].set_xlabel('Target Node')
        axes[1].set_ylabel('Source Node')
        plt.colorbar(im2, ax=axes[1])

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"비교 시각화 저장: {output_path}")
        else:
            plt.show()

        plt.close()

    def create_comprehensive_report(
        self,
        attention_results_path: Path,
        output_dir: Path = Path('results')
    ):
        """
        종합 리포트 생성 (테스트 결과 기반)

        Args:
            attention_results_path: .npz 파일 경로
            output_dir: 출력 디렉토리
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 데이터 로드
        data = self.load_attention_results(attention_results_path)

        distance_attention = data.get('distance_attention')
        velocity_attention = data.get('velocity_attention')
        positions = data.get('positions')
        velocities = data.get('velocities')

        print("=" * 80)
        print("어텐션 가중치 종합 분석 리포트")
        print("=" * 80)

        # 거리 기반 어텐션 분석
        if distance_attention is not None:
            print("\n[거리 기반 어텐션 분석]")
            dist_stats = self.analyze_attention_statistics(distance_attention, positions)
            print(f"  평균: {dist_stats['mean']:.4f}")
            print(f"  최대: {dist_stats['max']:.4f}")
            print(f"  엣지 수: {dist_stats['num_edges']}")
            print(f"  희소성: {dist_stats['sparsity']:.2%}")

            top_pairs = self.find_top_attention_pairs(distance_attention, positions, top_k=5)
            print(f"\n  상위 어텐션 쌍:")
            for pair in top_pairs:
                print(f"    노드 {pair['source']} → 노드 {pair['target']}: "
                      f"어텐션={pair['attention']:.4f}, 거리={pair.get('distance', 'N/A'):.2f}m")

        # 속도 기반 어텐션 분석
        if velocity_attention is not None:
            print("\n[속도 기반 어텐션 분석]")
            vel_stats = self.analyze_attention_statistics(velocity_attention, positions)
            print(f"  평균: {vel_stats['mean']:.4f}")
            print(f"  최대: {vel_stats['max']:.4f}")
            print(f"  엣지 수: {vel_stats['num_edges']}")

        # 비교 시각화
        if distance_attention is not None and velocity_attention is not None:
            self.compare_attention_types(
                distance_attention,
                velocity_attention,
                output_path=output_dir / 'attention_comparison.png'
            )

        # 맵 위 시각화
        if positions is not None and distance_attention is not None:
            self.visualizer.visualize_attention_on_map(
                positions,
                distance_attention,
                velocities,
                output_path=output_dir / 'attention_map_visualization.png',
                title='Distance-based Attention on Roundabout'
            )

        print(f"\n리포트 생성 완료: {output_dir}")


def main():
    """테스트용 메인 함수"""
    # 테스트 결과 파일 경로
    results_path = Path('results/attention_test_results.npz')

    if results_path.exists():
        analyzer = AttentionAnalyzer()
        analyzer.create_comprehensive_report(results_path)
    else:
        print(f"결과 파일을 찾을 수 없습니다: {results_path}")
        print("테스트를 먼저 실행하세요.")


if __name__ == "__main__":
    main()

