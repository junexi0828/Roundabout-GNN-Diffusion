"""
전체 평가 파이프라인
모델 예측 결과를 평가하고 리포트 생성
"""

import numpy as np
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from .metrics import (
    TrajectoryEvaluator,
    calculate_ade,
    calculate_fde,
    calculate_miss_rate,
    print_metrics,
    save_metrics
)


class ModelEvaluator:
    """모델 평가 클래스"""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        vehicle_radius: float = 2.5,
        miss_threshold: float = 2.0
    ):
        """
        Args:
            model: 평가할 모델
            device: 디바이스 (CPU/GPU)
            vehicle_radius: 차량 반경
            miss_threshold: Miss Rate 임계값
        """
        self.model = model.to(device)
        self.device = device
        self.evaluator = TrajectoryEvaluator(vehicle_radius, miss_threshold)
        self.results = []

    def evaluate_dataset(
        self,
        data_loader: torch.utils.data.DataLoader,
        max_batches: Optional[int] = None
    ) -> Dict[str, float]:
        """
        데이터셋 전체를 평가합니다.

        Args:
            data_loader: 데이터 로더
            max_batches: 최대 배치 수 (None이면 전체)

        Returns:
            평균 평가 지표
        """
        self.model.eval()

        all_predictions = []
        all_ground_truths = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc="평가 중")):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                # 데이터를 디바이스로 이동
                obs_data = batch['obs_data'].to(self.device)
                pred_target = batch['pred_data'].to(self.device)

                # 예측
                if 'graph' in batch:
                    graph = batch['graph'].to(self.device)
                    pred = self.model(graph.x, graph.edge_index, graph.edge_weight)
                else:
                    pred = self.model(obs_data)

                # CPU로 이동 및 numpy 변환
                pred_np = pred.cpu().numpy()
                gt_np = pred_target.cpu().numpy()

                all_predictions.append(pred_np)
                all_ground_truths.append(gt_np)

        # 전체 예측과 타겟 결합
        if all_predictions:
            all_pred = np.concatenate(all_predictions, axis=0)
            all_gt = np.concatenate(all_ground_truths, axis=0)
        else:
            return {}

        # 평가 지표 계산
        metrics = self.evaluator.evaluate(all_pred, all_gt)

        return metrics

    def evaluate_scenario(
        self,
        scenario_data: pd.DataFrame,
        scenario_name: str,
        output_dir: Optional[Path] = None
    ) -> Dict[str, float]:
        """
        특정 시나리오를 평가합니다.

        Args:
            scenario_data: 시나리오 데이터
            scenario_name: 시나리오 이름
            output_dir: 결과 저장 디렉토리

        Returns:
            평가 지표
        """
        # 실제 구현 시 시나리오 데이터를 모델 입력으로 변환
        # 여기서는 예시로 더미 평가 수행

        print(f"\n시나리오 평가: {scenario_name}")

        # 더미 예측 및 타겟 생성 (실제로는 모델 예측 사용)
        num_samples = 100
        num_timesteps = 50

        predicted = np.random.randn(num_samples, num_timesteps, 2) * 0.5
        ground_truth = predicted + np.random.randn(num_samples, num_timesteps, 2) * 0.1

        # 평가
        metrics = self.evaluator.evaluate(predicted, ground_truth)
        metrics['scenario'] = scenario_name

        # 결과 저장
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            save_metrics(metrics, output_dir / 'metrics.csv', scenario_name)

            # 시각화
            self._visualize_scenario_results(
                predicted, ground_truth, metrics, output_dir, scenario_name
            )

        return metrics

    def _visualize_scenario_results(
        self,
        predicted: np.ndarray,
        ground_truth: np.ndarray,
        metrics: Dict[str, float],
        output_dir: Path,
        scenario_name: str
    ):
        """시나리오 결과 시각화"""
        sns.set_style("whitegrid")

        # 샘플 궤적 시각화
        num_samples = min(5, predicted.shape[0])

        fig, axes = plt.subplots(1, num_samples, figsize=(5*num_samples, 5))
        if num_samples == 1:
            axes = [axes]

        for i in range(num_samples):
            ax = axes[i]

            # 실제 궤적
            ax.plot(
                ground_truth[i, :, 0],
                ground_truth[i, :, 1],
                'b-', label='실제', linewidth=2
            )
            ax.scatter(
                ground_truth[i, 0, 0],
                ground_truth[i, 0, 1],
                c='green', s=100, marker='o', label='시작점'
            )
            ax.scatter(
                ground_truth[i, -1, 0],
                ground_truth[i, -1, 1],
                c='red', s=100, marker='s', label='종료점'
            )

            # 예측 궤적
            ax.plot(
                predicted[i, :, 0],
                predicted[i, :, 1],
                'r--', label='예측', linewidth=2, alpha=0.7
            )

            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title(f'샘플 {i+1}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')

        plt.suptitle(f'시나리오: {scenario_name}', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / f'{scenario_name}_trajectories.png', dpi=300)
        plt.close()

        # 오차 분포 히스토그램
        errors = []
        for i in range(predicted.shape[0]):
            diff = predicted[i] - ground_truth[i]
            distances = np.linalg.norm(diff, axis=1)
            errors.extend(distances.tolist())

        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('오차 (m)')
        plt.ylabel('빈도')
        plt.title(f'오차 분포 - {scenario_name}')
        plt.axvline(x=metrics['ADE'], color='r', linestyle='--', label=f"ADE: {metrics['ADE']:.3f}m")
        plt.axvline(x=metrics['FDE'], color='g', linestyle='--', label=f"FDE: {metrics['FDE']:.3f}m")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f'{scenario_name}_error_distribution.png', dpi=300)
        plt.close()

    def compare_scenarios(
        self,
        scenario_results: Dict[str, Dict[str, float]],
        output_path: Path
    ):
        """여러 시나리오 결과 비교"""
        # DataFrame 생성
        df = pd.DataFrame(scenario_results).T

        # 시각화
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # ADE 비교
        ax = axes[0, 0]
        df['ADE'].plot(kind='bar', ax=ax, color='skyblue')
        ax.set_ylabel('ADE (m)')
        ax.set_title('Average Displacement Error')
        ax.tick_params(axis='x', rotation=45)

        # FDE 비교
        ax = axes[0, 1]
        df['FDE'].plot(kind='bar', ax=ax, color='lightcoral')
        ax.set_ylabel('FDE (m)')
        ax.set_title('Final Displacement Error')
        ax.tick_params(axis='x', rotation=45)

        # Miss Rate 비교
        ax = axes[1, 0]
        df['Miss_Rate'].plot(kind='bar', ax=ax, color='lightgreen')
        ax.set_ylabel('Miss Rate')
        ax.set_title('Miss Rate')
        ax.tick_params(axis='x', rotation=45)

        # Collision Rate 비교
        ax = axes[1, 1]
        if 'Collision_Rate' in df.columns:
            df['Collision_Rate'].plot(kind='bar', ax=ax, color='orange')
            ax.set_ylabel('Collision Rate')
            ax.set_title('Collision Rate')
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

        # CSV 저장
        df.to_csv(output_path.with_suffix('.csv'))
        print(f"비교 결과 저장: {output_path}")


def main():
    """테스트용 메인 함수"""
    # 더미 모델 생성
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return torch.randn(x.shape[0], 50, 2) * 0.5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DummyModel()

    evaluator = ModelEvaluator(model, device)

    # 더미 데이터 로더 (실제 구현 필요)
    print("평가 시스템 준비 완료")
    print("실제 데이터 로더와 함께 사용하면 평가를 수행할 수 있습니다.")


if __name__ == "__main__":
    main()

