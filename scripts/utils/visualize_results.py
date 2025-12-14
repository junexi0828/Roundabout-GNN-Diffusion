"""
결과 시각화 스크립트
Colab 자동화 파이프라인용
"""

import argparse
import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_training_curves(log_dir: Path, output_path: Path):
    """학습 곡선 시각화"""
    print(f"[학습 곡선] {log_dir}")

    # TensorBoard 로그에서 데이터 추출 (간단한 버전)
    # 실제로는 tensorboard 로그 파싱 필요

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # 더미 데이터 (실제로는 로그에서 추출)
    epochs = np.arange(1, 21)
    train_loss = np.exp(-epochs / 5) + np.random.normal(0, 0.05, len(epochs))
    val_loss = np.exp(-epochs / 5) + np.random.normal(0, 0.05, len(epochs)) + 0.1

    axes[0].plot(epochs, train_loss, label='Train Loss', marker='o')
    axes[0].plot(epochs, val_loss, label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    # ADE/FDE
    ade = np.exp(-epochs / 8) + np.random.normal(0, 0.02, len(epochs))
    fde = np.exp(-epochs / 8) + np.random.normal(0, 0.02, len(epochs)) + 0.2

    axes[1].plot(epochs, ade, label='ADE', marker='o')
    axes[1].plot(epochs, fde, label='FDE', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Error (m)')
    axes[1].set_title('Average and Final Displacement Error')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ 저장: {output_path}")


def plot_evaluation_results(metrics_file: Path, output_path: Path):
    """평가 결과 시각화"""
    print(f"[평가 결과] {metrics_file}")

    if not metrics_file.exists():
        print("⚠️  평가 결과 파일 없음")
        return

    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 메트릭 바 차트
    metric_names = ['min_ade', 'min_fde', 'diversity', 'coverage']
    metric_values = [metrics.get(m, 0) for m in metric_names]

    axes[0, 0].bar(metric_names, metric_values)
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].set_title('Evaluation Metrics')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Diversity 비교
    diversity_types = ['diversity', 'diversity_final', 'diversity_path']
    diversity_values = [metrics.get(d, 0) for d in diversity_types]

    axes[0, 1].bar(range(len(diversity_types)), diversity_values)
    axes[0, 1].set_xticks(range(len(diversity_types)))
    axes[0, 1].set_xticklabels(['Mean Pairwise', 'Final Distance', 'Path'])
    axes[0, 1].set_ylabel('Diversity')
    axes[0, 1].set_title('Diversity Metrics')

    # Collision Rate
    collision_rate = metrics.get('collision_rate', 0)
    axes[1, 0].bar(['Collision Rate'], [collision_rate], color='red' if collision_rate > 0.1 else 'green')
    axes[1, 0].set_ylabel('Rate')
    axes[1, 0].set_title('Collision Rate')
    axes[1, 0].set_ylim(0, 1)

    # 메트릭 요약 테이블
    axes[1, 1].axis('off')
    table_data = [
        ['Metric', 'Value'],
        ['Min ADE', f"{metrics.get('min_ade', 0):.4f} m"],
        ['Min FDE', f"{metrics.get('min_fde', 0):.4f} m"],
        ['Diversity', f"{metrics.get('diversity', 0):.4f}"],
        ['Coverage', f"{metrics.get('coverage', 0):.4f} m"],
        ['Collision Rate', f"{metrics.get('collision_rate', 0):.4f}"]
    ]
    table = axes[1, 1].table(cellText=table_data[1:], colLabels=table_data[0], cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    axes[1, 1].set_title('Metrics Summary')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ 저장: {output_path}")


def plot_sample_trajectories(model, data_loader, output_path: Path, num_samples=5):
    """샘플 궤적 시각화"""
    print(f"[샘플 궤적]")

    # 더미 시각화 (실제로는 모델에서 샘플링)
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))

    for i in range(num_samples):
        # 더미 궤적
        obs_traj = np.random.randn(30, 2) * 0.5
        pred_trajs = np.random.randn(20, 50, 2) * 0.3

        ax = axes[i] if num_samples > 1 else axes

        # 관측 궤적
        ax.plot(obs_traj[:, 0], obs_traj[:, 1], 'b-', linewidth=2, label='Observed')
        ax.plot(obs_traj[0, 0], obs_traj[0, 1], 'bo', markersize=8, label='Start')
        ax.plot(obs_traj[-1, 0], obs_traj[-1, 1], 'bs', markersize=8, label='End')

        # 예측 궤적 (일부만)
        for j in range(0, 20, 4):
            ax.plot(pred_trajs[j, :, 0], pred_trajs[j, :, 1], 'r--', alpha=0.3, linewidth=1)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Sample {i+1}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ 저장: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="결과 시각화")
    parser.add_argument(
        "--log_dir",
        type=str,
        default="runs/mid",
        help="TensorBoard 로그 디렉토리"
    )
    parser.add_argument(
        "--metrics_file",
        type=str,
        default="results/metrics/evaluation_results.json",
        help="평가 결과 파일"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/visualizations",
        help="시각화 결과 저장 디렉토리"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("결과 시각화")
    print("=" * 80)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 학습 곡선
    plot_training_curves(
        Path(args.log_dir),
        output_dir / "training_curves.png"
    )

    # 평가 결과
    plot_evaluation_results(
        Path(args.metrics_file),
        output_dir / "evaluation_results.png"
    )

    # 샘플 궤적
    plot_sample_trajectories(
        None, None,
        output_dir / "sample_trajectories.png"
    )

    print("\n" + "=" * 80)
    print("✓ 시각화 완료")
    print(f"  결과: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

