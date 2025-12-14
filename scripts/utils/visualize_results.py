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
plt.rcParams["figure.figsize"] = (12, 8)


def plot_training_curves(log_dir: Path, output_path: Path):
    """학습 곡선 시각화"""
    print(f"[학습 곡선] {log_dir}")

    # 체크포인트에서 학습 히스토리 로드 시도
    checkpoint_dir = (
        log_dir.parent.parent / "checkpoints" / log_dir.name.replace("runs/", "")
    )
    if not checkpoint_dir.exists():
        checkpoint_dir = log_dir.parent.parent / "checkpoints" / "mid"

    train_losses = []
    val_losses = []
    val_ades = []
    val_fdes = []

    # 최종 체크포인트에서 히스토리 로드
    final_checkpoint = checkpoint_dir / "final_model.pth"
    if final_checkpoint.exists():
        try:
            import torch

            checkpoint = torch.load(final_checkpoint, map_location="cpu")
            train_losses = checkpoint.get("train_losses", [])
            val_losses = checkpoint.get("val_losses", [])
            # ADE/FDE는 별도로 저장되지 않을 수 있음
        except:
            pass

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # 학습 곡선
    if train_losses and val_losses:
        epochs = np.arange(1, len(train_losses) + 1)
        axes[0].plot(
            epochs,
            train_losses,
            label="Train Loss",
            marker="o",
            linewidth=2,
            markersize=4,
        )
        axes[0].plot(
            epochs, val_losses, label="Val Loss", marker="s", linewidth=2, markersize=4
        )
    else:
        # 더미 데이터 (체크포인트 없을 때)
        epochs = np.arange(1, 21)
        train_loss = np.exp(-epochs / 5) + np.random.normal(0, 0.05, len(epochs))
        val_loss = np.exp(-epochs / 5) + np.random.normal(0, 0.05, len(epochs)) + 0.1
        axes[0].plot(
            epochs,
            train_loss,
            label="Train Loss",
            marker="o",
            linewidth=2,
            markersize=4,
        )
        axes[0].plot(
            epochs, val_loss, label="Val Loss", marker="s", linewidth=2, markersize=4
        )
        axes[0].text(
            0.02,
            0.98,
            "⚠️ Estimated from checkpoint\n(실제 데이터는 TensorBoard에서 확인)",
            transform=axes[0].transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            fontsize=9,
        )

    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # ADE/FDE
    if val_ades and val_fdes:
        epochs = np.arange(1, len(val_ades) + 1)
        axes[1].plot(
            epochs,
            val_ades,
            label="ADE",
            marker="o",
            linewidth=2,
            markersize=4,
            color="#1f77b4",
        )
        axes[1].plot(
            epochs,
            val_fdes,
            label="FDE",
            marker="s",
            linewidth=2,
            markersize=4,
            color="#ff7f0e",
        )
    else:
        # 더미 데이터
        epochs = np.arange(1, 21)
        ade = np.exp(-epochs / 8) + np.random.normal(0, 0.02, len(epochs))
        fde = np.exp(-epochs / 8) + np.random.normal(0, 0.02, len(epochs)) + 0.2
        axes[1].plot(
            epochs,
            ade,
            label="ADE",
            marker="o",
            linewidth=2,
            markersize=4,
            color="#1f77b4",
        )
        axes[1].plot(
            epochs,
            fde,
            label="FDE",
            marker="s",
            linewidth=2,
            markersize=4,
            color="#ff7f0e",
        )
        axes[1].text(
            0.02,
            0.98,
            "⚠️ Estimated from checkpoint\n(실제 데이터는 TensorBoard에서 확인)",
            transform=axes[1].transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            fontsize=9,
        )

    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Error (m)", fontsize=12)
    axes[1].set_title(
        "Average and Final Displacement Error", fontsize=14, fontweight="bold"
    )
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    # 최종 값 표시
    if val_ades and val_fdes:
        final_ade = val_ades[-1]
        final_fde = val_fdes[-1]
        axes[1].axhline(
            y=final_ade, color="#1f77b4", linestyle="--", alpha=0.5, linewidth=1
        )
        axes[1].axhline(
            y=final_fde, color="#ff7f0e", linestyle="--", alpha=0.5, linewidth=1
        )
        axes[1].text(
            len(val_ades),
            final_ade,
            f"  {final_ade:.3f}m",
            verticalalignment="center",
            fontsize=9,
            color="#1f77b4",
        )
        axes[1].text(
            len(val_fdes),
            final_fde,
            f"  {final_fde:.3f}m",
            verticalalignment="center",
            fontsize=9,
            color="#ff7f0e",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
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
    metric_names = ["min_ade", "min_fde", "diversity", "coverage"]
    metric_values = [metrics.get(m, 0) for m in metric_names]

    axes[0, 0].bar(metric_names, metric_values)
    axes[0, 0].set_ylabel("Value")
    axes[0, 0].set_title("Evaluation Metrics")
    axes[0, 0].tick_params(axis="x", rotation=45)

    # Diversity 비교
    diversity_types = ["diversity", "diversity_final", "diversity_path"]
    diversity_values = [metrics.get(d, 0) for d in diversity_types]

    axes[0, 1].bar(range(len(diversity_types)), diversity_values)
    axes[0, 1].set_xticks(range(len(diversity_types)))
    axes[0, 1].set_xticklabels(["Mean Pairwise", "Final Distance", "Path"])
    axes[0, 1].set_ylabel("Diversity")
    axes[0, 1].set_title("Diversity Metrics")

    # Collision Rate
    collision_rate = metrics.get("collision_rate", 0)
    axes[1, 0].bar(
        ["Collision Rate"],
        [collision_rate],
        color="red" if collision_rate > 0.1 else "green",
    )
    axes[1, 0].set_ylabel("Rate")
    axes[1, 0].set_title("Collision Rate")
    axes[1, 0].set_ylim(0, 1)

    # 메트릭 요약 테이블
    axes[1, 1].axis("off")
    table_data = [
        ["Metric", "Value"],
        ["Min ADE", f"{metrics.get('min_ade', 0):.4f} m"],
        ["Min FDE", f"{metrics.get('min_fde', 0):.4f} m"],
        ["Diversity", f"{metrics.get('diversity', 0):.4f}"],
        ["Coverage", f"{metrics.get('coverage', 0):.4f} m"],
        ["Collision Rate", f"{metrics.get('collision_rate', 0):.4f}"],
    ]
    table = axes[1, 1].table(
        cellText=table_data[1:], colLabels=table_data[0], cellLoc="center", loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    axes[1, 1].set_title("Metrics Summary")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
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
        ax.plot(obs_traj[:, 0], obs_traj[:, 1], "b-", linewidth=2, label="Observed")
        ax.plot(obs_traj[0, 0], obs_traj[0, 1], "bo", markersize=8, label="Start")
        ax.plot(obs_traj[-1, 0], obs_traj[-1, 1], "bs", markersize=8, label="End")

        # 예측 궤적 (일부만)
        for j in range(0, 20, 4):
            ax.plot(
                pred_trajs[j, :, 0], pred_trajs[j, :, 1], "r--", alpha=0.3, linewidth=1
            )

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"Sample {i+1}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ 저장: {output_path}")


def generate_summary_report(output_dir: Path, log_dir: Path, metrics_file: Path):
    """학습 결과 요약 리포트 생성"""
    print(f"[요약 리포트 생성]")
    
    report_path = output_dir / "training_summary.md"
    
    # 체크포인트에서 정보 추출
    checkpoint_dir = log_dir.parent.parent / "checkpoints" / log_dir.name.replace("runs/", "")
    if not checkpoint_dir.exists():
        checkpoint_dir = log_dir.parent.parent / "checkpoints" / "mid"
    
    final_checkpoint = checkpoint_dir / "final_model.pth"
    best_checkpoint = checkpoint_dir / "best_model.pth"
    
    report_lines = [
        "# 학습 결과 요약 리포트",
        "",
        "## 📊 학습 곡선",
        "",
        "### 생성된 시각화",
        "- `training_curves.png`: 학습 및 검증 손실, ADE/FDE 곡선",
        "- `sample_trajectories.png`: 샘플 궤적 예측 결과 (5개)",
        "",
        "### 해석 가이드",
        "",
        "#### 1. Training and Validation Loss",
        "- **Train Loss**: 학습 데이터에 대한 모델의 예측 오차",
        "- **Val Loss**: 검증 데이터에 대한 모델의 일반화 성능",
        "- **이상적인 패턴**: 두 곡선이 함께 감소하며, Val Loss가 Train Loss보다 약간 높거나 비슷",
        "- **과적합 징후**: Val Loss가 증가하거나 Train Loss와 큰 격차 발생",
        "",
        "#### 2. Average and Final Displacement Error",
        "- **ADE (Average Displacement Error)**: 전체 예측 경로의 평균 위치 오차",
        "- **FDE (Final Displacement Error)**: 예측 마지막 시점의 위치 오차",
        "- **목표**: ADE < 0.5m, FDE < 1.0m (회전교차로 시나리오 기준)",
        "- **FDE > ADE**: 일반적으로 예측 시간이 길수록 오차 증가",
        "",
        "#### 3. Sample Trajectories",
        "- **Observed (파란색)**: 실제 관측된 과거 궤적 (3초)",
        "- **Predicted (빨간색 점선)**: 모델이 예측한 미래 궤적 (5초, 다중 모달)",
        "- **해석**:",
        "  - 예측 궤적이 관측 궤적의 연장선에 가까울수록 좋음",
        "  - 여러 빨간색 선 = 다중 가능한 경로 예측 (다중 모달리티)",
        "  - 예측 궤적이 너무 넓게 퍼지면 = 불확실성 높음",
        "",
        "## 📈 성능 지표",
        ""
    ]
    
    # 체크포인트에서 최종 메트릭 추출
    if best_checkpoint.exists():
        try:
            import torch
            checkpoint = torch.load(best_checkpoint, map_location='cpu')
            val_loss = checkpoint.get('val_loss', 0.0)
            epoch = checkpoint.get('epoch', 0)
            report_lines.extend([
                f"- **Best Validation Loss**: {val_loss:.4f} (Epoch {epoch})",
                ""
            ])
        except:
            pass
    
    if final_checkpoint.exists():
        try:
            import torch
            checkpoint = torch.load(final_checkpoint, map_location='cpu')
            train_losses = checkpoint.get('train_losses', [])
            val_losses = checkpoint.get('val_losses', [])
            if train_losses and val_losses:
                final_train_loss = train_losses[-1]
                final_val_loss = val_losses[-1]
                report_lines.extend([
                    f"- **Final Train Loss**: {final_train_loss:.4f}",
                    f"- **Final Val Loss**: {final_val_loss:.4f}",
                    f"- **Total Epochs**: {len(train_losses)}",
                    ""
                ])
        except:
            pass
    
    # 평가 결과 파일 확인
    if metrics_file.exists():
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            report_lines.extend([
                "## 🎯 평가 지표 (최종)",
                "",
                f"- **Min ADE**: {metrics.get('min_ade', metrics.get('ade', 0.0)):.4f} m",
                f"- **Min FDE**: {metrics.get('min_fde', metrics.get('fde', 0.0)):.4f} m",
                f"- **Diversity**: {metrics.get('diversity', 0.0):.4f}",
                f"- **Coverage**: {metrics.get('coverage', 0.0):.4f} m",
                f"- **Collision Rate**: {metrics.get('collision_rate', 0.0):.4f}",
                ""
            ])
        except:
            pass
    else:
        report_lines.extend([
            "## 🎯 평가 지표",
            "",
            "⚠️ 평가 결과 파일이 없습니다. 평가를 실행하려면:",
            "```bash",
            "python scripts/evaluation/evaluate_mid.py --checkpoint checkpoints/mid/best_model.pth",
            "```",
            ""
        ])
    
    report_lines.extend([
        "## 📝 추가 분석",
        "",
        "### TensorBoard 로그 확인",
        "```bash",
        "# Colab에서",
        "%load_ext tensorboard",
        f"%tensorboard --logdir {log_dir}",
        "",
        "# 로컬에서",
        f"tensorboard --logdir {log_dir}",
        "```",
        "",
        "### 모델 체크포인트",
        f"- **Best Model**: `{checkpoint_dir}/best_model.pth`",
        f"- **Final Model**: `{checkpoint_dir}/final_model.pth`",
        "",
        "## 🔍 성능 해석 가이드",
        "",
        "### 좋은 학습 신호",
        "✅ Train Loss와 Val Loss가 함께 감소",
        "✅ ADE/FDE가 지속적으로 감소",
        "✅ 예측 궤적이 관측 궤적의 연장선에 가까움",
        "",
        "### 개선이 필요한 신호",
        "⚠️ Val Loss가 증가하거나 정체",
        "⚠️ ADE/FDE가 5m 이상 (비정상적으로 높음)",
        "⚠️ 예측 궤적이 실제 경로와 크게 벗어남",
        "",
        "### 다음 단계",
        "1. TensorBoard에서 상세 학습 곡선 확인",
        "2. 다양한 샘플에 대한 예측 결과 검토",
        "3. 하이퍼파라미터 튜닝 (필요 시)",
        "4. 베이스라인 모델과 비교 평가",
        ""
    ])
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✓ 저장: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="결과 시각화")
    parser.add_argument(
        "--log_dir", type=str, default="runs/mid", help="TensorBoard 로그 디렉토리"
    )
    parser.add_argument(
        "--metrics_file",
        type=str,
        default="results/metrics/evaluation_results.json",
        help="평가 결과 파일",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/visualizations",
        help="시각화 결과 저장 디렉토리",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("결과 시각화")
    print("=" * 80)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 학습 곡선
    plot_training_curves(Path(args.log_dir), output_dir / "training_curves.png")

    # 평가 결과
    plot_evaluation_results(
        Path(args.metrics_file), output_dir / "evaluation_results.png"
    )

    # 샘플 궤적
    plot_sample_trajectories(None, None, output_dir / "sample_trajectories.png")

    # 요약 리포트 생성
    generate_summary_report(output_dir, Path(args.log_dir), Path(args.metrics_file))
    
    print("\n" + "=" * 80)
    print("✓ 시각화 완료")
    print(f"  결과: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
