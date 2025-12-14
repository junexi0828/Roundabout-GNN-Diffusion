"""
MID 모델 평가 스크립트
Colab 자동화 파이프라인용
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
import json
from typing import Dict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.mid_model import create_mid_model
from src.evaluation.diffusion_metrics import DiffusionEvaluator
from src.training.data_loader import TrajectoryDataset, create_dataloader, split_dataset
import pickle


def load_model(checkpoint_path: Path, device: torch.device):
    """모델 로드"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 설정에서 모델 파라미터 추출
    config = checkpoint.get('config', {})
    model_config = config.get('model', {})

    model = create_mid_model(
        obs_steps=model_config.get('obs_steps', 30),
        pred_steps=model_config.get('pred_steps', 50),
        hidden_dim=model_config.get('hidden_dim', 128),
        num_diffusion_steps=model_config.get('num_diffusion_steps', 100),
        use_gnn=model_config.get('use_gnn', True),
        node_features=model_config.get('node_features', 9)
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model


def evaluate(model, test_loader, device, num_samples=20):
    """모델 평가"""
    evaluator = DiffusionEvaluator(k=num_samples)

    all_predictions = []
    all_ground_truths = []

    print("\n[평가 실행]")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if isinstance(batch, dict):
                obs_data = batch.get('obs_data', batch.get('obs_trajectory'))
                future_data = batch.get('future_data', batch.get('future_trajectory'))
                graph_data = batch.get('graph_data', batch.get('graph'))
            else:
                obs_data = batch[0]
                future_data = batch[1] if len(batch) > 1 else None
                graph_data = batch[2] if len(batch) > 2 else None

            if obs_data is not None:
                obs_data = obs_data.to(device)
            if future_data is not None:
                future_data = future_data.to(device)
            if graph_data is not None:
                graph_data = graph_data.to(device)

            # 샘플링
            samples = model.sample(
                graph_data=graph_data,
                obs_trajectory=obs_data[:, :, :2] if obs_data is not None else None,
                num_samples=num_samples,
                ddim_steps=2
            )

            # CPU로 이동
            samples_np = samples.detach().cpu().numpy()
            gt_np = future_data.detach().cpu().numpy() if future_data is not None else None

            all_predictions.append(samples_np)
            if gt_np is not None:
                all_ground_truths.append(gt_np)

            if (batch_idx + 1) % 10 == 0:
                print(f"  진행: {batch_idx + 1}/{len(test_loader)}")

    # 평가 지표 계산
    if all_predictions and all_ground_truths:
        samples = np.concatenate(all_predictions, axis=1)  # [num_samples, total_batch, pred_steps, 2]
        ground_truths = np.concatenate(all_ground_truths, axis=0)  # [total_batch, pred_steps, 2]

        metrics = evaluator.evaluate(samples, ground_truths)
        return metrics
    else:
        return {}


def main():
    parser = argparse.ArgumentParser(description="MID 모델 평가")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/mid/best_model.pth",
        help="체크포인트 경로"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="전처리된 데이터 디렉토리"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/metrics",
        help="결과 저장 디렉토리"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="평가 샘플 수"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("MID 모델 평가")
    print("=" * 80)

    # 디바이스
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"디바이스: {device}")

    # 모델 로드
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"⚠️  체크포인트 없음: {checkpoint_path}")
        print("학습이 완료되지 않았습니다.")
        return

    print(f"\n[모델 로드] {checkpoint_path}")
    model = load_model(checkpoint_path, device)
    print("✓ 모델 로드 완료")

    # 데이터 로드
    data_file = Path(args.data_dir) / "sdd_windows.pkl"
    if not data_file.exists():
        print(f"⚠️  데이터 파일 없음: {data_file}")
        return

    print(f"\n[데이터 로드] {data_file}")
    with open(data_file, "rb") as f:
        windows = pickle.load(f)

    _, _, test_windows = split_dataset(windows, 0.7, 0.15, 0.15)
    test_dataset = TrajectoryDataset(test_windows)
    test_loader = create_dataloader(test_dataset, batch_size=16, shuffle=False)

    print(f"✓ 테스트 데이터: {len(test_windows)}개")

    # 평가
    metrics = evaluate(model, test_loader, device, args.num_samples)

    # 결과 저장
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 80)
    print("평가 결과")
    print("=" * 80)
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    print(f"\n✓ 결과 저장: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()

