"""
Colab 전체 파이프라인 실행 스크립트
데이터 전처리부터 학습까지 자동 실행
"""

import argparse
import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(description="Colab 전체 파이프라인")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/content/Roundabout_AI/data/sdd/converted",
        help="변환된 데이터 디렉토리",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/content/Roundabout_AI/data/processed",
        help="전처리 결과 저장 디렉토리",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="배치 크기")
    parser.add_argument("--epochs", type=int, default=30, help="에폭 수")
    parser.add_argument(
        "--sample_ratio", type=float, default=0.3, help="데이터 샘플링 비율"
    )
    parser.add_argument(
        "--use_amp", action="store_true", help="Mixed Precision Training 사용"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Colab 전체 파이프라인 실행")
    print("=" * 80)

    # 1. 데이터 전처리
    print("\n[1/4] 데이터 전처리...")
    from src.data_processing.preprocessor import TrajectoryPreprocessor
    import pandas as pd

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 데이터 로드
    all_data = []
    for csv_file in sorted(data_dir.glob("*.csv")):
        df = pd.read_csv(csv_file)
        all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"  ✓ 데이터 로드: {len(combined_df):,}행")

    # 전처리
    preprocessor = TrajectoryPreprocessor(
        obs_window=30, pred_window=50, sampling_rate=10.0
    )

    windows = preprocessor.create_sliding_windows(combined_df)
    print(f"  ✓ 윈도우 생성: {len(windows)}개")

    # 저장
    import pickle

    with open(output_dir / "sdd_windows.pkl", "wb") as f:
        pickle.dump(windows, f)
    print(f"  ✓ 저장 완료: {output_dir / 'sdd_windows.pkl'}")

    # 2. 데이터 로더 생성
    print("\n[2/4] 데이터 로더 생성...")
    from src.training.data_loader import (
        TrajectoryDataset,
        create_dataloader,
        split_dataset,
    )

    train_windows, val_windows, test_windows = split_dataset(
        windows, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )

    train_dataset = TrajectoryDataset(train_windows)
    val_dataset = TrajectoryDataset(val_windows)

    train_loader = create_dataloader(train_dataset, batch_size=args.batch_size)
    val_loader = create_dataloader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )

    print(f"  ✓ 학습 데이터: {len(train_windows)}개")
    print(f"  ✓ 검증 데이터: {len(val_windows)}개")

    # 3. 모델 생성
    print("\n[3/4] 모델 생성...")
    import torch
    from src.models.a3tgcn_model import create_a3tgcn_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  디바이스: {device}")

    model = create_a3tgcn_model(
        node_features=9, hidden_channels=64, pred_steps=50, use_map=False
    )

    print(f"  ✓ 모델 생성 완료")
    print(f"    파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

    # 4. 학습 실행
    print("\n[4/4] 학습 시작...")
    from src.training.trainer import create_trainer

    config = {
        "optimizer": "adam",
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "scheduler": "reduce_on_plateau",
        "loss": "mse",
        "num_epochs": args.epochs,
        "early_stopping": {"patience": 10, "min_delta": 0.001},
        "log_dir": "/content/Roundabout_AI/runs",
        "save_dir": "/content/Roundabout_AI/checkpoints",
        "max_grad_norm": 1.0,
    }

    trainer = create_trainer(model, train_loader, val_loader, config)
    trainer.train(args.epochs)

    print("\n" + "=" * 80)
    print("✓ 전체 파이프라인 완료!")
    print("=" * 80)


if __name__ == "__main__":
    main()
