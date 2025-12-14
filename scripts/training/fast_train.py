"""
빠른 학습 스크립트
Mixed Precision Training, 데이터 샘플링, 모델 경량화 포함
목표: 4-6시간 → 1-2시간으로 단축
"""

import argparse
import yaml
from pathlib import Path
import torch
import torch.nn as nn
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.a3tgcn_model import create_a3tgcn_model
from src.training.data_loader import (
    TrajectoryDataset,
    create_dataloader,
    split_dataset
)
from src.training.fast_trainer import create_fast_trainer
from src.training.train import load_config, load_processed_data, setup_device, create_data_loaders


def create_lightweight_model(config: dict) -> nn.Module:
    """경량 모델 생성 (더 빠른 학습)"""
    model_config = config['model']

    # 모델 크기 축소
    hidden_channels = model_config.get('hidden_channels', 64)
    if hidden_channels > 32:
        hidden_channels = 32  # 경량화
        print(f"✓ 모델 경량화: hidden_channels {model_config['hidden_channels']} → {hidden_channels}")

    model = create_a3tgcn_model(
        node_features=model_config['node_features'],
        hidden_channels=hidden_channels,
        pred_steps=model_config['pred_steps'],
        use_map=False
    )

    return model


def main():
    parser = argparse.ArgumentParser(description='빠른 모델 학습 (최적화 버전)')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml')
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--batch_size', type=int, default=64, help='큰 배치 크기 (GPU 메모리 허용 시)')
    parser.add_argument('--epochs', type=int, default=30, help='에폭 수 (빠른 학습)')
    parser.add_argument('--sample_ratio', type=float, default=0.3, help='데이터 샘플링 비율 (0.3 = 30%만 사용)')
    parser.add_argument('--use_amp', action='store_true', default=True, help='Mixed Precision Training (FP16)')
    parser.add_argument('--lightweight', action='store_true', default=True, help='경량 모델 사용')

    args = parser.parse_args()

    print("=" * 80)
    print("🚀 빠른 학습 모드 (최적화 버전)")
    print("=" * 80)
    print(f"\n최적화 설정:")
    print(f"  ✓ Mixed Precision Training (FP16): {args.use_amp}")
    print(f"  ✓ 데이터 샘플링: {args.sample_ratio*100:.1f}%")
    print(f"  ✓ 배치 크기: {args.batch_size}")
    print(f"  ✓ 경량 모델: {args.lightweight}")
    print(f"  ✓ 에폭 수: {args.epochs}")
    print(f"\n예상 시간: 4-6시간 → 1-2시간 (약 3배 속도 향상)")

    # 설정 로드
    config_path = project_root / args.config
    config = load_config(config_path)

    # 하이퍼파라미터 오버라이드
    config['data']['batch_size'] = args.batch_size
    config['training']['num_epochs'] = args.epochs

    # 디바이스 설정
    device = setup_device(config)

    # 데이터 로드
    print("\n데이터 로딩 중...")
    data_dir = project_root / args.data_dir
    windows = load_processed_data(data_dir)

    # 데이터 로더 생성
    train_loader, val_loader, test_loader = create_data_loaders(
        windows, config, device
    )

    # 모델 생성
    if args.lightweight:
        model = create_lightweight_model(config)
    else:
        model_config = config['model']
        model = create_a3tgcn_model(
            node_features=model_config['node_features'],
            hidden_channels=model_config['hidden_channels'],
            pred_steps=model_config['pred_steps'],
            use_map=False
        )

    # 빠른 Trainer 생성
    trainer = create_fast_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config['training'],
        use_amp=args.use_amp,
        sample_ratio=args.sample_ratio
    )

    # 학습 시작
    print("\n학습 시작...")
    print("-" * 80)

    try:
        trainer.train(args.epochs)
        print("\n✓ 학습 완료!")
    except KeyboardInterrupt:
        print("\n\n학습이 중단되었습니다.")
    except Exception as e:
        print(f"\n\n오류 발생: {e}")
        raise


if __name__ == "__main__":
    main()

