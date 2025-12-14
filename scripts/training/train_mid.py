"""
MID 모델 학습 스크립트
"""

import argparse
import yaml
import torch
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.mid_model import create_mid_model
from src.training.mid_trainer import create_mid_trainer
from src.training.data_loader import (
    TrajectoryDataset,
    create_dataloader,
    split_dataset
)
import pickle


def load_config(config_path: str) -> dict:
    """설정 파일 로드"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_processed_data(data_dir: Path):
    """전처리된 데이터 로드"""
    data_file = data_dir / 'sdd_windows.pkl'

    if not data_file.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_file}")

    with open(data_file, 'rb') as f:
        windows = pickle.load(f)

    print(f"✓ 데이터 로드: {len(windows)}개 윈도우")
    return windows


def setup_device(config: dict) -> torch.device:
    """디바이스 설정"""
    # 디바이스 설정 (MPS 우선)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ CUDA 사용 (GPU: {torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ MPS 사용 (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("✓ CPU 사용")

    config['device'] = str(device)
    return device


def create_data_loaders(windows, config: dict):
    """데이터 로더 생성"""
    # 데이터 분할
    train_windows, val_windows, test_windows = split_dataset(
        windows,
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        test_ratio=config['data']['test_ratio']
    )

    print(f"  학습: {len(train_windows)}개")
    print(f"  검증: {len(val_windows)}개")
    print(f"  테스트: {len(test_windows)}개")

    # 데이터셋 생성
    train_dataset = TrajectoryDataset(train_windows)
    val_dataset = TrajectoryDataset(val_windows)

    # 데이터 로더 생성
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 2)
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data'].get('num_workers', 2)
    )

    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description='MID 모델 학습')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/mid_config.yaml',
        help='설정 파일 경로'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/processed',
        help='전처리된 데이터 디렉토리'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='체크포인트 경로 (재개용)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("MID 모델 학습")
    print("=" * 80)

    # 설정 로드
    config = load_config(args.config)
    print(f"\n✓ 설정 로드: {args.config}")

    # 디바이스 설정
    device = setup_device(config)

    # 데이터 로드
    data_dir = Path(args.data_dir)
    windows = load_processed_data(data_dir)

    # 데이터 로더 생성
    print("\n[데이터 로더 생성]")
    train_loader, val_loader = create_data_loaders(windows, config)

    # 모델 생성
    print("\n[모델 생성]")
    model_config = config['model']
    model = create_mid_model(
        obs_steps=model_config['obs_steps'],
        pred_steps=model_config['pred_steps'],
        hidden_dim=model_config['hidden_dim'],
        num_diffusion_steps=model_config['num_diffusion_steps'],
        use_gnn=model_config.get('use_gnn', True),
        node_features=model_config.get('node_features', 9)
    )

    try:
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  모델 파라미터 수: {param_count:,}")
    except:
        print("  모델 파라미터 수: (초기화 후 계산)")

    # 체크포인트 로드 (재개)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ 체크포인트 로드: {args.resume}")

    # Trainer 생성
    print("\n[Trainer 생성]")
    trainer_config = config['training'].copy()
    trainer_config.update({
        'log_dir': config['logging']['log_dir'],
        'save_dir': config['logging']['save_dir'],
        'use_amp': trainer_config.get('use_amp', False)
    })

    trainer = create_mid_trainer(
        model,
        train_loader,
        val_loader,
        trainer_config,
        device
    )

    # 학습 시작
    print("\n" + "=" * 80)
    print("학습 시작")
    print("=" * 80)

    num_epochs = trainer_config.get('num_epochs', 100)
    trainer.train(num_epochs)

    print("\n" + "=" * 80)
    print("✓ 학습 완료!")
    print("=" * 80)


if __name__ == "__main__":
    main()

