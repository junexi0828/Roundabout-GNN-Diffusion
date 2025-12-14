"""
학습 실행 스크립트
전체 학습 파이프라인을 실행하는 메인 스크립트
하이퍼파라미터 튜닝, GPU 메모리 최적화, 학습 모니터링 포함
"""

import argparse
import yaml
from pathlib import Path
import torch
import torch.nn as nn
import sys
import pickle
import numpy as np
from typing import List, Dict

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.a3tgcn_model import create_a3tgcn_model
from src.training.data_loader import (
    TrajectoryDataset,
    create_dataloader,
    split_dataset
)
from src.training.trainer import create_trainer


def load_config(config_path: Path) -> dict:
    """설정 파일 로드"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_processed_data(data_dir: Path) -> List[Dict]:
    """
    전처리된 데이터 로드

    Args:
        data_dir: 전처리된 데이터 디렉토리

    Returns:
        모든 윈도우를 포함한 리스트
    """
    data_dir = Path(data_dir)
    all_windows = []

    # 모든 전처리된 파일 로드
    processed_files = list(data_dir.glob("*_processed.pkl"))

    if not processed_files:
        raise FileNotFoundError(
            f"전처리된 데이터를 찾을 수 없습니다: {data_dir}\n"
            "먼저 전처리 스크립트를 실행하세요."
        )

    print(f"전처리된 파일 {len(processed_files)}개 발견")

    for file_path in processed_files:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            windows = data.get('windows', [])
            all_windows.extend(windows)

    print(f"총 {len(all_windows)}개 윈도우 로드 완료")
    return all_windows


def setup_device(config: dict) -> torch.device:
    """
    디바이스 설정 및 GPU 메모리 최적화

    Args:
        config: 설정 딕셔너리

    Returns:
        torch.device 객체
    """
    device_config = config.get('device', {})
    use_cuda = device_config.get('use_cuda', True) and torch.cuda.is_available()

    if use_cuda:
        device_id = device_config.get('device_id', 0)
        device = torch.device(f'cuda:{device_id}')

        # GPU 메모리 최적화 설정
        torch.cuda.set_device(device_id)
        torch.backends.cudnn.benchmark = True  # cuDNN 자동 튜닝
        torch.backends.cudnn.deterministic = False  # 성능 우선

        # GPU 메모리 정보 출력
        print(f"\nGPU 정보:")
        print(f"  디바이스: {torch.cuda.get_device_name(device_id)}")
        print(f"  메모리: {torch.cuda.get_device_properties(device_id).total_memory / 1e9:.2f} GB")

        # 메모리 캐시 정리
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
        print("\nCPU 모드로 실행")

    return device


def create_data_loaders(
    windows: List[Dict],
    config: dict,
    device: torch.device
) -> tuple:
    """
    학습/검증 데이터 로더 생성

    Args:
        windows: 전체 윈도우 리스트
        config: 설정 딕셔너리
        device: 디바이스

    Returns:
        (train_loader, val_loader, test_loader)
    """
    data_config = config['data']

    # 데이터 분할
    train_ratio = data_config.get('train_ratio', 0.7)
    val_ratio = data_config.get('val_ratio', 0.15)
    test_ratio = data_config.get('test_ratio', 0.15)

    print(f"\n데이터 분할:")
    print(f"  학습: {train_ratio*100:.1f}%, 검증: {val_ratio*100:.1f}%, 테스트: {test_ratio*100:.1f}%")

    train_windows, val_windows, test_windows = split_dataset(
        windows,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )

    print(f"  학습: {len(train_windows)}개, 검증: {len(val_windows)}개, 테스트: {len(test_windows)}개")

    # 데이터셋 생성
    train_dataset = TrajectoryDataset(train_windows)
    val_dataset = TrajectoryDataset(val_windows)
    test_dataset = TrajectoryDataset(test_windows)

    # 데이터 로더 생성
    batch_size = data_config.get('batch_size', 32)
    num_workers = data_config.get('num_workers', 0)

    # GPU 메모리 최적화를 위한 배치 크기 조정
    if device.type == 'cuda':
        # GPU 메모리에 따라 배치 크기 자동 조정
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory_gb < 8:
            # 8GB 미만 GPU는 배치 크기 감소
            batch_size = min(batch_size, 16)
            print(f"  GPU 메모리 최적화: 배치 크기를 {batch_size}로 조정")

    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = create_dataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


def print_training_info(config: dict, model: nn.Module, device: torch.device):
    """학습 정보 출력"""
    print("\n" + "=" * 80)
    print("학습 설정 요약")
    print("=" * 80)

    model_config = config['model']
    training_config = config['training']
    data_config = config['data']

    print(f"\n모델:")
    print(f"  이름: {model_config['name']}")
    print(f"  노드 특징: {model_config['node_features']}")
    print(f"  은닉 채널: {model_config['hidden_channels']}")
    print(f"  레이어 수: {model_config.get('num_layers', 2)}")
    print(f"  예측 스텝: {model_config['pred_steps']}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  파라미터 수: {total_params:,} (학습 가능: {trainable_params:,})")

    print(f"\n학습 하이퍼파라미터:")
    print(f"  Optimizer: {training_config['optimizer']}")
    print(f"  Learning Rate: {training_config['learning_rate']}")
    print(f"  Weight Decay: {training_config.get('weight_decay', 0)}")
    print(f"  Scheduler: {training_config.get('scheduler', 'None')}")
    print(f"  Loss: {training_config['loss']}")
    print(f"  Epochs: {training_config['num_epochs']}")
    print(f"  Gradient Clipping: {training_config.get('max_grad_norm', 'None')}")

    print(f"\n데이터:")
    print(f"  Batch Size: {data_config['batch_size']}")
    print(f"  Workers: {data_config.get('num_workers', 0)}")

    print(f"\n디바이스: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='회전교차로 상호작용 예측 모델 학습')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/training_config.yaml',
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
        help='체크포인트에서 재개할 경로'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=None,
        help='Learning rate (설정 파일 오버라이드)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Batch size (설정 파일 오버라이드)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs (설정 파일 오버라이드)'
    )

    args = parser.parse_args()

    # 설정 로드
    config_path = project_root / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")

    config = load_config(config_path)

    # 커맨드라인 인자로 하이퍼파라미터 오버라이드
    if args.learning_rate is not None:
        config['training']['learning_rate'] = args.learning_rate
    if args.batch_size is not None:
        config['data']['batch_size'] = args.batch_size
    if args.epochs is not None:
        config['training']['num_epochs'] = args.epochs

    print("=" * 80)
    print("회전교차로 상호작용 예측 모델 학습")
    print("=" * 80)
    print(f"\n설정 파일: {config_path}")
    print(f"데이터 디렉토리: {args.data_dir}")

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
    model_config = config['model']
    model = create_a3tgcn_model(
        node_features=model_config['node_features'],
        hidden_channels=model_config['hidden_channels'],
        pred_steps=model_config['pred_steps'],
        use_map=False
    )

    # 학습 정보 출력
    print_training_info(config, model, device)

    # Trainer 생성
    training_config = config['training']
    logging_config = config.get('logging', {})

    trainer_config = {
        **training_config,
        'log_dir': str(project_root / logging_config.get('log_dir', 'runs')),
        'save_dir': str(project_root / logging_config.get('save_dir', 'checkpoints'))
    }

    # 체크포인트에서 재개
    if args.resume:
        trainer = create_trainer(model, train_loader, val_loader, trainer_config)
        checkpoint_path = Path(args.resume)
        start_epoch = trainer.load_checkpoint(checkpoint_path)
        print(f"체크포인트에서 재개: {checkpoint_path} (Epoch {start_epoch})")
    else:
        trainer = create_trainer(model, train_loader, val_loader, trainer_config)

    # 학습 시작
    print("\n학습 시작...")
    print("TensorBoard 로그: tensorboard --logdir runs")
    print("-" * 80)

    try:
        trainer.train(training_config['num_epochs'])
        print("\n✓ 학습 완료!")
    except KeyboardInterrupt:
        print("\n\n학습이 중단되었습니다.")
        print("체크포인트는 저장되었습니다.")
    except Exception as e:
        print(f"\n\n오류 발생: {e}")
        raise


if __name__ == "__main__":
    main()
