"""
하이퍼파라미터 튜닝 스크립트
Grid Search 또는 Random Search를 통한 최적 하이퍼파라미터 탐색
"""

import argparse
import yaml
import json
from pathlib import Path
import torch
import numpy as np
from itertools import product
from typing import Dict, List, Any
import sys

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.train import (
    load_config,
    load_processed_data,
    setup_device,
    create_data_loaders,
    print_training_info
)
from src.models.a3tgcn_model import create_a3tgcn_model
from src.training.trainer import create_trainer


def generate_hyperparameter_grid() -> Dict[str, List[Any]]:
    """
    하이퍼파라미터 그리드 생성

    Returns:
        하이퍼파라미터 그리드 딕셔너리
    """
    return {
        'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        'batch_size': [16, 32, 64],
        'hidden_channels': [32, 64, 128],
        'num_epochs': [50, 100, 150],
        'weight_decay': [0, 1e-5, 1e-4],
        'optimizer': ['adam', 'adamw'],
        'scheduler': ['reduce_on_plateau', 'cosine', None]
    }


def random_search(
    grid: Dict[str, List[Any]],
    n_trials: int = 20
) -> List[Dict[str, Any]]:
    """
    Random Search를 통한 하이퍼파라미터 조합 생성

    Args:
        grid: 하이퍼파라미터 그리드
        n_trials: 시도 횟수

    Returns:
        하이퍼파라미터 조합 리스트
    """
    combinations = []
    for _ in range(n_trials):
        combo = {}
        for key, values in grid.items():
            combo[key] = np.random.choice(values)
        combinations.append(combo)
    return combinations


def grid_search(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Grid Search를 통한 모든 하이퍼파라미터 조합 생성

    Args:
        grid: 하이퍼파라미터 그리드

    Returns:
        하이퍼파라미터 조합 리스트
    """
    keys = grid.keys()
    values = grid.values()
    combinations = []

    for combo in product(*values):
        combinations.append(dict(zip(keys, combo)))

    return combinations


def evaluate_hyperparameters(
    config: Dict[str, Any],
    hyperparams: Dict[str, Any],
    train_loader,
    val_loader,
    device: torch.device,
    max_epochs: int = 30  # 빠른 평가를 위해 에폭 수 제한
) -> float:
    """
    특정 하이퍼파라미터 조합 평가

    Args:
        config: 기본 설정
        hyperparams: 평가할 하이퍼파라미터
        train_loader: 학습 데이터 로더
        val_loader: 검증 데이터 로더
        device: 디바이스
        max_epochs: 최대 에폭 수

    Returns:
        최고 검증 손실
    """
    # 설정 업데이트
    config['training']['learning_rate'] = hyperparams['learning_rate']
    config['data']['batch_size'] = hyperparams['batch_size']
    config['model']['hidden_channels'] = hyperparams['hidden_channels']
    config['training']['num_epochs'] = min(hyperparams['num_epochs'], max_epochs)
    config['training']['weight_decay'] = hyperparams['weight_decay']
    config['training']['optimizer'] = hyperparams['optimizer']
    config['training']['scheduler'] = hyperparams['scheduler']

    # 모델 생성
    model_config = config['model']
    model = create_a3tgcn_model(
        node_features=model_config['node_features'],
        hidden_channels=model_config['hidden_channels'],
        pred_steps=model_config['pred_steps'],
        use_map=False
    )

    # Trainer 생성
    training_config = config['training']
    logging_config = config.get('logging', {})

    trainer_config = {
        **training_config,
        'log_dir': str(project_root / 'runs' / 'hyperparameter_tuning'),
        'save_dir': str(project_root / 'checkpoints' / 'hyperparameter_tuning')
    }

    trainer = create_trainer(model, train_loader, val_loader, trainer_config)

    # 학습 (빠른 평가)
    trainer.train(max_epochs)

    # 최고 검증 손실 반환
    best_val_loss = min(trainer.history['val_loss'])

    return best_val_loss


def main():
    parser = argparse.ArgumentParser(description='하이퍼파라미터 튜닝')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/training_config.yaml',
        help='기본 설정 파일 경로'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/processed',
        help='전처리된 데이터 디렉토리'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['grid', 'random'],
        default='random',
        help='탐색 방법: grid 또는 random'
    )
    parser.add_argument(
        '--n_trials',
        type=int,
        default=20,
        help='Random search 시도 횟수'
    )
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=30,
        help='각 조합 평가 시 최대 에폭 수'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/hyperparameter_tuning.json',
        help='결과 저장 경로'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("하이퍼파라미터 튜닝")
    print("=" * 80)

    # 설정 로드
    config_path = project_root / args.config
    config = load_config(config_path)

    # 디바이스 설정
    device = setup_device(config)

    # 데이터 로드
    print("\n데이터 로딩 중...")
    data_dir = project_root / args.data_dir
    from src.training.train import load_processed_data
    windows = load_processed_data(data_dir)

    # 데이터 로더 생성 (전체 데이터 사용)
    train_loader, val_loader, _ = create_data_loaders(windows, config, device)

    # 하이퍼파라미터 그리드 생성
    grid = generate_hyperparameter_grid()

    if args.method == 'grid':
        combinations = grid_search(grid)
        print(f"\nGrid Search: {len(combinations)}개 조합 탐색")
    else:
        combinations = random_search(grid, n_trials=args.n_trials)
        print(f"\nRandom Search: {args.n_trials}개 조합 탐색")

    # 각 조합 평가
    results = []
    best_loss = float('inf')
    best_params = None

    for i, hyperparams in enumerate(combinations):
        print(f"\n{'='*80}")
        print(f"조합 {i+1}/{len(combinations)}")
        print(f"{'='*80}")
        print(f"Learning Rate: {hyperparams['learning_rate']}")
        print(f"Batch Size: {hyperparams['batch_size']}")
        print(f"Hidden Channels: {hyperparams['hidden_channels']}")
        print(f"Epochs: {hyperparams['num_epochs']}")
        print(f"Weight Decay: {hyperparams['weight_decay']}")
        print(f"Optimizer: {hyperparams['optimizer']}")
        print(f"Scheduler: {hyperparams['scheduler']}")

        try:
            val_loss = evaluate_hyperparameters(
                config,
                hyperparams,
                train_loader,
                val_loader,
                device,
                max_epochs=args.max_epochs
            )

            result = {
                'hyperparameters': hyperparams,
                'val_loss': float(val_loss)
            }
            results.append(result)

            print(f"\n검증 손실: {val_loss:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                best_params = hyperparams
                print(f"✓ 새로운 최고 성능!")

        except Exception as e:
            print(f"✗ 오류 발생: {e}")
            continue

    # 결과 저장
    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        'best_hyperparameters': best_params,
        'best_val_loss': float(best_loss),
        'all_results': results
    }

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print("하이퍼파라미터 튜닝 완료")
    print(f"{'='*80}")
    print(f"\n최고 성능:")
    print(f"  검증 손실: {best_loss:.4f}")
    print(f"  하이퍼파라미터:")
    for key, value in best_params.items():
        print(f"    {key}: {value}")
    print(f"\n결과 저장: {output_path}")


if __name__ == "__main__":
    main()

