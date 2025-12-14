"""
A3TGCN 베이스라인 학습 스크립트
Colab 자동화 파이프라인 호환
"""

import argparse
import yaml
import torch
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.baselines.a3tgcn_model import create_a3tgcn_model
from src.training.trainer import ModelTrainer
from src.training.data_loader import (
    TrajectoryDataset,
    create_dataloader,
    split_dataset,
)
from src.scene_graph.scene_graph_builder import SceneGraphBuilder
import pickle


def load_config(config_path: str) -> dict:
    """설정 파일 로드"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_processed_data(data_dir: Path):
    """전처리된 데이터 로드"""
    data_file = data_dir / "sdd_windows.pkl"

    if not data_file.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_file}")

    with open(data_file, "rb") as f:
        windows = pickle.load(f)

    print(f"✓ 데이터 로드: {len(windows)}개 윈도우")
    return windows


def setup_device(config: dict) -> torch.device:
    """디바이스 설정"""
    if config["device"]["use_cuda"] and torch.cuda.is_available():
        device_id = config["device"].get("device_id", 0)
        device = torch.device(f"cuda:{device_id}")
        print(f"✓ GPU 사용: {torch.cuda.get_device_name(device_id)}")
    else:
        device = torch.device("cpu")
        print("✓ CPU 사용")

    return device


def create_data_loaders(windows, config: dict, scene_graph_builder=None):
    """데이터 로더 생성"""
    # 데이터 분할
    train_windows, val_windows, test_windows = split_dataset(
        windows,
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
        test_ratio=config["data"]["test_ratio"],
    )

    print(f"  학습: {len(train_windows)}개")
    print(f"  검증: {len(val_windows)}개")
    print(f"  테스트: {len(test_windows)}개")

    # 데이터셋 생성 (씬 그래프 포함)
    train_dataset = TrajectoryDataset(
        train_windows,
        scene_graph_builder=scene_graph_builder,
        use_scene_graph=config["data"].get("use_scene_graph", True),
    )
    val_dataset = TrajectoryDataset(
        val_windows,
        scene_graph_builder=scene_graph_builder,
        use_scene_graph=config["data"].get("use_scene_graph", True),
    )

    # 데이터 로더 생성
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"].get("num_workers", 2),
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"].get("num_workers", 2),
    )

    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="A3TGCN 베이스라인 학습")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/a3tgcn_config.yaml",
        help="설정 파일 경로",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="전처리된 데이터 디렉토리",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="체크포인트 경로 (재개용)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("A3TGCN 베이스라인 학습")
    print("=" * 80)

    # 설정 로드
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"⚠️  설정 파일 없음: {config_path}")
        print("기본 설정 사용")
        config = {
            "model": {
                "node_features": 9,
                "hidden_channels": 64,
                "pred_steps": 50,
                "num_layers": 2,
                "use_map": False,
            },
            "data": {
                "data_dir": args.data_dir,
                "batch_size": 32,
                "train_ratio": 0.7,
                "val_ratio": 0.15,
                "test_ratio": 0.15,
                "use_scene_graph": True,
            },
            "training": {
                "optimizer": "adam",
                "learning_rate": 0.001,
                "num_epochs": 100,
                "early_stopping": {"patience": 15, "min_delta": 0.001},
            },
            "logging": {
                "log_dir": "runs/a3tgcn",
                "save_dir": "checkpoints/a3tgcn",
            },
            "device": {"use_cuda": True, "device_id": 0},
        }
    else:
        config = load_config(args.config)
        print(f"\n✓ 설정 로드: {args.config}")

    # 디바이스 설정
    device = setup_device(config)

    # 데이터 로드
    data_dir = Path(args.data_dir)
    windows = load_processed_data(data_dir)

    # 씬 그래프 빌더 생성
    scene_graph_builder = None
    if config["data"].get("use_scene_graph", True):
        scene_graph_builder = SceneGraphBuilder(spatial_threshold=20.0)
        print("  ✓ 씬 그래프 빌더 생성")

    # 데이터 로더 생성
    print("\n[데이터 로더 생성]")
    train_loader, val_loader = create_data_loaders(windows, config, scene_graph_builder)

    # 모델 생성
    print("\n[모델 생성]")
    model_config = config["model"]
    model = create_a3tgcn_model(
        node_features=model_config["node_features"],
        hidden_channels=model_config["hidden_channels"],
        pred_steps=model_config["pred_steps"],
        use_map=model_config.get("use_map", False),
    )

    print(f"  모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

    # 체크포인트 로드 (재개)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"✓ 체크포인트 로드: {args.resume}")

    # Trainer 생성
    print("\n[Trainer 생성]")
    trainer_config = config["training"].copy()
    trainer_config.update(
        {
            "log_dir": config["logging"]["log_dir"],
            "save_dir": config["logging"]["save_dir"],
        }
    )

    # Trainer 생성 (device 파라미터 추가)
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=trainer_config,
    )

    # 학습 시작
    print("\n" + "=" * 80)
    print("학습 시작")
    print("=" * 80)

    num_epochs = trainer_config.get("num_epochs", 100)
    trainer.train(num_epochs)

    print("\n" + "=" * 80)
    print("✓ 학습 완료!")
    print("=" * 80)


if __name__ == "__main__":
    main()
