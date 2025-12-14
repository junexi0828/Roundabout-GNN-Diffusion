"""
Trajectron++ 베이스라인 학습 스크립트
Colab 자동화 파이프라인 호환
"""

import argparse
import yaml
import torch
from pathlib import Path
import sys
import warnings

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.baselines.trajectron_integration import (
    TrajectronIntegration,
    check_trajectron_installation,
    print_installation_guide,
)
from src.training.data_loader import (
    TrajectoryDataset,
    create_dataloader,
    split_dataset,
)
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


def create_data_loaders(windows, config: dict):
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

    # 데이터셋 생성
    train_dataset = TrajectoryDataset(
        train_windows,
        scene_graph_builder=None,
        use_scene_graph=False,  # Trajectron++는 자체 맵 인코딩 사용
    )
    val_dataset = TrajectoryDataset(
        val_windows,
        scene_graph_builder=None,
        use_scene_graph=False,
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
    parser = argparse.ArgumentParser(description="Trajectron++ 베이스라인 학습")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/trajectron_config.yaml",
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
    print("Trajectron++ 베이스라인 학습")
    print("=" * 80)

    # Trajectron++ 설치 확인
    print("\n[Trajectron++ 설치 확인]")
    installation_status = check_trajectron_installation()

    if not installation_status["trajdata"]:
        print("❌ trajdata가 설치되지 않았습니다.")
        print_installation_guide()
        sys.exit(1)

    if not installation_status["trajectron"]:
        print("⚠️  Trajectron++가 설치되지 않았습니다.")
        print("  일부 기능이 제한될 수 있습니다.")
        print("\n설치 방법:")
        print("  git clone https://github.com/StanfordASL/Trajectron-plus-plus.git")
        print("  cd Trajectron-plus-plus")
        print("  pip install -e .")
        warnings.warn(
            "Trajectron++ 미설치 - 대체 구현을 사용합니다."
        )

    # 설정 로드
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"⚠️  설정 파일 없음: {config_path}")
        print("기본 설정 생성")

        # 기본 설정 생성
        integration = TrajectronIntegration(Path(args.data_dir))
        config = integration.get_config_template()

        # 추가 설정
        config["data"].update({
            "data_dir": args.data_dir,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "num_workers": 2,
        })
        config["logging"] = {
            "log_dir": "runs/trajectron",
            "save_dir": "checkpoints/trajectron",
        }
        config["device"] = {"use_cuda": True, "device_id": 0}

        # 설정 파일 저장
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"✓ 기본 설정 저장: {config_path}")
    else:
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

    print("\n" + "=" * 80)
    print("⚠️  Trajectron++ 학습 구현 중")
    print("=" * 80)
    print("\nTrajectron++는 복잡한 CVAE 기반 모델로,")
    print("완전한 통합을 위해서는 추가 구현이 필요합니다.")
    print("\n현재 상태:")
    print("  ✓ 데이터 로딩 완료")
    print("  ✓ 설정 파일 준비 완료")
    print("  ⚠️  모델 학습 코드 구현 필요")
    print("\n참고:")
    print("  - Trajectron++ 공식 저장소: https://github.com/StanfordASL/Trajectron-plus-plus")
    print("  - 설정 템플릿: configs/trajectron_config.yaml")
    print("\n베이스라인 비교를 위해서는:")
    print("  1. Trajectron++ 공식 구현 사용")
    print("  2. 또는 사전 학습된 모델 사용")
    print("  3. 또는 A3TGCN만으로 비교")


if __name__ == "__main__":
    main()
