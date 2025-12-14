"""
Colab MID 전체 파이프라인 실행 스크립트
HSG-Diffusion 학습 자동화
"""

import argparse
import sys
import yaml
import torch
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(description="Colab MID 파이프라인")
    parser.add_argument(
        "--mode",
        type=str,
        default="fast",
        choices=["fast", "standard"],
        help="학습 모드",
    )
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
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="설정 파일 경로 (None이면 mode에 따라 자동 선택)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("HSG-Diffusion Colab 자동화 파이프라인")
    print("=" * 80)

    # 1. 설정 로드
    print("\n[1/6] 설정 로드...")
    if args.config:
        config_file = Path(args.config)
    else:
        config_file = project_root / f"configs/mid_config_{args.mode}.yaml"

    if not config_file.exists():
        print(f"⚠️  설정 파일 없음: {config_file}")
        print("기본 설정 사용")
        config = {
            "model": {
                "obs_steps": 30,
                "pred_steps": 50,
                "hidden_dim": 128 if args.mode == "fast" else 256,
                "num_diffusion_steps": 50 if args.mode == "fast" else 100,
                "node_features": 9,
            },
            "data": {
                "batch_size": 16 if args.mode == "fast" else 32,
                "train_ratio": 0.7,
                "val_ratio": 0.15,
                "test_ratio": 0.15,
            },
            "training": {
                "num_epochs": 20 if args.mode == "fast" else 100,
                "optimizer": "adamw",
                "learning_rate": 0.001,
                "use_amp": True,
            },
            "evaluation": {"num_samples": 20, "ddim_steps": 2},
        }
    else:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
    print(f"  ✓ 설정: {config_file}")

    # 2. 데이터 전처리
    print("\n[2/6] 데이터 전처리...")
    from src.data_processing.preprocessor import TrajectoryPreprocessor
    from src.integration.sdd_data_adapter import SDDDataAdapter
    import pandas as pd
    import pickle

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

    # 에이전트 타입 확인
    if "agent_type" in combined_df.columns:
        agent_types = combined_df["agent_type"].unique()
        print(f"  ✓ 에이전트 타입: {list(agent_types)}")

    # 샘플링 (Fast 모드)
    if args.mode == "fast":
        import numpy as np

        sample_ratio = 0.3
        unique_tracks = combined_df["track_id"].unique()
        sampled_tracks = np.random.choice(
            unique_tracks, size=int(len(unique_tracks) * sample_ratio), replace=False
        )
        combined_df = combined_df[combined_df["track_id"].isin(sampled_tracks)]
        print(f"  ✓ 샘플링: {len(combined_df):,}행 ({sample_ratio*100:.0f}%)")

    # 전처리
    preprocessor = TrajectoryPreprocessor(
        obs_window=config["model"]["obs_steps"],
        pred_window=config["model"]["pred_steps"],
        sampling_rate=10.0,
    )

    windows = preprocessor.create_sliding_windows(combined_df)
    print(f"  ✓ 윈도우 생성: {len(windows)}개")

    # 저장
    with open(output_dir / "sdd_windows.pkl", "wb") as f:
        pickle.dump(windows, f)
    print(f"  ✓ 저장 완료: {output_dir / 'sdd_windows.pkl'}")

    # 3. 데이터 로더 생성 (씬 그래프 포함)
    print("\n[3/6] 데이터 로더 생성...")
    from src.training.data_loader import (
        TrajectoryDataset,
        create_dataloader,
        split_dataset,
    )
    from src.scene_graph.scene_graph_builder import SceneGraphBuilder

    # 씬 그래프 빌더 생성
    scene_graph_builder = SceneGraphBuilder(spatial_threshold=20.0)
    print("  ✓ 씬 그래프 빌더 생성")

    train_windows, val_windows, test_windows = split_dataset(
        windows,
        train_ratio=config["data"].get("train_ratio", 0.7),
        val_ratio=config["data"].get("val_ratio", 0.15),
        test_ratio=config["data"].get("test_ratio", 0.15),
    )

    # 씬 그래프 빌더를 데이터셋에 전달
    train_dataset = TrajectoryDataset(
        train_windows, scene_graph_builder=scene_graph_builder, use_scene_graph=True
    )
    val_dataset = TrajectoryDataset(
        val_windows, scene_graph_builder=scene_graph_builder, use_scene_graph=True
    )

    batch_size = config["data"].get("batch_size", 32)
    train_loader = create_dataloader(train_dataset, batch_size=batch_size)
    val_loader = create_dataloader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"  ✓ 학습 데이터: {len(train_windows)}개")
    print(f"  ✓ 검증 데이터: {len(val_windows)}개")

    # 4. MID 모델 생성 ✅
    print("\n[4/6] MID 모델 생성...")
    from src.models.mid_integrated import create_fully_integrated_mid

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  디바이스: {device}")

    model = create_fully_integrated_mid(
        obs_steps=config["model"]["obs_steps"],
        pred_steps=config["model"]["pred_steps"],
        hidden_dim=config["model"]["hidden_dim"],
        num_diffusion_steps=config["model"]["num_diffusion_steps"],
        node_features=config["model"]["node_features"],
        use_safety=True,
        node_types=["car", "pedestrian", "biker", "skater", "cart", "bus"],
    )
    model = model.to(device)

    print(f"  ✓ MID 모델 생성 완료")
    print(f"    파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

    # 5. MID Trainer로 학습 ✅
    print("\n[5/6] 학습 시작...")
    from src.training.mid_trainer import MIDTrainer

    trainer = MIDTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config["training"],
        device=device,
    )

    num_epochs = config["training"]["num_epochs"]
    trainer.train(num_epochs)

    # 6. 평가 (Diffusion 지표 포함) ✅
    print("\n[6/6] 평가...")
    from src.evaluation.diffusion_metrics import DiffusionEvaluator

    evaluator = DiffusionEvaluator(k=config["evaluation"]["num_samples"])

    model.eval()
    all_samples = []
    all_ground_truths = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if isinstance(batch, dict):
                obs_data = batch.get("obs_data", batch.get("obs_trajectory"))
                future_data = batch.get("future_data", batch.get("future_trajectory"))
                hetero_graph = batch.get("hetero_graph")
            else:
                obs_data = batch[0]
                future_data = batch[1] if len(batch) > 1 else None
                hetero_graph = batch[2] if len(batch) > 2 else None

            if obs_data is not None:
                obs_data = obs_data.to(device)
            if future_data is not None:
                future_data = future_data.to(device)
            if hetero_graph is not None:
                hetero_graph = hetero_graph.to(device)

            # 샘플링
            result = model.sample(
                hetero_data=hetero_graph,
                obs_trajectory=obs_data[:, :, :2] if obs_data is not None else None,
                num_samples=config["evaluation"]["num_samples"],
                ddim_steps=config["evaluation"]["ddim_steps"],
                use_safety_filter=True,
            )

            if isinstance(result, dict):
                samples = result.get("safe_samples", result.get("samples"))
            else:
                samples = result

            if samples is not None and future_data is not None:
                all_samples.append(samples.detach().cpu().numpy())
                all_ground_truths.append(future_data.detach().cpu().numpy())

            if batch_idx >= 10:  # 처음 10개 배치만 평가
                break

    if all_samples and all_ground_truths:
        import numpy as np

        samples_np = np.concatenate(
            all_samples, axis=1
        )  # [num_samples, total_batch, pred_steps, 2]
        ground_truths_np = np.concatenate(
            all_ground_truths, axis=0
        )  # [total_batch, pred_steps, 2]

        metrics = evaluator.evaluate(samples_np, ground_truths_np)

        print(f"\n📊 평가 결과:")
        print(f"  Min ADE: {metrics.get('min_ade', 0):.4f} m")
        print(f"  Min FDE: {metrics.get('min_fde', 0):.4f} m")
        print(f"  Diversity: {metrics.get('diversity', 0):.4f}")
        print(f"  Coverage: {metrics.get('coverage', 0):.4f}")
        print(f"  Collision Rate: {metrics.get('collision_rate', 0):.4f}")

    print("\n" + "=" * 80)
    print("✓ HSG-Diffusion 학습 완료!")
    print("=" * 80)


if __name__ == "__main__":
    main()
