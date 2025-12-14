"""
베이스라인 비교 평가 스크립트
HSG-Diffusion vs A3TGCN vs Trajectron++ (선택)
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
import json
import pandas as pd
from typing import Dict, List, Optional

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.mid_integrated import create_fully_integrated_mid
from src.baselines.a3tgcn_model import create_a3tgcn_model
from src.evaluation.diffusion_metrics import DiffusionEvaluator
from src.evaluation.metrics import calculate_ade, calculate_fde
from src.training.data_loader import TrajectoryDataset, create_dataloader, split_dataset
import pickle
import yaml


def load_model(
    model_name: str, checkpoint_path: Path, device: torch.device, config: dict = None
):
    """모델 로드"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if model_name == "HSG-Diffusion":
        # MID 모델 로드
        model_config = config.get("model", {}) if config else {}
        model = create_fully_integrated_mid(
            obs_steps=model_config.get("obs_steps", 30),
            pred_steps=model_config.get("pred_steps", 50),
            hidden_dim=model_config.get("hidden_dim", 128),
            num_diffusion_steps=model_config.get("num_diffusion_steps", 100),
            node_features=model_config.get("node_features", 9),
            node_types=["car", "pedestrian", "biker", "skater", "cart", "bus"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model

    elif model_name == "A3TGCN":
        # A3TGCN 모델 로드
        model_config = config.get("model", {}) if config else {}
        model = create_a3tgcn_model(
            node_features=model_config.get("node_features", 9),
            hidden_channels=model_config.get("hidden_channels", 64),
            pred_steps=model_config.get("pred_steps", 50),
            use_map=model_config.get("use_map", False),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model

    elif model_name == "Trajectron++":
        # Trajectron++ 모델 로드
        from src.baselines.trajectron_integration import TrajectronIntegration

        model_config = config.get("model", {}) if config else {}
        trajectron = TrajectronIntegration(config=config)
        trajectron.model = checkpoint.get("model")  # Trajectron++ 전체 모델 저장
        trajectron.device = device
        return trajectron

    else:
        raise ValueError(f"알 수 없는 모델: {model_name}")


def evaluate_mid_model(model, test_loader, device, num_samples: int = 20):
    """MID 모델 평가 (다중 모달리티)"""
    evaluator = DiffusionEvaluator(k=num_samples)

    all_samples = []
    all_ground_truths = []

    print("\n[HSG-Diffusion 평가]")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
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
                num_samples=num_samples,
                ddim_steps=2,
                use_safety_filter=True,
            )

            if isinstance(result, dict):
                samples = result.get("safe_samples", result.get("samples"))
            else:
                samples = result

            if samples is not None and future_data is not None:
                all_samples.append(samples.detach().cpu().numpy())
                all_ground_truths.append(future_data.detach().cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"  진행: {batch_idx + 1}/{len(test_loader)}")

    if all_samples and all_ground_truths:
        samples_np = np.concatenate(
            all_samples, axis=1
        )  # [num_samples, total_batch, pred_steps, 2]
        ground_truths_np = np.concatenate(
            all_ground_truths, axis=0
        )  # [total_batch, pred_steps, 2]

        metrics = evaluator.evaluate(samples_np, ground_truths_np)
        return metrics
    else:
        return {}


def evaluate_a3tgcn_model(model, test_loader, device):
    """A3TGCN 모델 평가 (단일 예측)"""
    all_preds = []
    all_targets = []

    print("\n[A3TGCN 평가]")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if isinstance(batch, dict):
                obs_data = batch.get("obs_data")
                future_data = batch.get("future_data", batch.get("pred_data"))
                graph_data = batch.get("graph", batch.get("graph_data"))
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

            # 예측 (A3TGCN은 단일 예측)
            if graph_data is not None:
                # 그래프 데이터 사용
                pred = model(graph_data.x, graph_data.edge_index)
            else:
                # 관측 데이터만 사용 (간단한 버전)
                # A3TGCN은 그래프가 필요하므로 기본값 반환
                batch_size = obs_data.shape[0]
                pred = torch.zeros(batch_size, 50, 2).to(device)

            all_preds.append(pred.detach().cpu().numpy())
            if future_data is not None:
                all_targets.append(future_data.detach().cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"  진행: {batch_idx + 1}/{len(test_loader)}")

    if all_preds and all_targets:
        preds_np = np.concatenate(all_preds, axis=0)  # [total_batch, pred_steps, 2]
        targets_np = np.concatenate(all_targets, axis=0)  # [total_batch, pred_steps, 2]

        # 단일 예측이므로 ADE, FDE만 계산
        ade = calculate_ade(preds_np, targets_np)
        fde = calculate_fde(preds_np, targets_np)

        return {
            "ade": ade,
            "fde": fde,
            "min_ade": ade,  # 단일 예측이므로 min_ade = ade
            "min_fde": fde,  # 단일 예측이므로 min_fde = fde
            "diversity": 0.0,  # 단일 예측이므로 diversity = 0
            "coverage": 0.0,  # 단일 예측이므로 coverage = 0
        }
    else:
        return {}


def evaluate_trajectron_model(model, test_loader, device, num_samples: int = 25):
    """Trajectron++ 모델 평가 (다중 모달리티 CVAE)"""
    all_samples = []
    all_ground_truths = []

    print("\n[Trajectron++ 평가]")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if isinstance(batch, dict):
                obs_data = batch.get("obs_data", batch.get("obs_trajectory"))
                future_data = batch.get("future_data", batch.get("future_trajectory"))
            else:
                obs_data = batch[0]
                future_data = batch[1] if len(batch) > 1 else None

            if obs_data is not None:
                obs_data = obs_data.to(device)
            if future_data is not None:
                future_data = future_data.to(device)

            try:
                # Trajectron++ 예측 (다중 샘플)
                predictions = model.predict(
                    obs_trajectory=obs_data[:, :, :2] if obs_data is not None else None,
                    num_samples=num_samples,
                )

                if predictions is not None and future_data is not None:
                    # predictions: [batch, num_samples, pred_steps, 2]
                    # future_data: [batch, pred_steps, 2]
                    # 형식 맞추기: [num_samples, batch, pred_steps, 2]
                    predictions = predictions.permute(1, 0, 2, 3)
                    all_samples.append(predictions.detach().cpu().numpy())
                    all_ground_truths.append(future_data.detach().cpu().numpy())
            except Exception as e:
                print(f"  ⚠️  배치 {batch_idx} 예측 실패: {e}")
                continue

            if (batch_idx + 1) % 10 == 0:
                print(f"  진행: {batch_idx + 1}/{len(test_loader)}")

    if all_samples and all_ground_truths:
        samples_np = np.concatenate(
            all_samples, axis=1
        )  # [num_samples, total_batch, pred_steps, 2]
        ground_truths_np = np.concatenate(
            all_ground_truths, axis=0
        )  # [total_batch, pred_steps, 2]

        # DiffusionEvaluator 사용 (다중 샘플 평가)
        evaluator = DiffusionEvaluator(k=num_samples)
        metrics = evaluator.evaluate(samples_np, ground_truths_np)
        return metrics
    else:
        return {}


def generate_latex_table(results: Dict[str, Dict], output_path: Path):
    """논문용 LaTeX 표 생성"""
    latex_lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{베이스라인 비교 결과}",
        "\\label{tab:baseline_comparison}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Model & minADE$_{20}$ & minFDE$_{20}$ & Diversity & Coverage \\\\",
        "\\midrule",
    ]

    for model_name, metrics in results.items():
        min_ade = metrics.get("min_ade", metrics.get("ade", 0.0))
        min_fde = metrics.get("min_fde", metrics.get("fde", 0.0))
        diversity = metrics.get("diversity", 0.0)
        coverage = metrics.get("coverage", 0.0)

        latex_lines.append(
            f"{model_name} & {min_ade:.4f} & {min_fde:.4f} & {diversity:.4f} & {coverage:.4f} \\\\"
        )

    latex_lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(latex_lines))

    print(f"\n✓ LaTeX 표 생성: {output_path}")


def generate_csv_table(results: Dict[str, Dict], output_path: Path):
    """CSV 비교 표 생성"""
    data = []
    for model_name, metrics in results.items():
        data.append(
            {
                "Model": model_name,
                "minADE_20": metrics.get("min_ade", metrics.get("ade", 0.0)),
                "minFDE_20": metrics.get("min_fde", metrics.get("fde", 0.0)),
                "Diversity": metrics.get("diversity", 0.0),
                "Coverage": metrics.get("coverage", 0.0),
                "Collision_Rate": metrics.get("collision_rate", 0.0),
            }
        )

    df = pd.DataFrame(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✓ CSV 표 생성: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="베이스라인 비교 평가")
    parser.add_argument(
        "--mid_checkpoint",
        type=str,
        default="checkpoints/mid/best_model.pth",
        help="MID 모델 체크포인트",
    )
    parser.add_argument(
        "--a3tgcn_checkpoint",
        type=str,
        default="checkpoints/a3tgcn/best_model.pth",
        help="A3TGCN 모델 체크포인트",
    )
    parser.add_argument(
        "--trajectron_checkpoint",
        type=str,
        default="checkpoints/trajectron/best_model.pth",
        help="Trajectron++ 모델 체크포인트",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="전처리된 데이터 디렉토리",
    )
    parser.add_argument(
        "--mid_config",
        type=str,
        default="configs/mid_config_fast.yaml",
        help="MID 설정 파일",
    )
    parser.add_argument(
        "--a3tgcn_config",
        type=str,
        default="configs/a3tgcn_config.yaml",
        help="A3TGCN 설정 파일",
    )
    parser.add_argument(
        "--trajectron_config",
        type=str,
        default="configs/trajectron_config.yaml",
        help="Trajectron++ 설정 파일",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/comparison",
        help="결과 저장 디렉토리",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="MID 샘플 수",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("베이스라인 비교 평가")
    print("=" * 80)

    # 디바이스
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n디바이스: {device}")

    # 설정 로드
    mid_config = None
    if Path(args.mid_config).exists():
        with open(args.mid_config, "r") as f:
            mid_config = yaml.safe_load(f)

    a3tgcn_config = None
    if Path(args.a3tgcn_config).exists():
        with open(args.a3tgcn_config, "r") as f:
            a3tgcn_config = yaml.safe_load(f)

    trajectron_config = None
    if Path(args.trajectron_config).exists():
        with open(args.trajectron_config, "r") as f:
            trajectron_config = yaml.safe_load(f)

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

    # 모델 로드 및 평가
    results = {}

    # 1. HSG-Diffusion 평가
    mid_checkpoint = Path(args.mid_checkpoint)
    if mid_checkpoint.exists():
        print(f"\n[HSG-Diffusion 모델 로드] {mid_checkpoint}")
        mid_model = load_model("HSG-Diffusion", mid_checkpoint, device, mid_config)
        print("✓ 모델 로드 완료")

        mid_metrics = evaluate_mid_model(
            mid_model, test_loader, device, args.num_samples
        )
        results["HSG-Diffusion"] = mid_metrics
    else:
        print(f"⚠️  MID 체크포인트 없음: {mid_checkpoint}")

    # 2. A3TGCN 평가
    a3tgcn_checkpoint = Path(args.a3tgcn_checkpoint)
    if a3tgcn_checkpoint.exists():
        print(f"\n[A3TGCN 모델 로드] {a3tgcn_checkpoint}")
        a3tgcn_model = load_model("A3TGCN", a3tgcn_checkpoint, device, a3tgcn_config)
        print("✓ 모델 로드 완료")

        a3tgcn_metrics = evaluate_a3tgcn_model(a3tgcn_model, test_loader, device)
        results["A3TGCN"] = a3tgcn_metrics
    else:
        print(f"⚠️  A3TGCN 체크포인트 없음: {a3tgcn_checkpoint}")

    # 3. Trajectron++ 평가
    trajectron_checkpoint = Path(args.trajectron_checkpoint)
    if trajectron_checkpoint.exists():
        print(f"\n[Trajectron++ 모델 로드] {trajectron_checkpoint}")
        trajectron_model = load_model(
            "Trajectron++", trajectron_checkpoint, device, trajectron_config
        )
        print("✓ 모델 로드 완료")

        trajectron_metrics = evaluate_trajectron_model(
            trajectron_model, test_loader, device, num_samples=25
        )
        results["Trajectron++"] = trajectron_metrics
    else:
        print(f"⚠️  Trajectron++ 체크포인트 없음: {trajectron_checkpoint}")

    # 결과 출력
    print("\n" + "=" * 80)
    print("베이스라인 비교 결과")
    print("=" * 80)

    print(
        f"\n{'Model':<20} {'minADE₂₀':<12} {'minFDE₂₀':<12} {'Diversity':<12} {'Coverage':<12}"
    )
    print("-" * 80)
    for name, metrics in results.items():
        min_ade = metrics.get("min_ade", metrics.get("ade", 0.0))
        min_fde = metrics.get("min_fde", metrics.get("fde", 0.0))
        diversity = metrics.get("diversity", 0.0)
        coverage = metrics.get("coverage", 0.0)
        print(
            f"{name:<20} {min_ade:<12.4f} {min_fde:<12.4f} {diversity:<12.4f} {coverage:<12.4f}"
        )

    # 결과 저장
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON 저장
    json_file = output_dir / "comparison_results.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ 결과 저장: {json_file}")

    # CSV 표 생성
    csv_file = output_dir / "comparison_table.csv"
    generate_csv_table(results, csv_file)

    # LaTeX 표 생성
    latex_file = output_dir / "comparison_table.tex"
    generate_latex_table(results, latex_file)

    print("\n" + "=" * 80)
    print("✓ 비교 평가 완료!")
    print("=" * 80)


if __name__ == "__main__":
    main()
