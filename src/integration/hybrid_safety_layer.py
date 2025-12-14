"""
Plan A/B 통합 전략: Safety Layer 구현
Plan B 안전 지표를 데이터 라벨로 활용하고, 모델 예측 결과에 Safety Layer 적용
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
from pathlib import Path
from torch_geometric.data import Data  # Type annotation용 추가

from ..evaluation.safety_metrics import (
    SafetyMetricsCalculator,
    VehicleState,
)
from ..evaluation.metrics import TrajectoryEvaluator


class SafetyLayer:
    """안전 검증 레이어: 모델 예측 결과를 안전 지표로 검증"""

    def __init__(
        self,
        ttc_threshold: float = 3.0,  # TTC 임계값 (초)
        drac_threshold: float = 5.0,  # DRAC 임계값 (m/s²)
        vehicle_radius: float = 2.5,
    ):
        """
        Args:
            ttc_threshold: 위험으로 간주할 TTC 임계값
            drac_threshold: 위험으로 간주할 DRAC 임계값
            vehicle_radius: 차량 반경
        """
        self.ttc_threshold = ttc_threshold
        self.drac_threshold = drac_threshold
        self.calculator = SafetyMetricsCalculator(vehicle_radius)

    def validate_prediction(
        self,
        predicted_trajectories: np.ndarray,
        current_states: np.ndarray,
        timestep: float = 0.1,
    ) -> Dict:
        """
        예측된 궤적의 안전성을 검증합니다.

        Args:
            predicted_trajectories: 예측 궤적 [num_agents, pred_steps, 2]
            current_states: 현재 상태 [num_agents, state_dim] (x, y, vx, vy 포함)
            timestep: 시간 간격 (초)

        Returns:
            검증 결과 딕셔너리
        """
        num_agents, pred_steps, _ = predicted_trajectories.shape

        violations = []
        risk_scores = []

        # 각 에이전트 쌍에 대해 검증
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                traj_i = predicted_trajectories[i]  # [pred_steps, 2]
                traj_j = predicted_trajectories[j]  # [pred_steps, 2]

                # 현재 상태에서 속도 추출
                vx_i, vy_i = current_states[i, 2], current_states[i, 3]
                vx_j, vy_j = current_states[j, 2], current_states[j, 3]

                # 각 시점에서 TTC 계산
                min_ttc = float("inf")
                min_ttc_time = None

                for t in range(pred_steps):
                    # 해당 시점의 위치
                    pos_i = traj_i[t]
                    pos_j = traj_j[t]

                    # 속도는 현재 속도로 가정 (실제로는 예측된 속도 사용 가능)
                    vehicle_i = VehicleState(x=pos_i[0], y=pos_i[1], vx=vx_i, vy=vy_i)
                    vehicle_j = VehicleState(x=pos_j[0], y=pos_j[1], vx=vx_j, vy=vy_j)

                    # TTC 계산
                    ttc = self.calculator.calculate_2d_ttc(vehicle_i, vehicle_j)

                    if ttc is not None and ttc < min_ttc:
                        min_ttc = ttc
                        min_ttc_time = t * timestep

                # 위험도 평가
                if min_ttc < self.ttc_threshold:
                    # DRAC 계산
                    if min_ttc_time is not None:
                        t_idx = int(min_ttc_time / timestep)
                        if t_idx < pred_steps:
                            pos_i = traj_i[t_idx]
                            pos_j = traj_j[t_idx]

                            vehicle_i = VehicleState(
                                x=pos_i[0], y=pos_i[1], vx=vx_i, vy=vy_i
                            )
                            vehicle_j = VehicleState(
                                x=pos_j[0], y=pos_j[1], vx=vx_j, vy=vy_j
                            )

                            drac = self.calculator.calculate_drac(
                                vehicle_i, vehicle_j, min_ttc
                            )

                            risk_score = self._calculate_risk_score(min_ttc, drac)

                            violations.append(
                                {
                                    "agent1": i,
                                    "agent2": j,
                                    "ttc": min_ttc,
                                    "drac": drac,
                                    "time": min_ttc_time,
                                    "risk_score": risk_score,
                                    "is_critical": risk_score > 0.7,
                                }
                            )

                            risk_scores.append(risk_score)

        return {
            "num_violations": len(violations),
            "violations": violations,
            "max_risk_score": max(risk_scores) if risk_scores else 0.0,
            "avg_risk_score": np.mean(risk_scores) if risk_scores else 0.0,
            "is_safe": len(violations) == 0,
        }

    def _calculate_risk_score(self, ttc: float, drac: Optional[float]) -> float:
        """
        위험도 점수 계산 (0~1)

        Args:
            ttc: Time-to-Collision
            drac: Deceleration Rate to Avoid Collision

        Returns:
            위험도 점수 (0: 안전, 1: 매우 위험)
        """
        # TTC 기반 위험도 (낮을수록 위험)
        ttc_risk = max(0, 1.0 - (ttc / self.ttc_threshold))

        # DRAC 기반 위험도
        if drac is not None and drac != float("inf"):
            drac_risk = min(1.0, drac / self.drac_threshold)
        else:
            drac_risk = 1.0  # 무한대 = 매우 위험

        # 종합 위험도 (가중 평균)
        risk_score = 0.6 * ttc_risk + 0.4 * drac_risk

        return min(1.0, risk_score)

    def filter_unsafe_predictions(
        self,
        predicted_trajectories: np.ndarray,
        current_states: np.ndarray,
        threshold: float = 0.7,
    ) -> Tuple[np.ndarray, List[int]]:
        """
        위험한 예측을 필터링합니다.

        Args:
            predicted_trajectories: 예측 궤적
            current_states: 현재 상태
            threshold: 위험도 임계값

        Returns:
            (필터링된 궤적, 제거된 에이전트 인덱스)
        """
        validation = self.validate_prediction(predicted_trajectories, current_states)

        # 위험도가 높은 에이전트 식별
        unsafe_agents = set()
        for violation in validation["violations"]:
            if violation["risk_score"] > threshold:
                unsafe_agents.add(violation["agent1"])
                unsafe_agents.add(violation["agent2"])

        # 안전한 에이전트만 선택
        safe_indices = [
            i for i in range(len(predicted_trajectories)) if i not in unsafe_agents
        ]

        if len(safe_indices) == 0:
            # 모두 위험하면 원본 반환
            return predicted_trajectories, []

        filtered_trajectories = predicted_trajectories[safe_indices]

        return filtered_trajectories, list(unsafe_agents)


class SafetyAwareLoss(nn.Module):
    """안전 지표를 반영한 손실 함수"""

    def __init__(
        self,
        base_loss: nn.Module,
        safety_weight: float = 0.1,
        ttc_threshold: float = 3.0,
    ):
        """
        Args:
            base_loss: 기본 손실 함수 (MSE 등)
            safety_weight: 안전 손실 가중치
            ttc_threshold: TTC 임계값
        """
        super(SafetyAwareLoss, self).__init__()
        self.base_loss = base_loss
        self.safety_weight = safety_weight
        self.ttc_threshold = ttc_threshold
        self.calculator = SafetyMetricsCalculator()

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        current_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            predicted: 예측 궤적 [batch, num_agents, pred_steps, 2]
            target: 실제 궤적 [batch, num_agents, pred_steps, 2]
            current_states: 현재 상태 [batch, num_agents, state_dim]
        """
        # 기본 손실
        base_loss_value = self.base_loss(predicted, target)

        # 안전 손실 (현재 상태가 있는 경우)
        if current_states is not None:
            safety_loss = self._compute_safety_loss(predicted, current_states)
            total_loss = base_loss_value + self.safety_weight * safety_loss
        else:
            total_loss = base_loss_value

        return total_loss

    def _compute_safety_loss(
        self, predicted: torch.Tensor, current_states: torch.Tensor
    ) -> torch.Tensor:
        """안전 손실 계산"""
        batch_size, num_agents, pred_steps, _ = predicted.shape

        safety_penalties = []

        for b in range(batch_size):
            batch_pred = predicted[b].detach().cpu().numpy()
            batch_states = current_states[b].detach().cpu().numpy()

            # 각 에이전트 쌍에 대해 TTC 계산
            for i in range(num_agents):
                for j in range(i + 1, num_agents):
                    traj_i = batch_pred[i]
                    traj_j = batch_pred[j]

                    # 첫 시점에서 TTC 계산
                    pos_i = traj_i[0]
                    pos_j = traj_j[0]
                    vx_i, vy_i = batch_states[i, 2], batch_states[i, 3]
                    vx_j, vy_j = batch_states[j, 2], batch_states[j, 3]

                    vehicle_i = VehicleState(x=pos_i[0], y=pos_i[1], vx=vx_i, vy=vy_i)
                    vehicle_j = VehicleState(x=pos_j[0], y=pos_j[1], vx=vx_j, vy=vy_j)

                    ttc = self.calculator.calculate_2d_ttc(vehicle_i, vehicle_j)

                    if ttc is not None and ttc < self.ttc_threshold:
                        # TTC가 낮을수록 큰 페널티
                        penalty = (self.ttc_threshold - ttc) / self.ttc_threshold
                        safety_penalties.append(penalty)

        if safety_penalties:
            safety_loss = torch.tensor(
                np.mean(safety_penalties),
                dtype=predicted.dtype,
                device=predicted.device,
            )
        else:
            safety_loss = torch.tensor(
                0.0, dtype=predicted.dtype, device=predicted.device
            )

        return safety_loss


class HybridPredictor:
    """Plan A와 Plan B를 통합한 예측 시스템"""

    def __init__(
        self, model: nn.Module, safety_layer: SafetyLayer, device: torch.device
    ):
        """
        Args:
            model: 예측 모델 (Plan A)
            safety_layer: 안전 검증 레이어 (Plan B)
            device: 디바이스
        """
        self.model = model.to(device)
        self.safety_layer = safety_layer
        self.device = device

    def predict_with_safety_check(
        self,
        obs_data: torch.Tensor,
        graph_data: Optional[Data] = None,
        return_risk: bool = False,
    ) -> Dict:
        """
        안전 검증을 포함한 예측 수행

        Args:
            obs_data: 관측 데이터
            graph_data: 그래프 데이터 (선택사항)
            return_risk: 위험도 정보 반환 여부

        Returns:
            예측 결과 딕셔너리
        """
        self.model.eval()

        with torch.no_grad():
            # 모델 예측 (Plan A)
            if graph_data is not None:
                graph_data = graph_data.to(self.device)
                if (
                    hasattr(graph_data, "edge_weight")
                    and graph_data.edge_weight is not None
                ):
                    predicted = self.model(
                        graph_data.x, graph_data.edge_index, graph_data.edge_weight
                    )
                else:
                    predicted = self.model(graph_data.x, graph_data.edge_index)
            else:
                predicted = self.model(obs_data)

            # CPU로 이동 및 numpy 변환
            predicted_np = predicted.cpu().numpy()
            obs_data_np = obs_data.cpu().numpy()

            # 현재 상태 추출 (마지막 관측 시점)
            if obs_data_np.ndim == 3:  # [batch, time, features]
                current_states = obs_data_np[:, -1, :]  # 마지막 시점
            else:
                current_states = obs_data_np[-1, :]  # 단일 시퀀스

            # 안전 검증 (Plan B)
            if predicted_np.ndim == 3:  # [batch, agents, pred_steps, 2]
                # 배치 처리
                validated_predictions = []
                risk_info = []

                for b in range(predicted_np.shape[0]):
                    validation = self.safety_layer.validate_prediction(
                        predicted_np[b],
                        (
                            current_states[b]
                            if current_states.ndim > 1
                            else current_states
                        ),
                    )

                    # 위험한 예측 필터링
                    filtered, unsafe = self.safety_layer.filter_unsafe_predictions(
                        predicted_np[b],
                        (
                            current_states[b]
                            if current_states.ndim > 1
                            else current_states
                        ),
                    )

                    validated_predictions.append(filtered)
                    if return_risk:
                        risk_info.append(validation)

                result = {
                    "predictions": np.array(validated_predictions),
                    "is_safe": (
                        all(v["is_safe"] for v in risk_info) if risk_info else True
                    ),
                }

                if return_risk:
                    result["risk_info"] = risk_info
            else:
                # 단일 예측
                validation = self.safety_layer.validate_prediction(
                    predicted_np, current_states
                )

                filtered, unsafe = self.safety_layer.filter_unsafe_predictions(
                    predicted_np, current_states
                )

                result = {
                    "predictions": filtered,
                    "is_safe": validation["is_safe"],
                    "unsafe_agents": unsafe,
                }

                if return_risk:
                    result["risk_info"] = validation

        return result

    def post_process_with_safety(
        self, predicted: np.ndarray, current_states: np.ndarray
    ) -> np.ndarray:
        """
        예측 결과를 안전하게 후처리

        Args:
            predicted: 예측 궤적
            current_states: 현재 상태

        Returns:
            후처리된 예측 궤적
        """
        validation = self.safety_layer.validate_prediction(predicted, current_states)

        # 위험한 예측이 있으면 보정
        if not validation["is_safe"]:
            # 간단한 보정: 위험한 에이전트의 속도를 감소
            corrected = predicted.copy()

            for violation in validation["violations"]:
                if violation["is_critical"]:
                    agent_idx = violation["agent1"]
                    # 속도 감소 (안전한 방향으로)
                    # 실제로는 더 정교한 보정 알고리즘 필요
                    corrected[agent_idx] = predicted[agent_idx] * 0.9

            return corrected

        return predicted


def create_safety_labels(
    trajectory_data: pd.DataFrame, output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Plan B 안전 지표를 데이터 라벨로 생성

    Args:
        trajectory_data: 궤적 데이터
        output_path: 저장 경로

    Returns:
        안전 라벨이 추가된 DataFrame
    """
    calculator = SafetyMetricsCalculator()

    # 프레임별로 안전 지표 계산
    frame_ids = sorted(trajectory_data["frame_id"].unique())

    safety_labels = []

    for frame_id in frame_ids:
        frame_data = trajectory_data[trajectory_data["frame_id"] == frame_id]

        if len(frame_data) < 2:
            continue

        # 각 차량 쌍에 대해 TTC 계산
        for i, row_i in frame_data.iterrows():
            for j, row_j in frame_data.iterrows():
                if i >= j:
                    continue

                vehicle_i = VehicleState(
                    x=row_i["x"],
                    y=row_i["y"],
                    vx=row_i.get("vx", 0.0),
                    vy=row_i.get("vy", 0.0),
                )
                vehicle_j = VehicleState(
                    x=row_j["x"],
                    y=row_j["y"],
                    vx=row_j.get("vx", 0.0),
                    vy=row_j.get("vy", 0.0),
                )

                ttc = calculator.calculate_2d_ttc(vehicle_i, vehicle_j)
                drac = calculator.calculate_drac(vehicle_i, vehicle_j, ttc)

                # 위험도 라벨
                is_risky = (ttc is not None and ttc < 3.0) or (
                    drac is not None and drac > 5.0
                )

                safety_labels.append(
                    {
                        "frame_id": frame_id,
                        "agent1_id": row_i["track_id"],
                        "agent2_id": row_j["track_id"],
                        "ttc": ttc,
                        "drac": drac,
                        "is_risky": is_risky,
                        "risk_level": "high" if is_risky else "low",
                    }
                )

    labels_df = pd.DataFrame(safety_labels)

    if output_path:
        labels_df.to_csv(output_path, index=False)
        print(f"안전 라벨 저장: {output_path}")

    return labels_df


def main():
    """테스트용 메인 함수"""
    # Safety Layer 테스트
    safety_layer = SafetyLayer(ttc_threshold=3.0, drac_threshold=5.0)

    # 더미 예측 데이터
    num_agents = 3
    pred_steps = 50

    predicted = np.random.randn(num_agents, pred_steps, 2) * 0.5
    current_states = np.random.randn(num_agents, 4)  # x, y, vx, vy

    # 검증
    validation = safety_layer.validate_prediction(predicted, current_states)

    print("안전 검증 결과:")
    print(f"  위반 수: {validation['num_violations']}")
    print(f"  최대 위험도: {validation['max_risk_score']:.4f}")
    print(f"  안전 여부: {validation['is_safe']}")


if __name__ == "__main__":
    main()
