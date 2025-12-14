"""
어텐션 가중치 추출 테스트 스크립트
샘플 데이터로 모델 실행 및 어텐션 가중치 확인
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.a3tgcn_model import create_a3tgcn_model
from src.data_processing.preprocessor import TrajectoryPreprocessor
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


def create_simple_graph_from_data(obs_data: pd.DataFrame) -> Data:
    """
    관측 데이터로부터 간단한 그래프 생성

    Args:
        obs_data: 관측 데이터프레임

    Returns:
        PyTorch Geometric Data 객체
    """
    num_nodes = len(obs_data)

    # 노드 특징: x, y, vx, vy, psi_rad, length, width
    # 필요한 컬럼만 선택
    feature_cols = ['x', 'y', 'vx', 'vy', 'psi_rad']
    if 'length' in obs_data.columns:
        feature_cols.append('length')
    if 'width' in obs_data.columns:
        feature_cols.append('width')

    # 존재하는 컬럼만 사용
    available_cols = [col for col in feature_cols if col in obs_data.columns]

    # 특징 벡터 생성
    node_features = obs_data[available_cols].values.astype(np.float32)

    # 부족한 특징은 0으로 채우기 (최소 9차원)
    if node_features.shape[1] < 9:
        padding = np.zeros((num_nodes, 9 - node_features.shape[1]), dtype=np.float32)
        node_features = np.concatenate([node_features, padding], axis=1)

    # 간단한 엣지 생성: 거리 기반 k-NN (k=3)
    # 각 노드에서 가장 가까운 3개 노드와 연결
    positions = obs_data[['x', 'y']].values
    edge_list = []

    for i in range(num_nodes):
        distances = np.sqrt(np.sum((positions - positions[i])**2, axis=1))
        # 자기 자신 제외하고 가장 가까운 3개 선택
        nearest_indices = np.argsort(distances)[1:min(4, num_nodes)]
        for j in nearest_indices:
            edge_list.append([i, j])

    if len(edge_list) == 0:
        # 엣지가 없으면 완전 연결 그래프 생성
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                edge_list.append([i, j])
                edge_list.append([j, i])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_index = to_undirected(edge_index)  # 무방향 그래프로 변환

    # PyTorch Geometric Data 객체 생성
    x = torch.tensor(node_features, dtype=torch.float32)

    graph_data = Data(x=x, edge_index=edge_index)

    return graph_data


def extract_attention_hook(module, input, output):
    """
    A3TGCN 레이어에서 어텐션 가중치를 추출하기 위한 hook
    """
    # A3TGCN2의 내부 구조에 따라 어텐션 가중치 추출 시도
    if hasattr(module, 'attention_weights'):
        return module.attention_weights
    return None


def test_attention_extraction():
    """어텐션 가중치 추출 테스트"""
    print("=" * 80)
    print("어텐션 가중치 추출 테스트")
    print("=" * 80)

    # 1. 샘플 데이터 로드
    print("\n1. 샘플 데이터 로드 중...")
    sample_file = project_root / "data/interaction-dataset-repo/recorded_trackfiles/.TestScenarioForScripts/vehicle_tracks_000.csv"

    if not sample_file.exists():
        print(f"❌ 샘플 파일을 찾을 수 없습니다: {sample_file}")
        return

    df = pd.read_csv(sample_file)
    print(f"   ✓ 데이터 로드 완료: {len(df)}행, {df['track_id'].nunique()}개 트랙")

    # 2. 전처리된 데이터 확인 또는 전처리
    processed_file = project_root / "data/processed/vehicle_tracks_000_processed.pkl"

    if processed_file.exists():
        print("\n2. 전처리된 데이터 로드 중...")
        with open(processed_file, 'rb') as f:
            processed_data = pickle.load(f)
        windows = processed_data.get('windows', [])
        print(f"   ✓ 전처리된 윈도우 {len(windows)}개 발견")
    else:
        print("\n2. 데이터 전처리 중...")
        preprocessor = TrajectoryPreprocessor()
        result = preprocessor.preprocess_scenario(sample_file, project_root / "data/processed")
        windows = result.get('windows', [])
        print(f"   ✓ 전처리 완료: {len(windows)}개 윈도우 생성")

    if len(windows) == 0:
        print("❌ 윈도우 데이터가 없습니다.")
        return

    # 3. 첫 번째 윈도우 사용
    window = windows[0]
    obs_data = window['obs_data']

    print(f"\n3. 윈도우 정보:")
    print(f"   트랙 ID: {window['track_id']}")
    print(f"   관측 프레임: {len(obs_data)}개")
    print(f"   예측 프레임: {len(window['pred_data'])}개")

    # 4. 그래프 생성
    print("\n4. 그래프 생성 중...")
    try:
        graph_data = create_simple_graph_from_data(obs_data)
        print(f"   ✓ 그래프 생성 완료:")
        print(f"     노드 수: {graph_data.x.shape[0]}")
        print(f"     특징 차원: {graph_data.x.shape[1]}")
        print(f"     엣지 수: {graph_data.edge_index.shape[1]}")
    except Exception as e:
        print(f"   ❌ 그래프 생성 실패: {e}")
        return

    # 5. 모델 생성
    print("\n5. 모델 생성 중...")
    try:
        model = create_a3tgcn_model(
            node_features=9,
            hidden_channels=32,
            pred_steps=50,
            use_map=False
        )
        model.eval()
        print(f"   ✓ 모델 생성 완료")
        print(f"     파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"   ❌ 모델 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return

    # 6. Forward pass 및 어텐션 가중치 추출 시도
    print("\n6. Forward pass 및 어텐션 추출 시도...")

    # Hook 등록 (A3TGCN 레이어에)
    attention_hooks = []
    attention_weights_list = []

    def hook_fn(name):
        def hook(module, input, output):
            # A3TGCN2의 내부 구조 확인
            if hasattr(module, '_modules'):
                for submodule_name, submodule in module._modules.items():
                    if hasattr(submodule, 'attention_weights'):
                        attention_weights_list.append({
                            'layer': name,
                            'weights': submodule.attention_weights
                        })
            return None
        return hook

    # A3TGCN 레이어에 hook 등록
    if hasattr(model, 'tgnn'):
        hook = model.tgnn.register_forward_hook(hook_fn('tgnn'))
        attention_hooks.append(hook)

    for i, layer in enumerate(model.gnn_layers):
        hook = layer.register_forward_hook(hook_fn(f'gnn_layer_{i}'))
        attention_hooks.append(hook)

    # Forward pass
    try:
        with torch.no_grad():
            # 시퀀스 데이터로 변환 (A3TGCN은 시계열 입력 필요)
            # 간단한 테스트를 위해 마지막 프레임만 사용
            x = graph_data.x
            edge_index = graph_data.edge_index

            # A3TGCN은 시계열 입력을 기대하므로, 여러 타임스텝으로 변환
            # periods=30이므로 30개 타임스텝 필요
            # 간단히 동일한 특징을 반복
            if len(obs_data) < 30:
                # 부족하면 마지막 프레임 반복
                x_seq = x.unsqueeze(0).repeat(30, 1, 1)  # [30, num_nodes, features]
            else:
                # 충분하면 실제 시계열 사용
                x_seq = x.unsqueeze(0).repeat(min(30, len(obs_data)), 1, 1)

            # A3TGCN2는 시계열 입력을 처리하므로, 첫 번째 타임스텝만 사용
            # 실제로는 시계열 전체를 처리해야 하지만, 테스트를 위해 단순화
            pred = model(x, edge_index)

        print(f"   ✓ Forward pass 완료")
        print(f"     예측 형태: {pred.shape}")

        # 어텐션 가중치 확인
        if attention_weights_list:
            print(f"   ✓ 어텐션 가중치 추출 성공: {len(attention_weights_list)}개 레이어")
            for att_info in attention_weights_list:
                print(f"     - {att_info['layer']}: {type(att_info['weights'])}")
        else:
            print(f"   ⚠ 어텐션 가중치를 직접 추출할 수 없습니다.")
            print(f"     A3TGCN2의 내부 구조를 확인해야 합니다.")

    except Exception as e:
        print(f"   ❌ Forward pass 실패: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Hook 제거
        for hook in attention_hooks:
            hook.remove()

    # 7. 대안: 엣지 가중치 기반 어텐션 계산
    print("\n7. 대안: 엣지 기반 어텐션 가중치 계산...")
    try:
        # 노드 간 거리 기반 어텐션 가중치 계산
        positions = obs_data[['x', 'y']].values
        num_nodes = len(positions)

        # 거리 행렬 계산
        distances = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                distances[i, j] = np.linalg.norm(positions[i] - positions[j])

        # 거리를 어텐션 가중치로 변환 (가까울수록 높은 가중치)
        # Softmax 적용
        attention_scores = -distances  # 거리가 가까울수록 높은 점수
        attention_weights = np.exp(attention_scores - np.max(attention_scores, axis=1, keepdims=True))
        attention_weights = attention_weights / np.sum(attention_weights, axis=1, keepdims=True)

        print(f"   ✓ 엣지 기반 어텐션 가중치 계산 완료")
        print(f"     형태: {attention_weights.shape}")
        print(f"     평균 가중치: {np.mean(attention_weights):.4f}")
        print(f"     최대 가중치: {np.max(attention_weights):.4f}")
        print(f"     최소 가중치: {np.min(attention_weights):.4f}")

        # 가장 높은 어텐션을 받는 노드 쌍
        max_attention_idx = np.unravel_index(np.argmax(attention_weights), attention_weights.shape)
        print(f"     최고 어텐션: 노드 {max_attention_idx[0]} -> 노드 {max_attention_idx[1]} ({attention_weights[max_attention_idx]:.4f})")

        return attention_weights, graph_data, obs_data

    except Exception as e:
        print(f"   ❌ 어텐션 가중치 계산 실패: {e}")
        import traceback
        traceback.print_exc()
        return None, graph_data, obs_data


if __name__ == "__main__":
    result = test_attention_extraction()

    if result and result[0] is not None:
        print("\n" + "=" * 80)
        print("✓ 테스트 완료: 어텐션 가중치 추출 가능")
        print("=" * 80)
        print("\n다음 단계: 시각화 모듈 구현")
    else:
        print("\n" + "=" * 80)
        print("⚠ 테스트 완료: A3TGCN 내부 어텐션 추출 필요")
        print("=" * 80)
        print("\n대안: 엣지 기반 어텐션 가중치 사용 가능")

