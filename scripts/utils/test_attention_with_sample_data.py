"""
어텐션 가중치 계산 테스트 (샘플 데이터)
PyTorch 없이 샘플 데이터로 어텐션 가중치 계산 로직 테스트
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_sample_data():
    """샘플 데이터 로드"""
    print("=" * 80)
    print("샘플 데이터 로드")
    print("=" * 80)

    # 1. 원본 샘플 데이터
    sample_file = project_root / "data/interaction-dataset-repo/recorded_trackfiles/.TestScenarioForScripts/vehicle_tracks_000.csv"

    if not sample_file.exists():
        print(f"❌ 샘플 파일을 찾을 수 없습니다: {sample_file}")
        return None, None

    df = pd.read_csv(sample_file)
    print(f"\n✓ 원본 데이터 로드:")
    print(f"  총 행 수: {len(df)}")
    print(f"  트랙 수: {df['track_id'].nunique()}")
    print(f"  프레임 범위: {df['frame_id'].min()} ~ {df['frame_id'].max()}")
    print(f"  컬럼: {list(df.columns)}")

    # 2. 전처리된 데이터 확인
    processed_file = project_root / "data/processed/vehicle_tracks_000_processed.pkl"

    if processed_file.exists():
        print(f"\n✓ 전처리된 데이터 발견: {processed_file}")
        with open(processed_file, 'rb') as f:
            processed_data = pickle.load(f)
        windows = processed_data.get('windows', [])
        print(f"  윈도우 수: {len(windows)}")
        return df, windows
    else:
        print(f"\n⚠ 전처리된 데이터가 없습니다. 원본 데이터만 사용합니다.")
        return df, None


def calculate_distance_based_attention(positions: np.ndarray, method='euclidean'):
    """
    거리 기반 어텐션 가중치 계산

    Args:
        positions: 노드 위치 배열 [num_nodes, 2]
        method: 거리 계산 방법 ('euclidean', 'manhattan')

    Returns:
        어텐션 가중치 행렬 [num_nodes, num_nodes]
    """
    num_nodes = len(positions)

    # 거리 행렬 계산
    distances = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(num_nodes):
            if method == 'euclidean':
                distances[i, j] = np.linalg.norm(positions[i] - positions[j])
            elif method == 'manhattan':
                distances[i, j] = np.sum(np.abs(positions[i] - positions[j]))

    # 거리를 어텐션 점수로 변환 (가까울수록 높은 점수)
    # 거리의 역수를 사용하되, 자기 자신은 제외
    attention_scores = np.zeros_like(distances)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                attention_scores[i, j] = 0  # 자기 자신은 0
            else:
                # 거리가 가까울수록 높은 점수 (역수 사용)
                attention_scores[i, j] = 1.0 / (distances[i, j] + 1e-6)

    # Softmax 적용하여 어텐션 가중치로 변환
    attention_weights = np.exp(attention_scores - np.max(attention_scores, axis=1, keepdims=True))
    attention_weights = attention_weights / np.sum(attention_weights, axis=1, keepdims=True)

    # 자기 자신의 가중치는 0으로 유지
    np.fill_diagonal(attention_weights, 0.0)
    # 정규화 재적용
    row_sums = np.sum(attention_weights, axis=1, keepdims=True)
    attention_weights = attention_weights / (row_sums + 1e-6)

    return attention_weights, distances


def calculate_velocity_based_attention(positions: np.ndarray, velocities: np.ndarray):
    """
    속도 기반 어텐션 가중치 계산
    유사한 속도와 방향을 가진 차량에 더 높은 가중치 부여

    Args:
        positions: 노드 위치 [num_nodes, 2]
        velocities: 노드 속도 [num_nodes, 2]

    Returns:
        어텐션 가중치 행렬 [num_nodes, num_nodes]
    """
    num_nodes = len(positions)

    # 속도 유사도 계산
    velocity_similarity = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                velocity_similarity[i, j] = 0
            else:
                # 속도 벡터 간 코사인 유사도
                v_i = velocities[i]
                v_j = velocities[j]

                dot_product = np.dot(v_i, v_j)
                norm_i = np.linalg.norm(v_i)
                norm_j = np.linalg.norm(v_j)

                if norm_i > 1e-6 and norm_j > 1e-6:
                    cosine_sim = dot_product / (norm_i * norm_j)
                    velocity_similarity[i, j] = (cosine_sim + 1) / 2  # [-1, 1] -> [0, 1]
                else:
                    velocity_similarity[i, j] = 0

    # 거리 정보도 결합
    distance_weights, _ = calculate_distance_based_attention(positions)

    # 속도 유사도와 거리 가중치 결합
    combined_attention = velocity_similarity * distance_weights

    # Softmax 적용
    attention_weights = np.exp(combined_attention - np.max(combined_attention, axis=1, keepdims=True))
    attention_weights = attention_weights / (np.sum(attention_weights, axis=1, keepdims=True) + 1e-6)

    return attention_weights


def test_attention_calculation():
    """어텐션 가중치 계산 테스트"""
    print("\n" + "=" * 80)
    print("어텐션 가중치 계산 테스트")
    print("=" * 80)

    # 데이터 로드
    df, windows = load_sample_data()

    if df is None:
        return

    # 첫 번째 트랙의 데이터 사용
    first_track = df[df['track_id'] == df['track_id'].iloc[0]].sort_values('frame_id')

    print(f"\n사용할 트랙 정보:")
    print(f"  트랙 ID: {first_track['track_id'].iloc[0]}")
    print(f"  프레임 수: {len(first_track)}")

    # 특정 시점의 데이터 선택 (예: 중간 시점)
    mid_frame = len(first_track) // 2
    frame_data = first_track.iloc[mid_frame:mid_frame+10]  # 10개 프레임 사용

    print(f"\n선택한 프레임:")
    print(f"  프레임 범위: {frame_data['frame_id'].min()} ~ {frame_data['frame_id'].max()}")
    print(f"  차량 수: {len(frame_data)}")

    # 위치와 속도 추출
    positions = frame_data[['x', 'y']].values
    velocities = frame_data[['vx', 'vy']].values

    print(f"\n데이터 통계:")
    print(f"  위치 범위: X [{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}], "
          f"Y [{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}]")
    print(f"  속도 범위: VX [{velocities[:, 0].min():.2f}, {velocities[:, 0].max():.2f}], "
          f"VY [{velocities[:, 1].min():.2f}, {velocities[:, 1].max():.2f}]")

    # 1. 거리 기반 어텐션 계산
    print(f"\n{'='*80}")
    print("1. 거리 기반 어텐션 가중치 계산")
    print(f"{'='*80}")

    distance_attention, distances = calculate_distance_based_attention(positions)

    print(f"\n어텐션 가중치 통계:")
    print(f"  형태: {distance_attention.shape}")
    print(f"  평균: {np.mean(distance_attention):.4f}")
    print(f"  최대: {np.max(distance_attention):.4f}")
    print(f"  최소: {np.min(distance_attention):.4f}")
    print(f"  표준편차: {np.std(distance_attention):.4f}")

    # 가장 높은 어텐션을 받는 노드 쌍
    max_idx = np.unravel_index(np.argmax(distance_attention), distance_attention.shape)
    print(f"\n최고 어텐션:")
    print(f"  노드 {max_idx[0]} -> 노드 {max_idx[1]}: {distance_attention[max_idx]:.4f}")
    print(f"  거리: {distances[max_idx]:.2f}")

    # 각 노드가 받는 총 어텐션
    incoming_attention = np.sum(distance_attention, axis=0)
    most_attended_node = np.argmax(incoming_attention)
    print(f"\n가장 많은 어텐션을 받는 노드: {most_attended_node} (총 {incoming_attention[most_attended_node]:.4f})")

    # 2. 속도 기반 어텐션 계산
    print(f"\n{'='*80}")
    print("2. 속도 기반 어텐션 가중치 계산")
    print(f"{'='*80}")

    velocity_attention = calculate_velocity_based_attention(positions, velocities)

    print(f"\n어텐션 가중치 통계:")
    print(f"  형태: {velocity_attention.shape}")
    print(f"  평균: {np.mean(velocity_attention):.4f}")
    print(f"  최대: {np.max(velocity_attention):.4f}")
    print(f"  최소: {np.min(velocity_attention):.4f}")

    # 3. 시각화 (간단한 텍스트 출력)
    print(f"\n{'='*80}")
    print("3. 어텐션 가중치 행렬 (상위 5x5)")
    print(f"{'='*80}")

    size = min(5, len(distance_attention))
    print("\n거리 기반 어텐션 (상위 5x5):")
    print(distance_attention[:size, :size])

    print("\n속도 기반 어텐션 (상위 5x5):")
    print(velocity_attention[:size, :size])

    # 4. 결과 저장
    results = {
        'positions': positions,
        'velocities': velocities,
        'distance_attention': distance_attention,
        'velocity_attention': velocity_attention,
        'distances': distances
    }

    output_file = project_root / "results/attention_test_results.npz"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_file, **results)

    print(f"\n{'='*80}")
    print("✓ 테스트 완료")
    print(f"{'='*80}")
    print(f"\n결과 저장: {output_file}")
    print(f"\n다음 단계:")
    print(f"  1. 시각화 모듈 구현")
    print(f"  2. A3TGCN 모델과 통합")
    print(f"  3. 실제 학습 데이터로 테스트")

    return results


if __name__ == "__main__":
    results = test_attention_calculation()

    if results:
        print("\n✓ 샘플 데이터로 어텐션 가중치 계산 성공!")
        print("  시각화 모듈 구현 준비 완료")

