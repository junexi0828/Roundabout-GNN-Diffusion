"""
어텐션 가중치 간단한 시각화 (샘플 데이터)
matplotlib을 사용한 기본 시각화
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 결과 로드
results_file = Path(__file__).parent.parent / "results/attention_test_results.npz"

if not results_file.exists():
    print(f"결과 파일을 찾을 수 없습니다: {results_file}")
    print("먼저 test_attention_with_sample_data.py를 실행하세요.")
    exit(1)

data = np.load(results_file, allow_pickle=True)
positions = data['positions']
velocities = data['velocities']
distance_attention = data['distance_attention']
velocity_attention = data['velocity_attention']

print("=" * 80)
print("어텐션 가중치 시각화 (샘플 데이터)")
print("=" * 80)

# 1. 위치 및 어텐션 히트맵
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1-1. 노드 위치 시각화
ax = axes[0, 0]
ax.scatter(positions[:, 0], positions[:, 1], s=100, c='blue', alpha=0.6, edgecolors='black')
for i, (x, y) in enumerate(positions):
    ax.annotate(f'{i}', (x, y), fontsize=8, ha='center', va='center')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('노드 위치')
ax.grid(True, alpha=0.3)

# 1-2. 거리 기반 어텐션 히트맵
ax = axes[0, 1]
im = ax.imshow(distance_attention, cmap='YlOrRd', aspect='auto')
ax.set_xlabel('Target Node')
ax.set_ylabel('Source Node')
ax.set_title('거리 기반 어텐션 가중치')
plt.colorbar(im, ax=ax)

# 1-3. 속도 기반 어텐션 히트맵
ax = axes[1, 0]
im = ax.imshow(velocity_attention, cmap='Blues', aspect='auto')
ax.set_xlabel('Target Node')
ax.set_ylabel('Source Node')
ax.set_title('속도 기반 어텐션 가중치')
plt.colorbar(im, ax=ax)

# 1-4. 어텐션 가중치 비교
ax = axes[1, 1]
x = np.arange(len(positions))
distance_incoming = np.sum(distance_attention, axis=0)
velocity_incoming = np.sum(velocity_attention, axis=0)
ax.bar(x - 0.2, distance_incoming, 0.4, label='거리 기반', alpha=0.7)
ax.bar(x + 0.2, velocity_incoming, 0.4, label='속도 기반', alpha=0.7)
ax.set_xlabel('Node ID')
ax.set_ylabel('총 어텐션 가중치')
ax.set_title('각 노드가 받는 총 어텐션')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
output_file = Path(__file__).parent.parent / "results/attention_visualization_sample.png"
output_file.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✓ 시각화 저장: {output_file}")

# 2. 어텐션 그래프 시각화
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# 2-1. 거리 기반 어텐션 그래프
ax = axes[0]
ax.scatter(positions[:, 0], positions[:, 1], s=200, c='blue', alpha=0.7, edgecolors='black', zorder=3)
for i, (x, y) in enumerate(positions):
    ax.annotate(f'{i}', (x, y), fontsize=10, ha='center', va='center', color='white', weight='bold')

# 상위 어텐션 엣지만 그리기
threshold = np.percentile(distance_attention[distance_attention > 0], 80)
for i in range(len(positions)):
    for j in range(len(positions)):
        if i != j and distance_attention[i, j] > threshold:
            ax.plot([positions[i, 0], positions[j, 0]],
                   [positions[i, 1], positions[j, 1]],
                   'r-', alpha=distance_attention[i, j] * 2, linewidth=2, zorder=1)

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title(f'거리 기반 어텐션 그래프 (상위 20% 엣지)')
ax.grid(True, alpha=0.3)

# 2-2. 속도 기반 어텐션 그래프
ax = axes[1]
ax.scatter(positions[:, 0], positions[:, 1], s=200, c='green', alpha=0.7, edgecolors='black', zorder=3)
for i, (x, y) in enumerate(positions):
    ax.annotate(f'{i}', (x, y), fontsize=10, ha='center', va='center', color='white', weight='bold')

# 속도 벡터 표시
for i, (pos, vel) in enumerate(zip(positions, velocities)):
    if np.linalg.norm(vel) > 0.1:
        ax.arrow(pos[0], pos[1], vel[0]*0.5, vel[1]*0.5,
                head_width=0.3, head_length=0.2, fc='red', ec='red', alpha=0.6)

# 상위 어텐션 엣지만 그리기
threshold = np.percentile(velocity_attention[velocity_attention > 0], 80)
for i in range(len(positions)):
    for j in range(len(positions)):
        if i != j and velocity_attention[i, j] > threshold:
            ax.plot([positions[i, 0], positions[j, 0]],
                   [positions[i, 1], positions[j, 1]],
                   'b-', alpha=velocity_attention[i, j] * 2, linewidth=2, zorder=1)

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title(f'속도 기반 어텐션 그래프 (상위 20% 엣지)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_file2 = Path(__file__).parent.parent / "results/attention_graph_sample.png"
plt.savefig(output_file2, dpi=150, bbox_inches='tight')
print(f"✓ 그래프 시각화 저장: {output_file2}")

# 통계 출력
print(f"\n{'='*80}")
print("어텐션 가중치 통계")
print(f"{'='*80}")
print(f"\n거리 기반 어텐션:")
print(f"  평균: {np.mean(distance_attention):.4f}")
print(f"  최대: {np.max(distance_attention):.4f}")
print(f"  최소 (0 제외): {np.min(distance_attention[distance_attention > 0]):.4f}")
print(f"  표준편차: {np.std(distance_attention):.4f}")

print(f"\n속도 기반 어텐션:")
print(f"  평균: {np.mean(velocity_attention):.4f}")
print(f"  최대: {np.max(velocity_attention):.4f}")
print(f"  최소 (0 제외): {np.min(velocity_attention[velocity_attention > 0]):.4f}")
print(f"  표준편차: {np.std(velocity_attention):.4f}")

# 가장 높은 어텐션 쌍
max_dist_idx = np.unravel_index(np.argmax(distance_attention), distance_attention.shape)
max_vel_idx = np.unravel_index(np.argmax(velocity_attention), velocity_attention.shape)

print(f"\n최고 어텐션 쌍:")
print(f"  거리 기반: 노드 {max_dist_idx[0]} -> 노드 {max_dist_idx[1]} ({distance_attention[max_dist_idx]:.4f})")
print(f"  속도 기반: 노드 {max_vel_idx[0]} -> 노드 {max_vel_idx[1]} ({velocity_attention[max_vel_idx]:.4f})")

print(f"\n{'='*80}")
print("✓ 시각화 완료")
print(f"{'='*80}")
print(f"\n생성된 파일:")
print(f"  1. {output_file}")
print(f"  2. {output_file2}")

plt.show()

