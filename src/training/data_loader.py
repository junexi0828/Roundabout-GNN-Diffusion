"""
학습용 데이터 로더 구현
Index Batching을 활용한 메모리 효율적인 데이터 로딩
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import pickle


class TrajectoryDataset(Dataset):
    """
    궤적 예측을 위한 데이터셋 클래스
    전처리된 윈도우 데이터를 로드
    """

    def __init__(
        self,
        windows: List[Dict],
        scene_graph_builder=None,
        use_scene_graph: bool = True
    ):
        """
        Args:
            windows: 전처리된 윈도우 리스트 (preprocessor에서 생성)
            scene_graph_builder: 씬 그래프 빌더 인스턴스
            use_scene_graph: 씬 그래프 사용 여부
        """
        self.windows = windows
        self.scene_graph_builder = scene_graph_builder
        self.use_scene_graph = use_scene_graph

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict:
        """
        데이터 샘플 반환

        Returns:
            {
                'obs_data': 관측 데이터 (DataFrame 또는 Tensor),
                'pred_data': 예측 타겟 (DataFrame 또는 Tensor),
                'graph': 씬 그래프 (Data 객체, 선택사항)
            }
        """
        window = self.windows[idx]

        obs_data = window['obs_data']
        pred_data = window['pred_data']

        result = {
            'obs_data': obs_data,
            'pred_data': pred_data,
            'track_id': window['track_id']
        }

        # 씬 그래프 생성
        if self.use_scene_graph and self.scene_graph_builder is not None:
            # 마지막 관측 프레임으로 그래프 생성
            last_frame = obs_data.iloc[-1] if isinstance(obs_data, pd.DataFrame) else obs_data[-1]
            frame_data = obs_data if isinstance(obs_data, pd.DataFrame) else pd.DataFrame([obs_data])

            graph = self.scene_graph_builder.build_from_frame(frame_data)
            pyg_data = self.scene_graph_builder.to_pytorch_geometric()
            result['graph'] = pyg_data

        return result


class IndexBatchingDataLoader:
    """
    Index Batching을 활용한 메모리 효율적인 데이터 로더
    전체 데이터를 메모리에 올리지 않고 인덱스만 관리
    """

    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0
    ):
        """
        Args:
            data_dir: 전처리된 데이터 디렉토리
            batch_size: 배치 크기
            shuffle: 셔플 여부
            num_workers: 데이터 로딩 워커 수
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        # 데이터 파일 목록 로드
        self.data_files = list(self.data_dir.glob("*_processed.pkl"))
        self.indices = list(range(len(self.data_files)))

    def __len__(self) -> int:
        return len(self.data_files) // self.batch_size + (
            1 if len(self.data_files) % self.batch_size > 0 else 0
        )

    def __iter__(self):
        """이터레이터"""
        if self.shuffle:
            np.random.shuffle(self.indices)

        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            batch = self._load_batch(batch_indices)
            yield batch

    def _load_batch(self, indices: List[int]) -> List[Dict]:
        """배치 데이터 로드"""
        batch = []
        for idx in indices:
            data_file = self.data_files[idx]
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
                # 각 윈도우를 배치에 추가
                batch.extend(data.get('windows', []))
        return batch


def collate_fn(batch: List[Dict]) -> Dict:
    """
    배치 데이터를 모델 입력 형식으로 변환

    Args:
        batch: 데이터 샘플 리스트

    Returns:
        배치화된 데이터 딕셔너리
    """
    # 관측 데이터 수집
    obs_data_list = []
    pred_data_list = []
    graphs = []
    track_ids = []

    for sample in batch:
        obs_data = sample['obs_data']
        pred_data = sample['pred_data']

        # DataFrame을 Tensor로 변환
        if isinstance(obs_data, pd.DataFrame):
            obs_tensor = torch.tensor(
                obs_data[['x', 'y', 'vx', 'vy', 'psi_rad']].values,
                dtype=torch.float
            )
        else:
            obs_tensor = torch.tensor(obs_data, dtype=torch.float)

        if isinstance(pred_data, pd.DataFrame):
            pred_tensor = torch.tensor(
                pred_data[['x', 'y']].values,
                dtype=torch.float
            )
        else:
            pred_tensor = torch.tensor(pred_data, dtype=torch.float)

        obs_data_list.append(obs_tensor)
        pred_data_list.append(pred_tensor)
        track_ids.append(sample['track_id'])

        # 그래프가 있으면 추가
        if 'graph' in sample:
            graphs.append(sample['graph'])

    # 배치화
    result = {
        'obs_data': torch.stack(obs_data_list),  # [batch_size, obs_len, features]
        'pred_data': torch.stack(pred_data_list),  # [batch_size, pred_len, 2]
        'future_data': torch.stack(pred_data_list),  # MID 호환성 (future_trajectory)
        'obs_trajectory': torch.stack(obs_data_list)[:, :, :2],  # MID용 (x, y만)
        'future_trajectory': torch.stack(pred_data_list),  # MID용
        'track_ids': track_ids
    }

    # 그래프 배치화
    if graphs:
        result['graph'] = Batch.from_data_list(graphs)
        result['graph_data'] = result['graph']  # MID 호환성

    return result


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    collate_fn=collate_fn
) -> DataLoader:
    """
    데이터 로더 생성

    Args:
        dataset: 데이터셋 객체
        batch_size: 배치 크기
        shuffle: 셔플 여부
        num_workers: 워커 수
        collate_fn: 배치화 함수

    Returns:
        DataLoader 객체
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )


def split_dataset(
    windows: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    데이터셋을 train/val/test로 분할

    Args:
        windows: 전체 윈도우 리스트
        train_ratio: 학습 데이터 비율
        val_ratio: 검증 데이터 비율
        test_ratio: 테스트 데이터 비율

    Returns:
        (train_windows, val_windows, test_windows)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "비율의 합이 1이어야 합니다."

    # 셔플
    indices = np.random.permutation(len(windows))

    # 분할 인덱스 계산
    n_train = int(len(windows) * train_ratio)
    n_val = int(len(windows) * val_ratio)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    train_windows = [windows[i] for i in train_indices]
    val_windows = [windows[i] for i in val_indices]
    test_windows = [windows[i] for i in test_indices]

    return train_windows, val_windows, test_windows


def main():
    """테스트용 메인 함수"""
    # 더미 데이터 생성
    windows = []
    for i in range(100):
        windows.append({
            'track_id': i,
            'obs_frames': list(range(30)),
            'pred_frames': list(range(30, 80)),
            'obs_data': pd.DataFrame({
                'x': np.random.randn(30),
                'y': np.random.randn(30),
                'vx': np.random.randn(30),
                'vy': np.random.randn(30),
                'psi_rad': np.random.randn(30)
            }),
            'pred_data': pd.DataFrame({
                'x': np.random.randn(50),
                'y': np.random.randn(50)
            })
        })

    # 데이터셋 분할
    train_windows, val_windows, test_windows = split_dataset(windows)

    print(f"데이터셋 분할:")
    print(f"  학습: {len(train_windows)}개")
    print(f"  검증: {len(val_windows)}개")
    print(f"  테스트: {len(test_windows)}개")

    # 데이터셋 생성
    train_dataset = TrajectoryDataset(train_windows)
    train_loader = create_dataloader(train_dataset, batch_size=8)

    # 배치 확인
    for batch in train_loader:
        print(f"\n배치 구조:")
        print(f"  관측 데이터: {batch['obs_data'].shape}")
        print(f"  예측 데이터: {batch['pred_data'].shape}")
        print(f"  트랙 ID 수: {len(batch['track_ids'])}")
        break


if __name__ == "__main__":
    main()

