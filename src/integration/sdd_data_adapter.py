"""
SDD Death Circle 데이터셋 어댑터
호모그래피 변환 및 이기종 에이전트 처리
"""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle


class HomographyEstimator:
    """호모그래피 행렬 추정 클래스"""

    def __init__(self):
        self.homography_matrix = None

    def estimate_from_correspondences(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray,
        method: str = 'RANSAC'
    ) -> np.ndarray:
        """
        대응점으로부터 호모그래피 행렬 추정

        Args:
            src_points: 소스 점들 (픽셀 좌표) [N, 2]
            dst_points: 대상 점들 (미터 좌표) [N, 2]
            method: 추정 방법 ('RANSAC', 'LEAST_SQUARES')

        Returns:
            호모그래피 행렬 [3, 3]
        """
        if method == 'RANSAC':
            self.homography_matrix, mask = cv2.findHomography(
                src_points.astype(np.float32),
                dst_points.astype(np.float32),
                cv2.RANSAC,
                5.0
            )
        else:
            self.homography_matrix, mask = cv2.findHomography(
                src_points.astype(np.float32),
                dst_points.astype(np.float32),
                0
            )

        return self.homography_matrix

    def transform_points(
        self,
        points: np.ndarray
    ) -> np.ndarray:
        """
        픽셀 좌표를 미터 좌표로 변환

        Args:
            points: 픽셀 좌표 [N, 2]

        Returns:
            미터 좌표 [N, 2]
        """
        if self.homography_matrix is None:
            raise ValueError("호모그래피 행렬이 추정되지 않았습니다.")

        # 동차 좌표로 변환
        points_homogeneous = np.column_stack([points, np.ones(len(points))])

        # 변환
        transformed = (self.homography_matrix @ points_homogeneous.T).T

        # 정규화 (동차 좌표에서 일반 좌표로)
        transformed = transformed[:, :2] / transformed[:, 2:3]

        return transformed

    def load_from_file(self, file_path: Path):
        """파일에서 호모그래피 행렬 로드"""
        if file_path.suffix == '.txt':
            self.homography_matrix = np.loadtxt(file_path)
        elif file_path.suffix == '.pkl':
            with open(file_path, 'rb') as f:
                self.homography_matrix = pickle.load(f)
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {file_path.suffix}")

    def save_to_file(self, file_path: Path):
        """호모그래피 행렬을 파일로 저장"""
        if self.homography_matrix is None:
            raise ValueError("저장할 호모그래피 행렬이 없습니다.")

        if file_path.suffix == '.txt':
            np.savetxt(file_path, self.homography_matrix)
        elif file_path.suffix == '.pkl':
            with open(file_path, 'wb') as f:
                pickle.dump(self.homography_matrix, f)
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {file_path.suffix}")


class SDDDataAdapter:
    """SDD Death Circle 데이터셋 어댑터"""

    def __init__(
        self,
        homography_path: Optional[Path] = None,
        homography_matrix: Optional[np.ndarray] = None
    ):
        """
        Args:
            homography_path: 호모그래피 행렬 파일 경로
            homography_matrix: 호모그래피 행렬 (직접 제공)
        """
        self.homography = HomographyEstimator()

        if homography_path:
            self.homography.load_from_file(homography_path)
        elif homography_matrix is not None:
            self.homography.homography_matrix = homography_matrix

    def convert_pixel_to_meter(
        self,
        trajectory_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        픽셀 좌표를 미터 좌표로 변환

        Args:
            trajectory_data: SDD 원본 데이터 (픽셀 좌표)

        Returns:
            변환된 데이터 (미터 좌표)
        """
        df = trajectory_data.copy()

        # 픽셀 좌표 추출
        pixel_coords = df[['x', 'y']].values

        # 미터 좌표로 변환
        meter_coords = self.homography.transform_points(pixel_coords)

        # 좌표 업데이트
        df['x'] = meter_coords[:, 0]
        df['y'] = meter_coords[:, 1]

        return df

    def extract_agent_features(
        self,
        trajectory_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        에이전트 특징 추출 (속도, 가속도 등)

        Args:
            trajectory_data: 궤적 데이터

        Returns:
            특징이 추가된 데이터
        """
        df = trajectory_data.copy()

        # track_id별로 그룹화하여 속도/가속도 계산
        for track_id in df['track_id'].unique():
            track_data = df[df['track_id'] == track_id].sort_values('frame_id')

            if len(track_data) < 2:
                continue

            # 시간 간격 (SDD는 보통 30fps, 다운샘플링 시 조정)
            dt = 1.0 / 10.0  # 10Hz 가정

            # 속도 계산
            dx = track_data['x'].diff()
            dy = track_data['y'].diff()

            df.loc[df['track_id'] == track_id, 'vx'] = dx / dt
            df.loc[df['track_id'] == track_id, 'vy'] = dy / dt

            # 가속도 계산
            dvx = df.loc[df['track_id'] == track_id, 'vx'].diff()
            dvy = df.loc[df['track_id'] == track_id, 'vy'].diff()

            df.loc[df['track_id'] == track_id, 'ax'] = dvx / dt
            df.loc[df['track_id'] == track_id, 'ay'] = dvy / dt

        # 첫 프레임의 속도/가속도는 0으로 설정
        df['vx'] = df['vx'].fillna(0.0)
        df['vy'] = df['vy'].fillna(0.0)
        df['ax'] = df['ax'].fillna(0.0)
        df['ay'] = df['ay'].fillna(0.0)

        return df

    def create_heterogeneous_graph_data(
        self,
        frame_data: pd.DataFrame
    ) -> Dict:
        """
        이기종 그래프 데이터 생성

        Args:
            frame_data: 프레임 데이터

        Returns:
            이기종 그래프 데이터 딕셔너리
        """
        # 에이전트 타입별로 그룹화
        agent_types = frame_data['agent_type'].unique() if 'agent_type' in frame_data.columns else ['unknown']

        nodes_by_type = {}
        for agent_type in agent_types:
            type_data = frame_data[frame_data['agent_type'] == agent_type]
            nodes_by_type[agent_type] = type_data

        # 관계 타입 정의
        # SDD Death Circle의 이기종 관계
        relations = {
            ('car', 'yield', 'pedestrian'): [],
            ('biker', 'overtake', 'car'): [],
            ('pedestrian', 'avoid', 'pedestrian'): [],
            ('biker', 'filter', 'car'): []
        }

        # 관계 인스턴스 생성 (간단한 구현)
        # 실제로는 더 정교한 로직 필요
        for (src_type, relation, dst_type), edge_list in relations.items():
            src_nodes = nodes_by_type.get(src_type, pd.DataFrame())
            dst_nodes = nodes_by_type.get(dst_type, pd.DataFrame())

            # 거리 기반으로 엣지 생성
            for _, src_row in src_nodes.iterrows():
                for _, dst_row in dst_nodes.iterrows():
                    dist = np.sqrt(
                        (src_row['x'] - dst_row['x'])**2 +
                        (src_row['y'] - dst_row['y'])**2
                    )

                    if dist < 15.0:  # 15m 이내
                        edge_list.append({
                            'source': src_row['track_id'],
                            'target': dst_row['track_id'],
                            'distance': dist,
                            'relation': relation
                        })

        return {
            'nodes_by_type': nodes_by_type,
            'relations': relations
        }


def main():
    """테스트용 메인 함수"""
    # 호모그래피 추정 예시
    estimator = HomographyEstimator()

    # 더미 대응점
    src_points = np.array([[100, 100], [200, 100], [200, 200], [100, 200]])
    dst_points = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])

    H = estimator.estimate_from_correspondences(src_points, dst_points)
    print(f"호모그래피 행렬:\n{H}")

    # 변환 테스트
    test_point = np.array([[150, 150]])
    transformed = estimator.transform_points(test_point)
    print(f"변환된 좌표: {transformed}")


if __name__ == "__main__":
    main()

