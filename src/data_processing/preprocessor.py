"""
데이터 전처리 파이프라인
- 좌표계 변환 (회전교차로 중심 기준 상대 좌표)
- 시계열 윈도우 생성 (과거 3초/미래 5초)
- 이상치 제거 및 보간
- Feature 정규화 (Z-Score)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class TrajectoryPreprocessor:
    """궤적 데이터 전처리 클래스"""

    def __init__(
        self,
        obs_window: int = 30,  # 과거 3초 (10Hz 기준)
        pred_window: int = 50,  # 미래 5초 (10Hz 기준)
        overlap: float = 0.5,  # 윈도우 간 50% 중첩
        sampling_rate: float = 10.0  # Hz
    ):
        """
        Args:
            obs_window: 관측 윈도우 길이 (프레임 수)
            pred_window: 예측 윈도우 길이 (프레임 수)
            overlap: 윈도우 간 중첩 비율 (0~1)
            sampling_rate: 데이터 샘플링 속도 (Hz)
        """
        self.obs_window = obs_window
        self.pred_window = pred_window
        self.overlap = overlap
        self.sampling_rate = sampling_rate

        # 정규화를 위한 스케일러 (학습 데이터로 fit 필요)
        self.scaler = StandardScaler()
        self.scaler_fitted = False

        # 회전교차로 중심점 (데이터에서 계산)
        self.roundabout_center = None

    def calculate_roundabout_center(self, df: pd.DataFrame) -> Tuple[float, float]:
        """
        회전교차로의 중심점을 계산합니다.
        모든 차량의 평균 위치를 사용합니다.

        Args:
            df: 전체 궤적 데이터프레임

        Returns:
            (cx, cy): 중심점 좌표
        """
        # 모든 차량의 평균 위치 계산
        cx = df['x'].mean()
        cy = df['y'].mean()

        self.roundabout_center = (cx, cy)
        return (cx, cy)

    def transform_coordinates(
        self,
        df: pd.DataFrame,
        center: Optional[Tuple[float, float]] = None
    ) -> pd.DataFrame:
        """
        좌표계를 회전교차로 중심 기준 상대 좌표로 변환합니다.

        Args:
            df: 원본 데이터프레임
            center: 중심점 좌표 (None이면 자동 계산)

        Returns:
            변환된 데이터프레임
        """
        df = df.copy()

        if center is None:
            if self.roundabout_center is None:
                center = self.calculate_roundabout_center(df)
            else:
                center = self.roundabout_center
        else:
            self.roundabout_center = center

        cx, cy = center

        # 좌표 변환
        df['x'] = df['x'] - cx
        df['y'] = df['y'] - cy

        return df

    def remove_outliers(
        self,
        df: pd.DataFrame,
        speed_threshold: float = 50.0,  # m/s (약 180 km/h)
        acceleration_threshold: float = 10.0  # m/s²
    ) -> pd.DataFrame:
        """
        이상치를 제거합니다.

        Args:
            df: 데이터프레임
            speed_threshold: 최대 속도 임계값
            acceleration_threshold: 최대 가속도 임계값

        Returns:
            이상치가 제거된 데이터프레임
        """
        df = df.copy()

        # 속도 계산
        if 'vx' in df.columns and 'vy' in df.columns:
            df['speed'] = np.sqrt(df['vx']**2 + df['vy']**2)

            # 속도 이상치 제거
            speed_mask = df['speed'] <= speed_threshold
            df = df[speed_mask]

        # 가속도 계산 및 이상치 제거
        if 'track_id' in df.columns:
            for track_id in df['track_id'].unique():
                track_data = df[df['track_id'] == track_id].sort_values('frame_id')

                if len(track_data) > 1:
                    # 가속도 계산 (속도 변화율)
                    dt = 1.0 / self.sampling_rate
                    dvx = track_data['vx'].diff() / dt
                    dvy = track_data['vy'].diff() / dt
                    accel = np.sqrt(dvx**2 + dvy**2)

                    # 가속도 이상치가 있는 프레임 제거
                    accel_mask = accel <= acceleration_threshold
                    accel_mask.iloc[0] = True  # 첫 프레임은 유지

                    valid_frames = track_data[accel_mask]['frame_id'].values
                    df = df[~((df['track_id'] == track_id) &
                             (~df['frame_id'].isin(valid_frames)))]

        return df

    def interpolate_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        결측치를 선형 보간법으로 채웁니다.

        Args:
            df: 데이터프레임

        Returns:
            보간된 데이터프레임
        """
        df = df.copy()

        # track_id별로 그룹화하여 보간
        numeric_columns = df.select_dtypes(include=[np.number]).columns

        for track_id in df['track_id'].unique():
            track_data = df[df['track_id'] == track_id].sort_values('frame_id')

            for col in numeric_columns:
                if col in ['track_id', 'frame_id']:
                    continue

                # 결측치가 있는 경우 보간
                if track_data[col].isnull().any():
                    track_data[col] = track_data[col].interpolate(method='linear')
                    # 양 끝 결측치는 forward/backward fill
                    track_data[col] = track_data[col].fillna(method='ffill')
                    track_data[col] = track_data[col].fillna(method='bfill')

                    # 업데이트
                    df.loc[df['track_id'] == track_id, col] = track_data[col].values

        return df

    def create_sliding_windows(
        self,
        df: pd.DataFrame
    ) -> List[Dict]:
        """
        시계열 윈도우를 생성합니다.

        Args:
            df: 전처리된 데이터프레임

        Returns:
            윈도우 리스트. 각 윈도우는 {
                'track_id': ...,
                'obs_frames': [...],  # 관측 프레임
                'pred_frames': [...],  # 예측 프레임
                'obs_data': DataFrame,  # 관측 데이터
                'pred_data': DataFrame  # 예측 데이터
            } 형태
        """
        windows = []
        step_size = int(self.obs_window * (1 - self.overlap))

        for track_id in df['track_id'].unique():
            track_data = df[df['track_id'] == track_id].sort_values('frame_id')
            track_frames = track_data['frame_id'].values

            if len(track_frames) < self.obs_window + self.pred_window:
                continue

            # 슬라이딩 윈도우 생성
            for start_idx in range(0, len(track_frames) - self.obs_window - self.pred_window + 1, step_size):
                obs_end_idx = start_idx + self.obs_window
                pred_end_idx = obs_end_idx + self.pred_window

                obs_frames = track_frames[start_idx:obs_end_idx]
                pred_frames = track_frames[obs_end_idx:pred_end_idx]

                obs_data = track_data[track_data['frame_id'].isin(obs_frames)]
                pred_data = track_data[track_data['frame_id'].isin(pred_frames)]

                if len(obs_data) == self.obs_window and len(pred_data) == self.pred_window:
                    windows.append({
                        'track_id': track_id,
                        'obs_frames': obs_frames.tolist(),
                        'pred_frames': pred_frames.tolist(),
                        'obs_data': obs_data,
                        'pred_data': pred_data
                    })

        return windows

    def normalize_features(
        self,
        df: pd.DataFrame,
        fit: bool = False,
        feature_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Feature를 Z-Score 정규화합니다.

        Args:
            df: 데이터프레임
            fit: True이면 스케일러를 fit하고, False이면 기존 스케일러 사용
            feature_columns: 정규화할 컬럼 목록 (None이면 자동 선택)

        Returns:
            정규화된 데이터프레임
        """
        df = df.copy()

        if feature_columns is None:
            # 기본 feature 컬럼
            feature_columns = ['x', 'y', 'vx', 'vy']
            if 'psi_rad' in df.columns:
                feature_columns.append('psi_rad')

        # 존재하는 컬럼만 선택
        feature_columns = [col for col in feature_columns if col in df.columns]

        if not feature_columns:
            return df

        if fit or not self.scaler_fitted:
            # 스케일러 학습
            self.scaler.fit(df[feature_columns])
            self.scaler_fitted = True

        # 정규화 적용
        df[feature_columns] = self.scaler.transform(df[feature_columns])

        return df

    def preprocess_scenario(
        self,
        csv_path: Path,
        output_dir: Optional[Path] = None
    ) -> Dict:
        """
        시나리오 파일 전체를 전처리합니다.

        Args:
            csv_path: 원본 CSV 파일 경로
            output_dir: 전처리된 데이터 저장 디렉토리 (None이면 저장 안 함)

        Returns:
            전처리 결과 딕셔너리
        """
        print(f"전처리 중: {csv_path.name}")

        # 데이터 로드
        df = pd.read_csv(csv_path)

        # 1. 좌표계 변환
        df = self.transform_coordinates(df)

        # 2. 이상치 제거
        df = self.remove_outliers(df)

        # 3. 결측치 보간
        df = self.interpolate_missing_values(df)

        # 4. Feature 정규화 (fit=True로 처음에 학습)
        df = self.normalize_features(df, fit=True)

        # 5. 윈도우 생성
        windows = self.create_sliding_windows(df)

        result = {
            'original_rows': len(pd.read_csv(csv_path)),
            'processed_rows': len(df),
            'num_windows': len(windows),
            'windows': windows
        }

        # 저장
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{csv_path.stem}_processed.pkl"
            import pickle
            with open(output_path, 'wb') as f:
                pickle.dump(result, f)
            print(f"  저장 완료: {output_path}")

        return result


def main():
    """테스트용 메인 함수"""
    from pathlib import Path
    import sys

    project_root = Path(__file__).parent.parent.parent

    if len(sys.argv) < 2:
        print("사용법: python preprocessor.py <csv_file_path>")
        return

    csv_path = Path(sys.argv[1])
    output_dir = project_root / "data" / "processed"

    preprocessor = TrajectoryPreprocessor()
    result = preprocessor.preprocess_scenario(csv_path, output_dir)

    print(f"\n전처리 완료:")
    print(f"  원본 행 수: {result['original_rows']:,}")
    print(f"  전처리 후 행 수: {result['processed_rows']:,}")
    print(f"  생성된 윈도우 수: {result['num_windows']:,}")


if __name__ == "__main__":
    main()

