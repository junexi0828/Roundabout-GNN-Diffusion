"""
SDD (Stanford Drone Dataset) Death Circle 데이터 어댑터
SDD 포맷을 프로젝트 표준 포맷으로 변환
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class SDDAdapter:
    """
    SDD Death Circle 데이터를 프로젝트 포맷으로 변환
    """

    def __init__(self, sdd_dir: Path):
        """
        Args:
            sdd_dir: SDD Death Circle 데이터 디렉토리
        """
        self.sdd_dir = Path(sdd_dir)
        self.homography_matrix = None

    def load_annotations(self, video_dir: Path) -> pd.DataFrame:
        """
        SDD 어노테이션 파일 로드

        Args:
            video_dir: 비디오 디렉토리 (annotations.txt 포함)

        Returns:
            변환된 데이터프레임
        """
        ann_file = video_dir / "annotations.txt"

        if not ann_file.exists():
            raise FileNotFoundError(f"어노테이션 파일 없음: {ann_file}")

        # SDD 포맷: track_id, xmin, ymin, xmax, ymax, frame, lost, occluded, generated, label
        df = pd.read_csv(
            ann_file,
            sep=' ',
            header=None,
            names=['track_id', 'xmin', 'ymin', 'xmax', 'ymax',
                   'frame', 'lost', 'occluded', 'generated', 'label'],
            quotechar='"'
        )

        # 중심점 계산
        df['x'] = (df['xmin'] + df['xmax']) / 2
        df['y'] = (df['ymin'] + df['ymax']) / 2

        # lost나 occluded가 1인 경우 제외 (선택사항)
        # df = df[(df['lost'] == 0) & (df['occluded'] == 0)]

        return df

    def apply_homography(self, x: np.ndarray, y: np.ndarray,
                        H: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        호모그래피 변환 적용 (픽셀 → 미터)

        Args:
            x: 픽셀 X 좌표
            y: 픽셀 Y 좌표
            H: 호모그래피 행렬 (None이면 추정)

        Returns:
            (x_meter, y_meter) 미터 단위 좌표
        """
        if H is None:
            # 호모그래피 행렬이 없으면 간단한 스케일링 사용
            # SDD Death Circle의 대략적인 크기: 약 30m x 40m
            # 픽셀 크기: 약 1400 x 1900
            scale_x = 30.0 / 1400.0  # 대략적인 스케일
            scale_y = 40.0 / 1900.0

            x_meter = x * scale_x
            y_meter = y * scale_y
        else:
            # 호모그래피 변환 적용
            # [x', y', w'] = H @ [x, y, 1]
            coords = np.stack([x, y, np.ones_like(x)], axis=0)
            transformed = H @ coords
            x_meter = transformed[0] / transformed[2]
            y_meter = transformed[1] / transformed[2]

        return x_meter, y_meter

    def calculate_velocity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        속도 계산 (픽셀/프레임 → m/s)

        Args:
            df: 데이터프레임

        Returns:
            속도 컬럼이 추가된 데이터프레임
        """
        df = df.copy()
        df = df.sort_values(['track_id', 'frame'])

        # 프레임 간 시간 차이 (SDD는 약 2 FPS, 0.5초 간격)
        dt = 0.5  # 초

        # 속도 계산
        df['vx'] = df.groupby('track_id')['x'].diff() / dt
        df['vy'] = df.groupby('track_id')['y'].diff() / dt

        # 첫 프레임은 0으로 설정
        df['vx'] = df['vx'].fillna(0)
        df['vy'] = df['vy'].fillna(0)

        return df

    def convert_to_project_format(self, video_dir: Path,
                                  output_path: Optional[Path] = None) -> pd.DataFrame:
        """
        SDD 포맷을 프로젝트 표준 포맷으로 변환

        Args:
            video_dir: 비디오 디렉토리
            output_path: 출력 CSV 경로 (None이면 반환만)

        Returns:
            변환된 데이터프레임
        """
        # 어노테이션 로드
        df = self.load_annotations(video_dir)

        # 속도 계산
        df = self.calculate_velocity(df)

        # 호모그래피 변환 (픽셀 → 미터)
        x_meter, y_meter = self.apply_homography(df['x'].values, df['y'].values)
        df['x'] = x_meter
        df['y'] = y_meter

        # 속도도 미터 단위로 변환
        scale_x = 30.0 / 1400.0
        scale_y = 40.0 / 1900.0
        df['vx'] = df['vx'] * scale_x
        df['vy'] = df['vy'] * scale_y

        # 프로젝트 표준 포맷으로 변환
        result_df = pd.DataFrame({
            'track_id': df['track_id'],
            'frame_id': df['frame'],
            'timestamp_ms': df['frame'] * 500,  # 0.5초 = 500ms 간격
            'agent_type': df['label'].str.lower(),
            'x': df['x'],
            'y': df['y'],
            'vx': df['vx'],
            'vy': df['vy'],
            'psi_rad': 0.0,  # 헤딩 각도 (추후 계산 가능)
            'length': (df['xmax'] - df['xmin']) * scale_x,
            'width': (df['ymax'] - df['ymin']) * scale_y
        })

        # lost나 occluded가 1인 경우 제외
        mask = (df['lost'] == 0) & (df['occluded'] == 0)
        result_df = result_df[mask].reset_index(drop=True)

        # 저장
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result_df.to_csv(output_path, index=False)
            print(f"✓ 변환 완료: {output_path}")

        return result_df

    def convert_all_videos(self, output_dir: Path):
        """
        모든 비디오를 변환

        Args:
            output_dir: 출력 디렉토리
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        video_dirs = sorted([d for d in self.sdd_dir.iterdir()
                            if d.is_dir() and d.name.startswith('video')])

        print(f"\n총 {len(video_dirs)}개 비디오 변환 시작...")

        for video_dir in video_dirs:
            video_name = video_dir.name
            print(f"\n변환 중: {video_name}")

            output_path = output_dir / f"{video_name}_converted.csv"

            try:
                df = self.convert_to_project_format(video_dir, output_path)
                print(f"  ✓ {len(df):,}행, {df['track_id'].nunique()}개 트랙")
            except Exception as e:
                print(f"  ❌ 오류: {e}")
                continue

        print(f"\n{'='*80}")
        print("✓ 모든 비디오 변환 완료")
        print(f"{'='*80}")
        print(f"\n출력 위치: {output_dir}")


def main():
    """메인 함수"""
    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent

    if len(sys.argv) > 1:
        sdd_dir = Path(sys.argv[1])
    else:
        sdd_dir = project_root / "data" / "sdd" / "deathCircle"

    output_dir = project_root / "data" / "sdd" / "converted"

    adapter = SDDAdapter(sdd_dir)
    adapter.convert_all_videos(output_dir)


if __name__ == "__main__":
    main()

