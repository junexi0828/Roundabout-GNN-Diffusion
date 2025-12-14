"""
자동 호모그래피 추정 스크립트
OpenCV SIFT/ORB를 사용하여 자동으로 특징점 매칭 및 호모그래피 계산
하루 내 완료 목표: 수동 작업 최소화
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import json

# OpenCV는 선택적 (SIFT/ORB 사용 시만 필요)
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


class AutoHomographyEstimator:
    """자동 호모그래피 추정 클래스"""

    def __init__(self, method='SIFT'):
        """
        Args:
            method: 'SIFT' 또는 'ORB' (SIFT가 더 정확하지만 느림)
        """
        self.method = method

        if method == 'SIFT':
            self.detector = cv2.SIFT_create()
            self.matcher = cv2.BFMatcher()
        else:  # ORB
            self.detector = cv2.ORB_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def extract_features(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        이미지에서 특징점 추출

        Args:
            image: 입력 이미지 (BGR)

        Returns:
            (keypoints, descriptors)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        return keypoints, descriptors

    def match_features(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
        ratio_thresh: float = 0.75
    ) -> list:
        """
        특징점 매칭

        Args:
            desc1: 첫 번째 이미지의 디스크립터
            desc2: 두 번째 이미지의 디스크립터
            ratio_thresh: Lowe's ratio test 임계값

        Returns:
            매칭된 특징점 리스트
        """
        if self.method == 'SIFT':
            # Lowe's ratio test
            matches = self.matcher.knnMatch(desc1, desc2, k=2)
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < ratio_thresh * n.distance:
                        good_matches.append(m)
        else:  # ORB
            matches = self.matcher.match(desc1, desc2)
            good_matches = sorted(matches, key=lambda x: x.distance)[:50]

        return good_matches

    def estimate_homography(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        min_matches: int = 10
    ) -> Tuple[Optional[np.ndarray], int, np.ndarray]:
        """
        두 이미지 간 호모그래피 추정

        Args:
            img1: 첫 번째 이미지 (SDD 비디오 프레임)
            img2: 두 번째 이미지 (위성 지도 또는 참조 이미지)
            min_matches: 최소 매칭 점 수

        Returns:
            (호모그래피 행렬, 매칭 점 수, 매칭 시각화 이미지)
        """
        # 특징점 추출
        kp1, desc1 = self.extract_features(img1)
        kp2, desc2 = self.extract_features(img2)

        if desc1 is None or desc2 is None:
            return None, 0, img1

        # 특징점 매칭
        matches = self.match_features(desc1, desc2)

        if len(matches) < min_matches:
            print(f"⚠️  매칭 점 부족: {len(matches)}개 (최소 {min_matches}개 필요)")
            return None, len(matches), img1

        # 매칭 점 좌표 추출
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # RANSAC으로 호모그래피 추정
        H, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            cv2.RANSAC,
            ransacReprojThreshold=5.0
        )

        # 매칭 시각화
        matches_img = cv2.drawMatches(
            img1, kp1, img2, kp2,
            [matches[i] for i in range(len(matches)) if mask[i]],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        inliers = int(mask.sum())
        print(f"✓ 호모그래피 추정 완료: {inliers}/{len(matches)} inliers")

        return H, inliers, matches_img

    def estimate_from_known_points(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        알려진 대응점으로부터 호모그래피 추정 (가장 빠른 방법)

        Args:
            src_points: 픽셀 좌표 (N, 2)
            dst_points: 미터 좌표 (N, 2)

        Returns:
            호모그래피 행렬
        """
        if len(src_points) < 4:
            raise ValueError("최소 4개의 대응점 필요")

        H, mask = cv2.findHomography(
            src_points.reshape(-1, 1, 2),
            dst_points.reshape(-1, 1, 2),
            cv2.RANSAC,
            ransacReprojThreshold=1.0
        )

        return H


def load_sdd_frame(video_path: Path, frame_idx: int = 0) -> Optional[np.ndarray]:
    """SDD 비디오에서 프레임 추출"""
    if not video_path.exists():
        return None

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    return frame if ret else None


def estimate_scale_from_reference(
    pixel_coords: np.ndarray,
    known_distance_pixels: float,
    known_distance_meters: float
) -> Tuple[float, float]:
    """
    알려진 거리로부터 스케일 팩터 추정 (가장 간단한 방법)

    Args:
        pixel_coords: 픽셀 좌표
        known_distance_pixels: 픽셀 단위 알려진 거리 (예: 차선 폭)
        known_distance_meters: 미터 단위 알려진 거리 (예: 3.0m)

    Returns:
        (scale_x, scale_y)
    """
    scale = known_distance_meters / known_distance_pixels
    return scale, scale


def quick_homography_for_sdd():
    """
    SDD Death Circle용 빠른 호모그래피 추정
    회전교차로 중심점과 알려진 거리로부터 추정
    OpenCV 불필요 - 순수 NumPy만 사용
    """
    # SDD Death Circle 대략적 크기
    # 회전교차로 직경: 약 20-30m
    # 이미지 크기: 약 1400x1900 픽셀

    # 방법 1: 간단한 스케일링 (이미 구현됨)
    scale_x = 30.0 / 1400.0
    scale_y = 40.0 / 1900.0

    # 방법 2: 회전교차로 중심점 기준 변환
    # 중심점: (700, 950) 픽셀 → (0, 0) 미터
    center_pixel = np.array([700, 950])
    center_meter = np.array([0, 0])

    # 아핀 변환 행렬 (회전 없음, 스케일만)
    H_affine = np.array([
        [scale_x, 0, -center_pixel[0] * scale_x + center_meter[0]],
        [0, scale_y, -center_pixel[1] * scale_y + center_meter[1]],
        [0, 0, 1]
    ])

    return H_affine


def save_homography(H: np.ndarray, output_path: Path):
    """호모그래피 행렬 저장"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, H, fmt='%.8f')
    print(f"✓ 호모그래피 저장: {output_path}")


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='자동 호모그래피 추정')
    parser.add_argument('--method', choices=['SIFT', 'ORB', 'quick'], default='quick',
                       help='추정 방법 (quick: 가장 빠름, SIFT: 가장 정확)')
    parser.add_argument('--video', type=str, help='SDD 비디오 경로')
    parser.add_argument('--satellite', type=str, help='위성 지도 이미지 경로')
    parser.add_argument('--output', type=str, default='data/sdd/homography/H.txt',
                       help='출력 경로')

    args = parser.parse_args()

    if args.method == 'quick':
        # 가장 빠른 방법: 알려진 스케일 사용
        print("🚀 빠른 호모그래피 추정 (스케일 기반)...")
        H = quick_homography_for_sdd()
        save_homography(H, Path(args.output))
        print("✓ 완료! (약 1분 소요)")

    elif args.video and args.satellite:
        # 자동 특징점 매칭
        if not HAS_OPENCV:
            print("❌ OpenCV가 필요합니다. 설치: pip install opencv-python")
            return

        print(f"🔍 {args.method} 특징점 매칭 중...")

        estimator = AutoHomographyEstimator(method=args.method)

        img1 = cv2.imread(args.video) if Path(args.video).suffix in ['.jpg', '.png'] else load_sdd_frame(Path(args.video))
        img2 = cv2.imread(args.satellite)

        if img1 is None or img2 is None:
            print("❌ 이미지 로드 실패")
            return

        H, num_matches, matches_img = estimator.estimate_homography(img1, img2)

        if H is not None:
            save_homography(H, Path(args.output))
            cv2.imwrite(str(Path(args.output).parent / 'matches.jpg'), matches_img)
            print(f"✓ 완료! ({num_matches}개 매칭)")
        else:
            print("❌ 호모그래피 추정 실패")

    else:
        print("사용법:")
        print("  빠른 방법: python scripts/auto_homography_estimation.py --method quick")
        print("  자동 매칭: python scripts/auto_homography_estimation.py --method SIFT --video <비디오> --satellite <위성지도>")


if __name__ == "__main__":
    main()

