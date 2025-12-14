"""
Plan B: 안전 지표 산출 시스템
2D Time-to-Collision (TTC), Post-Encroachment Time (PET),
Deceleration Rate to Avoid Collision (DRAC) 계산
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
from scipy.spatial import cKDTree
import pandas as pd


@dataclass
class VehicleState:
    """차량 상태 데이터 클래스"""
    x: float
    y: float
    vx: float  # x 방향 속도 (m/s)
    vy: float  # y 방향 속도 (m/s)
    ax: float = 0.0  # x 방향 가속도 (m/s²)
    ay: float = 0.0  # y 방향 가속도 (m/s²)
    width: float = 2.0  # 차량 폭 (m)
    length: float = 4.5  # 차량 길이 (m)

    @property
    def position(self) -> np.ndarray:
        """위치 벡터"""
        return np.array([self.x, self.y])

    @property
    def velocity(self) -> np.ndarray:
        """속도 벡터"""
        return np.array([self.vx, self.vy])

    @property
    def acceleration(self) -> np.ndarray:
        """가속도 벡터"""
        return np.array([self.ax, self.ay])

    @property
    def speed(self) -> float:
        """속력"""
        return np.linalg.norm(self.velocity)

    @property
    def radius(self) -> float:
        """차량을 원으로 근사할 때의 반경"""
        return np.sqrt(self.width**2 + self.length**2) / 2


class SafetyMetricsCalculator:
    """안전 지표 계산 클래스"""

    def __init__(self, vehicle_radius: float = 2.5):
        """
        Args:
            vehicle_radius: 차량을 원으로 근사할 때의 반경 (m)
        """
        self.vehicle_radius = vehicle_radius

    def calculate_2d_ttc(
        self,
        vehicle1: VehicleState,
        vehicle2: VehicleState
    ) -> Optional[float]:
        """
        2D Time-to-Collision (TTC) 계산

        두 차량이 충돌할 때까지의 예상 시간을 계산합니다.
        차량을 원으로 근사하여 2차원 평면에서의 충돌을 계산합니다.

        Args:
            vehicle1: 첫 번째 차량 상태
            vehicle2: 두 번째 차량 상태

        Returns:
            TTC (초), 충돌하지 않으면 None
        """
        # 상대 위치 벡터
        delta_p = vehicle1.position - vehicle2.position
        # 상대 속도 벡터
        delta_v = vehicle1.velocity - vehicle2.velocity

        # 충돌 조건: ||delta_p + delta_v * t|| = 2R
        # 이를 전개하면: ||delta_v||² * t² + 2(delta_p · delta_v) * t + ||delta_p||² - (2R)² = 0
        R = self.vehicle_radius

        A = np.dot(delta_v, delta_v)
        B = 2 * np.dot(delta_p, delta_v)
        C = np.dot(delta_p, delta_p) - (2 * R) ** 2

        # 판별식
        discriminant = B**2 - 4 * A * C

        if discriminant < 0 or A == 0:
            # 충돌하지 않음 (발산하거나 평행)
            return None

        # 근의 공식
        sqrt_discriminant = np.sqrt(discriminant)
        t1 = (-B - sqrt_discriminant) / (2 * A)
        t2 = (-B + sqrt_discriminant) / (2 * A)

        # 양의 실근 중 최솟값
        positive_roots = [t for t in [t1, t2] if t > 0]

        if not positive_roots:
            return None

        ttc = min(positive_roots)

        # 미래에 충돌하는지 확인
        future_pos1 = vehicle1.position + vehicle1.velocity * ttc
        future_pos2 = vehicle2.position + vehicle2.velocity * ttc
        future_dist = np.linalg.norm(future_pos1 - future_pos2)

        if future_dist <= 2 * R + 1e-6:  # 작은 오차 허용
            return ttc

        return None

    def calculate_pet(
        self,
        trajectory1: np.ndarray,
        trajectory2: np.ndarray,
        conflict_area: Polygon,
        timestamps: Optional[np.ndarray] = None
    ) -> Optional[float]:
        """
        Post-Encroachment Time (PET) 계산

        선행 차량이 충돌 영역을 벗어난 시점과 후행 차량이 진입한 시점의 시간차

        Args:
            trajectory1: 첫 번째 차량의 궤적 [[x1, y1], [x2, y2], ...]
            trajectory2: 두 번째 차량의 궤적
            conflict_area: 충돌 영역 (Shapely Polygon)
            timestamps: 각 궤적 포인트의 타임스탬프 (초)

        Returns:
            PET (초), 계산 불가능하면 None
        """
        if timestamps is None:
            # 타임스탬프가 없으면 인덱스를 시간으로 사용
            timestamps1 = np.arange(len(trajectory1))
            timestamps2 = np.arange(len(trajectory2))
        else:
            timestamps1 = timestamps
            timestamps2 = timestamps

        # 궤적을 LineString으로 변환
        traj1_line = LineString(trajectory1)
        traj2_line = LineString(trajectory2)

        # 충돌 영역과의 교차점 찾기
        intersection1 = traj1_line.intersection(conflict_area)
        intersection2 = traj2_line.intersection(conflict_area)

        if intersection1.is_empty or intersection2.is_empty:
            return None

        # 진입/출구 시점 계산
        # 간단한 구현: 궤적 포인트가 충돌 영역 내부에 있는지 확인
        entry1 = None
        exit1 = None
        entry2 = None
        exit2 = None

        for i, point in enumerate(trajectory1):
            p = Point(point)
            if conflict_area.contains(p) or conflict_area.boundary.distance(p) < 0.1:
                if entry1 is None:
                    entry1 = timestamps1[i]
                exit1 = timestamps1[i]

        for i, point in enumerate(trajectory2):
            p = Point(point)
            if conflict_area.contains(p) or conflict_area.boundary.distance(p) < 0.1:
                if entry2 is None:
                    entry2 = timestamps2[i]
                exit2 = timestamps2[i]

        if entry1 is None or entry2 is None:
            return None

        # PET 계산: |entry2 - exit1| 또는 |entry1 - exit2|
        # 더 작은 값 사용 (더 위험한 상황)
        pet1 = abs(entry2 - exit1) if exit1 is not None else None
        pet2 = abs(entry1 - exit2) if exit2 is not None else None

        pets = [p for p in [pet1, pet2] if p is not None]
        if not pets:
            return None

        return min(pets)

    def calculate_drac(
        self,
        vehicle1: VehicleState,
        vehicle2: VehicleState,
        ttc: Optional[float] = None
    ) -> Optional[float]:
        """
        Deceleration Rate to Avoid Collision (DRAC) 계산

        충돌을 피하기 위해 필요한 최소 감속도

        Args:
            vehicle1: 첫 번째 차량 상태
            vehicle2: 두 번째 차량 상태
            ttc: TTC 값 (None이면 계산)

        Returns:
            DRAC (m/s²), 계산 불가능하면 None
        """
        if ttc is None:
            ttc = self.calculate_2d_ttc(vehicle1, vehicle2)

        if ttc is None or ttc <= 0:
            return None

        # 현재 거리
        distance = np.linalg.norm(vehicle1.position - vehicle2.position)

        # 안전 거리 (차량 반경의 2배)
        safe_distance = 2 * self.vehicle_radius

        # 필요한 감속도: v² = u² + 2as
        # 여기서 v=0 (정지), u=현재속도, s=거리
        # a = -u² / (2s)
        current_speed = vehicle1.speed

        if distance <= safe_distance:
            # 이미 너무 가까움
            return float('inf')

        required_deceleration = (current_speed**2) / (2 * (distance - safe_distance))

        return required_deceleration

    def identify_conflict_points(
        self,
        trajectory1: np.ndarray,
        trajectory2: np.ndarray,
        vehicle_width: float = 2.0
    ) -> List[Dict]:
        """
        충돌 지점 식별

        두 차량의 궤적에서 충돌 가능성이 있는 지점을 찾습니다.

        Args:
            trajectory1: 첫 번째 차량의 궤적
            trajectory2: 두 번째 차량의 궤적
            vehicle_width: 차량 폭 (m)

        Returns:
            충돌 지점 리스트 [{'point': [x, y], 'time1': t1, 'time2': t2, 'distance': d}, ...]
        """
        conflict_points = []

        # 각 궤적을 폭을 고려한 Polygon으로 변환
        traj1_polygons = []
        traj2_polygons = []

        for i in range(len(trajectory1) - 1):
            segment = LineString([trajectory1[i], trajectory1[i + 1]])
            buffered = segment.buffer(vehicle_width / 2)
            traj1_polygons.append((buffered, i))

        for i in range(len(trajectory2) - 1):
            segment = LineString([trajectory2[i], trajectory2[i + 1]])
            buffered = segment.buffer(vehicle_width / 2)
            traj2_polygons.append((buffered, i))

        # 교차 검사
        for poly1, idx1 in traj1_polygons:
            for poly2, idx2 in traj2_polygons:
                intersection = poly1.intersection(poly2)

                if not intersection.is_empty:
                    # 교차점의 중심 계산
                    if intersection.geom_type == 'Point':
                        conflict_point = [intersection.x, intersection.y]
                    elif intersection.geom_type == 'Polygon':
                        centroid = intersection.centroid
                        conflict_point = [centroid.x, centroid.y]
                    else:
                        # MultiPoint 등
                        centroid = intersection.centroid
                        conflict_point = [centroid.x, centroid.y]

                    # 거리 계산
                    dist = np.linalg.norm(
                        np.array(trajectory1[idx1]) - np.array(trajectory2[idx2])
                    )

                    conflict_points.append({
                        'point': conflict_point,
                        'time1': idx1,
                        'time2': idx2,
                        'distance': dist
                    })

        return conflict_points


class SpatialHasher:
    """공간 해싱을 통한 효율적인 충돌 검사"""

    def __init__(self, cell_size: float = 10.0):
        """
        Args:
            cell_size: 격자 셀 크기 (m)
        """
        self.cell_size = cell_size
        self.grid = {}

    def _get_cell_key(self, x: float, y: float) -> Tuple[int, int]:
        """좌표를 셀 키로 변환"""
        cell_x = int(x / self.cell_size)
        cell_y = int(y / self.cell_size)
        return (cell_x, cell_y)

    def add_vehicle(self, vehicle_id: int, x: float, y: float):
        """차량을 격자에 추가"""
        cell_key = self._get_cell_key(x, y)
        if cell_key not in self.grid:
            self.grid[cell_key] = []
        self.grid[cell_key].append(vehicle_id)

    def get_nearby_vehicles(self, x: float, y: float, radius: float) -> List[int]:
        """반경 내의 차량 반환"""
        nearby = set()

        # 반경을 셀 단위로 변환
        cell_radius = int(np.ceil(radius / self.cell_size))

        center_cell = self._get_cell_key(x, y)

        # 주변 셀 검사
        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                cell_key = (center_cell[0] + dx, center_cell[1] + dy)
                if cell_key in self.grid:
                    nearby.update(self.grid[cell_key])

        return list(nearby)

    def clear(self):
        """격자 초기화"""
        self.grid.clear()


def analyze_frame_safety(
    frame_data: pd.DataFrame,
    calculator: SafetyMetricsCalculator,
    spatial_hasher: Optional[SpatialHasher] = None
) -> pd.DataFrame:
    """
    프레임 단위 안전 분석

    Args:
        frame_data: 프레임 데이터 (columns: track_id, x, y, vx, vy, ...)
        calculator: 안전 지표 계산기
        spatial_hasher: 공간 해싱 객체 (선택사항)

    Returns:
        안전 지표 결과 DataFrame
    """
    results = []

    # 공간 해싱 사용
    if spatial_hasher is None:
        spatial_hasher = SpatialHasher()

    spatial_hasher.clear()

    # 차량을 격자에 추가
    vehicles = {}
    for _, row in frame_data.iterrows():
        vehicle = VehicleState(
            x=row['x'],
            y=row['y'],
            vx=row.get('vx', 0.0),
            vy=row.get('vy', 0.0),
            width=row.get('width', 2.0),
            length=row.get('length', 4.5)
        )
        vehicles[row['track_id']] = vehicle
        spatial_hasher.add_vehicle(row['track_id'], row['x'], row['y'])

    # 각 차량 쌍에 대해 안전 지표 계산
    vehicle_ids = list(vehicles.keys())

    for i, id1 in enumerate(vehicle_ids):
        vehicle1 = vehicles[id1]

        # 공간 해싱으로 인접 차량만 검사
        nearby_ids = spatial_hasher.get_nearby_vehicles(
            vehicle1.x, vehicle1.y, radius=50.0  # 50m 반경
        )

        for id2 in nearby_ids:
            if id1 >= id2:  # 중복 방지
                continue

            vehicle2 = vehicles[id2]

            # TTC 계산
            ttc = calculator.calculate_2d_ttc(vehicle1, vehicle2)

            # DRAC 계산
            drac = calculator.calculate_drac(vehicle1, vehicle2, ttc)

            results.append({
                'vehicle1_id': id1,
                'vehicle2_id': id2,
                'ttc': ttc,
                'drac': drac,
                'distance': np.linalg.norm(vehicle1.position - vehicle2.position),
                'is_critical': ttc is not None and ttc < 3.0  # 3초 이하면 위험
            })

    return pd.DataFrame(results)


def main():
    """테스트용 메인 함수"""
    # 테스트 데이터
    vehicle1 = VehicleState(
        x=0.0, y=0.0,
        vx=5.0, vy=0.0,
        width=2.0, length=4.5
    )

    vehicle2 = VehicleState(
        x=10.0, y=0.0,
        vx=-5.0, vy=0.0,  # 반대 방향
        width=2.0, length=4.5
    )

    calculator = SafetyMetricsCalculator(vehicle_radius=2.5)

    # TTC 계산
    ttc = calculator.calculate_2d_ttc(vehicle1, vehicle2)
    print(f"TTC: {ttc:.2f}초" if ttc else "TTC: 충돌 없음")

    # DRAC 계산
    drac = calculator.calculate_drac(vehicle1, vehicle2, ttc)
    print(f"DRAC: {drac:.2f} m/s²" if drac and drac != float('inf') else "DRAC: 계산 불가")

    # 충돌 지점 식별
    traj1 = np.array([[0, 0], [5, 0], [10, 0]])
    traj2 = np.array([[10, 0], [5, 0], [0, 0]])
    conflicts = calculator.identify_conflict_points(traj1, traj2)
    print(f"\n충돌 지점 수: {len(conflicts)}")
    for i, conflict in enumerate(conflicts):
        print(f"  {i+1}. 위치: {conflict['point']}, 거리: {conflict['distance']:.2f}m")


if __name__ == "__main__":
    main()

