"""
Trajectron++ 통합 가이드 및 래퍼
trajdata를 통한 INTERACTION Dataset 로딩 및 Trajectron++ 모델 사용
"""

from typing import Optional, Dict, Any, Union, TYPE_CHECKING
from pathlib import Path
import warnings

# Trajectron++는 별도 설치가 필요하므로 선택적 import
if TYPE_CHECKING:
    # 타입 체크 시에만 import (실제 런타임에는 조건부)
    try:
        from trajdata import UnifiedDataset, AgentBatch, SceneBatch
    except ImportError:
        UnifiedDataset = Any
        AgentBatch = Any
        SceneBatch = Any

try:
    from trajdata import UnifiedDataset, AgentBatch, SceneBatch

    TRAJDATA_AVAILABLE = True
except ImportError:
    TRAJDATA_AVAILABLE = False
    # 타입 힌트용 더미 클래스
    UnifiedDataset = Any
    AgentBatch = Any
    SceneBatch = Any
    warnings.warn(
        "trajdata가 설치되지 않았습니다. Trajectron++ 통합을 사용할 수 없습니다."
    )


class TrajectronIntegration:
    """
    Trajectron++ 모델 통합 클래스
    trajdata를 통한 데이터 로딩 및 모델 설정
    """

    def __init__(self, data_dir: Path):
        """
        Args:
            data_dir: INTERACTION Dataset 디렉토리 경로
        """
        if not TRAJDATA_AVAILABLE:
            raise ImportError(
                "trajdata가 설치되지 않았습니다. " "설치: pip install trajdata"
            )

        self.data_dir = Path(data_dir)
        self.dataset = None

    def load_dataset(
        self, scenario_name: Optional[str] = None, cache_location: Optional[Path] = None
    ) -> UnifiedDataset:
        """
        INTERACTION Dataset을 로드합니다.

        Args:
            scenario_name: 특정 시나리오 이름 (None이면 전체)
            cache_location: 캐시 저장 위치

        Returns:
            UnifiedDataset 객체
        """
        if scenario_name:
            desired_data = [f"interaction_{scenario_name}"]
        else:
            desired_data = ["interaction_multi"]  # 모든 INTERACTION 데이터

        data_dirs = {"interaction_multi": str(self.data_dir)}

        self.dataset = UnifiedDataset(
            desired_data=desired_data,
            data_dirs=data_dirs,
            cache_location=str(cache_location) if cache_location else None,
        )

        return self.dataset

    def get_config_template(self) -> Dict[str, Any]:
        """
        Trajectron++ 설정 템플릿 반환
        회전교차로 연구에 최적화된 설정
        """
        return {
            "model": {
                "name": "trajectron++",
                "hyperparams": {
                    "k": 1,  # 예측 모드 수 (CVAE)
                    "k_eval": 25,  # 평가 시 샘플 수
                    "history_length": 8,  # 과거 0.8초 (10Hz 기준)
                    "future_length": 12,  # 미래 1.2초
                    "map_encoding": True,  # HD Map 인코딩 활성화
                    "dynamics_integration": True,  # 동역학 모델 통합
                    "state": {
                        "P": ["x", "y"],  # 위치
                        "V": ["x", "y"],  # 속도
                        "A": ["x", "y"],  # 가속도
                    },
                },
            },
            "data": {
                "train": {"batch_size": 32, "shuffle": True},
                "val": {"batch_size": 32, "shuffle": False},
            },
            "training": {
                "optimizer": "adam",
                "learning_rate": 1e-3,
                "num_epochs": 100,
                "early_stopping": {"patience": 10, "min_delta": 0.001},
            },
        }

    def create_config_file(self, output_path: Path):
        """
        설정 파일 생성

        Args:
            output_path: 출력 경로
        """
        import yaml

        config = self.get_config_template()

        with open(output_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        print(f"설정 파일 생성 완료: {output_path}")


def check_trajectron_installation() -> Dict[str, bool]:
    """
    Trajectron++ 및 관련 라이브러리 설치 여부 확인

    Returns:
        설치 여부 딕셔너리
    """
    results = {"trajdata": False, "trajectron": False}

    try:
        import trajdata

        results["trajdata"] = True
    except ImportError:
        pass

    try:
        # Trajectron++는 별도 저장소에서 설치 필요
        # 일반적으로: pip install git+https://github.com/StanfordASL/Trajectron-plus-plus.git
        import trajectron

        results["trajectron"] = True
    except ImportError:
        pass

    return results


def print_installation_guide():
    """Trajectron++ 설치 가이드 출력"""
    print("=" * 80)
    print("Trajectron++ 설치 가이드")
    print("=" * 80)
    print()

    print("1. trajdata 설치:")
    print("   pip install trajdata")
    print()

    print("2. Trajectron++ 설치:")
    print("   git clone https://github.com/StanfordASL/Trajectron-plus-plus.git")
    print("   cd Trajectron-plus-plus")
    print("   pip install -e .")
    print()

    print("3. Lanelet2 설치 (맵 처리용):")
    print("   conda install -c conda-forge lanelet2")
    print("   또는")
    print("   pip install lanelet2  # (소스 빌드 필요)")
    print()

    print("4. 설치 확인:")
    checks = check_trajectron_installation()
    for lib, installed in checks.items():
        status = "✓" if installed else "✗"
        print(f"   {status} {lib}")
    print()


def main():
    """메인 함수"""
    print_installation_guide()

    # 설정 템플릿 생성 예시
    integration = TrajectronIntegration(Path("./data/interaction"))
    config = integration.get_config_template()

    print("\n설정 템플릿:")
    print(f"  맵 인코딩: {config['model']['hyperparams']['map_encoding']}")
    print(f"  동역학 통합: {config['model']['hyperparams']['dynamics_integration']}")
    print(f"  학습률: {config['training']['learning_rate']}")


if __name__ == "__main__":
    main()
