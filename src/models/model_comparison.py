"""
모델 비교 및 선택 유틸리티
세 가지 모델 (A3TGCN, Trajectron++, Social-STGCNN)의 특징과 적용 가능성 분석
"""

from typing import Dict, List
from dataclasses import dataclass
from enum import Enum


class ModelType(Enum):
    """모델 타입 열거형"""
    A3TGCN = "A3TGCN"
    TRAJECTRON_PP = "Trajectron++"
    SOCIAL_STGCNN = "Social-STGCNN"


@dataclass
class ModelCharacteristics:
    """모델 특성 데이터 클래스"""
    name: str
    library: str
    supports_map: bool
    supports_dynamics: bool
    supports_multimodal: bool
    implementation_difficulty: int  # 1-5 (5가 가장 어려움)
    training_speed: int  # 1-5 (5가 가장 빠름)
    inference_speed: int  # 1-5 (5가 가장 빠름)
    roundabout_suitability: int  # 1-5 (5가 가장 적합)
    pros: List[str]
    cons: List[str]


class ModelComparator:
    """모델 비교 클래스"""

    def __init__(self):
        self.models = {
            ModelType.A3TGCN: ModelCharacteristics(
                name="A3TGCN",
                library="PyTorch Geometric Temporal",
                supports_map=False,
                supports_dynamics=False,
                supports_multimodal=False,
                implementation_difficulty=3,
                training_speed=3,
                inference_speed=4,
                roundabout_suitability=3,
                pros=[
                    "어텐션 메커니즘으로 시간적 중요도 학습",
                    "PyTorch Geometric Temporal에 내장",
                    "동적 그래프 처리 지원",
                    "구현이 상대적으로 간단"
                ],
                cons=[
                    "HD Map 정보 직접 활용 어려움",
                    "차량 동역학 제약 반영 제한적"
                ]
            ),
            ModelType.TRAJECTRON_PP: ModelCharacteristics(
                name="Trajectron++",
                library="Stanford ASL",
                supports_map=True,
                supports_dynamics=True,
                supports_multimodal=True,
                implementation_difficulty=2,
                training_speed=2,
                inference_speed=3,
                roundabout_suitability=5,
                pros=[
                    "HD Map 정보 직접 인코딩 (Lanelet2 지원)",
                    "차량 동역학 모델 통합",
                    "다중 모드 예측 (CVAE)",
                    "trajdata 라이브러리와 통합 가능"
                ],
                cons=[
                    "구현 복잡도 높음",
                    "학습 시간 길음",
                    "메모리 사용량 큼"
                ]
            ),
            ModelType.SOCIAL_STGCNN: ModelCharacteristics(
                name="Social-STGCNN",
                library="GitHub (abduallahmohamed)",
                supports_map=False,
                supports_dynamics=False,
                supports_multimodal=False,
                implementation_difficulty=4,
                training_speed=5,
                inference_speed=5,
                roundabout_suitability=2,
                pros=[
                    "매우 빠른 추론 속도",
                    "적은 파라미터 수",
                    "적은 데이터로도 학습 가능"
                ],
                cons=[
                    "보행자 중심 설계 (차량에 부적합)",
                    "맵 정보 미반영",
                    "동역학 제약 없음"
                ]
            )
        }

    def compare_models(self) -> Dict:
        """모든 모델을 비교하여 결과 반환"""
        comparison = {}

        for model_type, characteristics in self.models.items():
            comparison[model_type.value] = {
                "특성": {
                    "라이브러리": characteristics.library,
                    "맵 지원": "✅" if characteristics.supports_map else "❌",
                    "동역학 지원": "✅" if characteristics.supports_dynamics else "❌",
                    "다중 모드": "✅" if characteristics.supports_multimodal else "❌"
                },
                "성능": {
                    "구현 난이도": "⭐" * characteristics.implementation_difficulty,
                    "학습 속도": "⭐" * characteristics.training_speed,
                    "추론 속도": "⭐" * characteristics.inference_speed,
                    "회전교차로 적합성": "⭐" * characteristics.roundabout_suitability
                },
                "장점": characteristics.pros,
                "단점": characteristics.cons
            }

        return comparison

    def recommend_model(self, priority: str = "roundabout_suitability") -> ModelType:
        """
        우선순위에 따라 모델 추천

        Args:
            priority: 우선순위 기준
                - "roundabout_suitability": 회전교차로 적합성
                - "implementation_difficulty": 구현 난이도 (낮을수록 좋음)
                - "training_speed": 학습 속도
                - "inference_speed": 추론 속도

        Returns:
            추천 모델 타입
        """
        if priority == "roundabout_suitability":
            return max(
                self.models.items(),
                key=lambda x: x[1].roundabout_suitability
            )[0]
        elif priority == "implementation_difficulty":
            return min(
                self.models.items(),
                key=lambda x: x[1].implementation_difficulty
            )[0]
        elif priority == "training_speed":
            return max(
                self.models.items(),
                key=lambda x: x[1].training_speed
            )[0]
        elif priority == "inference_speed":
            return max(
                self.models.items(),
                key=lambda x: x[1].inference_speed
            )[0]
        else:
            raise ValueError(f"Unknown priority: {priority}")

    def get_model_info(self, model_type: ModelType) -> ModelCharacteristics:
        """특정 모델의 정보 반환"""
        return self.models[model_type]

    def print_comparison(self):
        """비교 결과를 출력"""
        print("=" * 80)
        print("모델 비교 분석")
        print("=" * 80)
        print()

        comparison = self.compare_models()

        for model_name, info in comparison.items():
            print(f"\n{'='*80}")
            print(f"모델: {model_name}")
            print(f"{'='*80}")

            print("\n[특성]")
            for key, value in info["특성"].items():
                print(f"  {key}: {value}")

            print("\n[성능]")
            for key, value in info["성능"].items():
                print(f"  {key}: {value}")

            print("\n[장점]")
            for pro in info["장점"]:
                print(f"  ✓ {pro}")

            print("\n[단점]")
            for con in info["단점"]:
                print(f"  ✗ {con}")

        print(f"\n{'='*80}")
        print("추천 모델")
        print(f"{'='*80}")

        recommended = self.recommend_model("roundabout_suitability")
        print(f"\n1순위 (회전교차로 적합성): {recommended.value}")
        info = self.get_model_info(recommended)
        print(f"  적합성 점수: {'⭐' * info.roundabout_suitability}")
        print(f"  주요 장점: {', '.join(info.pros[:2])}")

        print(f"\n2순위 (구현 용이성): {self.recommend_model('implementation_difficulty').value}")
        print(f"\n3순위 (추론 속도): {self.recommend_model('inference_speed').value}")


def main():
    """메인 함수"""
    comparator = ModelComparator()
    comparator.print_comparison()


if __name__ == "__main__":
    main()

