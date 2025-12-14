"""
로컬 완전 자동화 파이프라인
MacBook Air 및 일반 로컬 환경용

사용법:
    python scripts/local_auto_pipeline.py --mode fast
    python scripts/local_auto_pipeline.py --mode full
    python scripts/local_auto_pipeline.py --mode ultra_fast
"""

import argparse
import sys
import os
from pathlib import Path
import subprocess
import time
from typing import Dict, Optional
import yaml

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class LocalAutoPipeline:
    """로컬 완전 자동화 파이프라인"""

    def __init__(self, mode: str = "fast", data_dir: Optional[str] = None):
        """
        Args:
            mode: 실행 모드 ('ultra_fast', 'fast', 'full')
            data_dir: 데이터 디렉토리 (None이면 자동)
        """
        self.mode = mode
        self.project_root = project_root
        self.data_dir = Path(data_dir) if data_dir else self.project_root / "data"
        self.results_dir = self.project_root / "results"

        # 모드별 설정
        self.config = self._get_config()

    def _get_config(self) -> Dict:
        """모드별 설정"""
        if self.mode == "ultra_fast":
            return {
                "data_sample_ratio": 0.05,  # 5% 데이터
                "num_epochs": 10,
                "batch_size": 128,
                "eval_every": 20,
                "save_every": 20,
            }
        elif self.mode == "fast":
            return {
                "data_sample_ratio": 0.3,
                "num_epochs": 50,
                "batch_size": 64,
                "eval_every": 5,
                "save_every": 5,
            }
        else:  # full
            return {
                "data_sample_ratio": 1.0,
                "num_epochs": 100,
                "batch_size": 32,
                "eval_every": 10,
                "save_every": 10,
            }

    def step(self, step_num: int, total: int, name: str, func):
        """단계 실행 및 로깅"""
        print("\n" + "=" * 80)
        print(f"[{step_num}/{total}] {name}")
        print("=" * 80)
        start_time = time.time()

        try:
            result = func()
            elapsed = time.time() - start_time
            print(f"✓ 완료 ({elapsed:.1f}초)")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ 실패 ({elapsed:.1f}초)")
            print(f"오류: {e}")
            raise

    def check_environment(self):
        """1. 환경 확인"""
        print("\n[환경 확인]")

        # Python 버전 확인 및 경고
        python_version = sys.version_info
        print(f"Python 버전: {sys.version}")

        if python_version.major != 3 or python_version.minor not in [10, 11, 12]:
            if python_version.minor >= 13:
                print(
                    "\n⚠️  경고: Python 3.13+는 PyTorch와 호환성 문제가 있을 수 있습니다."
                )
                print("  권장: Python 3.10 사용")
            elif python_version.minor < 10:
                print("\n⚠️  경고: Python 3.9 이하는 지원되지 않습니다.")
                print("  권장: Python 3.10 사용")

        # Python 3.10 사용 권장
        if python_version.minor != 10:
            print(
                f"\n💡 권장: Python 3.10 사용 (현재: {python_version.major}.{python_version.minor})"
            )

        print(f"프로젝트 경로: {self.project_root}")

        # 필수 라이브러리 확인
        required_packages = [
            "torch",
            "torch_geometric",
            "pandas",
            "numpy",
            "yaml",
            "matplotlib",
        ]

        missing = []
        for pkg in required_packages:
            try:
                __import__(pkg.replace("-", "_"))
                print(f"  ✓ {pkg}")
            except ImportError:
                print(f"  ❌ {pkg} (누락)")
                missing.append(pkg)

        if missing:
            print(f"\n⚠️  누락된 패키지: {', '.join(missing)}")
            print("다음 명령어로 설치하세요:")
            print(f"  pip install {' '.join(missing)}")
            return False

        # GPU/MPS 확인
        try:
            import torch

            if torch.cuda.is_available():
                print(f"✓ GPU 사용 가능: {torch.cuda.get_device_name(0)}")
            elif torch.backends.mps.is_available():
                print("✓ MPS 사용 가능 (Apple Silicon)")
            else:
                print("⚠️  CPU 모드")
        except:
            print("⚠️  PyTorch 확인 실패")

        return True

    def check_data(self):
        """2. 데이터 확인"""
        print("\n[데이터 확인]")

        # 전처리된 데이터 우선 확인
        converted_dir = self.data_dir / "sdd" / "converted"
        if converted_dir.exists():
            csv_files = list(converted_dir.glob("*.csv"))
            if csv_files:
                print(f"✓ 전처리된 데이터 발견: {len(csv_files)}개 CSV 파일")
                total_size = sum(f.stat().st_size for f in csv_files)
                print(f"  크기: {total_size / 1024 / 1024:.1f} MB")
                return str(converted_dir)

        # 원본 데이터 확인
        sdd_dir = self.data_dir / "sdd" / "deathCircle"
        if sdd_dir.exists() and list(sdd_dir.glob("**/annotations.txt")):
            print(f"✓ 원본 SDD 데이터 발견: {sdd_dir}")
            print("  전처리 필요")
            return str(sdd_dir)

        # 데이터 없음
        print("❌ 데이터를 찾을 수 없습니다")
        print("\n다음 방법 중 하나를 선택하세요:")
        print("1. 수동 다운로드:")
        print("   https://github.com/flclain/StanfordDroneDataset")
        print("   → data/sdd/deathCircle/에 저장")
        print("\n2. Colab에서 다운로드 후 로컬로 복사")
        return None

    def preprocess_data(self, data_path: str):
        """3. 데이터 전처리 (CSV → 윈도우 생성)"""
        print("\n[데이터 전처리]")

        data_path_obj = Path(data_path)
        processed_dir = self.project_root / "data" / "processed"

        # 윈도우 pkl 파일이 이미 있는지 확인
        windows_file = processed_dir / "sdd_windows.pkl"
        if windows_file.exists():
            print(f"✓ 윈도우 파일 이미 존재: {windows_file}")
            return str(processed_dir)

        # CSV가 아닌 경우 (원본 데이터) - CSV로 변환
        if "converted" not in str(data_path_obj):
            converted_dir = self.data_dir / "sdd" / "converted"
            if not converted_dir.exists() or not list(converted_dir.glob("*.csv")):
                print("CSV 변환 중...")
                try:
                    from src.data_processing.sdd_adapter import SDDAdapter

                    adapter = SDDAdapter(data_path_obj)
                    converted_dir.mkdir(parents=True, exist_ok=True)
                    adapter.convert_all_videos(converted_dir)
                    print(f"✓ CSV 변환 완료")
                except Exception as e:
                    print(f"❌ CSV 변환 실패: {e}")
                    return None
            data_path_obj = converted_dir

        # CSV → 윈도우 pkl 변환
        print(f"\n[윈도우 생성] {data_path_obj} → {windows_file}")

        preprocess_script = self.project_root / "scripts" / "data" / "preprocess_sdd.py"
        if not preprocess_script.exists():
            print(f"❌ 전처리 스크립트 없음: {preprocess_script}")
            return None

        # preprocess_sdd.py 실행
        result = subprocess.run(
            [
                sys.executable,
                str(preprocess_script),
                "--data_dir",
                str(data_path_obj),
                "--output_dir",
                str(processed_dir),
                "--sample_ratio",
                str(self.config["data_sample_ratio"]),
            ],
            cwd=self.project_root,
        )

        if result.returncode == 0 and windows_file.exists():
            print(f"✓ 윈도우 생성 완료: {windows_file}")
            return str(processed_dir)
        else:
            print("❌ 윈도우 생성 실패")
            return None

    # ========================================================================
    # 베이스라인 학습 메서드 (주석 처리 - 나중에 비교 실험 시 사용)
    # ========================================================================
    # def train_baseline(self, data_dir: str, baseline_name: str = "a3tgcn"):
    #     """GNN 기반 베이스라인 모델 학습"""
    #     print(f"\n[GNN 모델 학습: {baseline_name.upper()}]")
    #
    #     if baseline_name == "a3tgcn":
    #         train_script = (
    #             self.project_root / "scripts" / "training" / "train_a3tgcn.py"
    #         )
    #         config_file = self.project_root / "configs" / "a3tgcn_config.yaml"
    #     else:
    #         print(f"⚠️  알 수 없는 베이스라인: {baseline_name}")
    #         return False
    #
    #     if not train_script.exists():
    #         print(f"⚠️  학습 스크립트 없음: {train_script}")
    #         return False
    #
    #     print(f"  설정 파일: {config_file}")
    #     print(f"  학습 스크립트: {train_script}")
    #
    #     result = subprocess.run(
    #         [
    #             sys.executable,
    #             str(train_script),
    #             "--config",
    #             str(config_file),
    #             "--data_dir",
    #             data_dir,
    #         ],
    #         cwd=self.project_root,
    #     )
    #
    #     if result.returncode == 0:
    #         print(f"\n✓ {baseline_name.upper()} 학습 완료")
    #         return True
    #     else:
    #         print(f"\n⚠️  {baseline_name.upper()} 학습 중 오류 발생")
    #         return False

    def train_model(self, data_dir: str):
        """MID 모델 학습 (GNN 다음 단계)"""
        print("\n[모델 학습: MID]")

        # 설정 파일 (모드별 우선순위)
        config_files = [
            self.project_root / "configs" / f"mid_config_{self.mode}.yaml",
            self.project_root / "configs" / "mid_config_fast.yaml",
            self.project_root / "configs" / "mid_config_standard.yaml",
        ]

        config_file = None
        for cfg in config_files:
            if cfg.exists():
                config_file = cfg
                break

        if not config_file:
            print(f"⚠️  설정 파일 없음: {config_files[0]}")
            print("기본 설정 파일을 찾을 수 없습니다.")
            return False

        print(f"✓ 설정 파일: {config_file}")

        # 학습 스크립트
        train_script = self.project_root / "scripts" / "training" / "train_mid.py"

        if not train_script.exists():
            print(f"❌ 학습 스크립트 없음: {train_script}")
            return False

        print(f"✓ 학습 스크립트: {train_script}")
        print("\n[학습 시작]")
        print("=" * 80)

        # 학습 실행
        result = subprocess.run(
            [
                sys.executable,
                str(train_script),
                "--config",
                str(config_file),
                "--data_dir",
                data_dir,
            ],
            cwd=self.project_root,
        )

        if result.returncode == 0:
            print("\n✓ 학습 완료")
            return True
        else:
            print("\n⚠️  학습 중 오류 발생")
            return False

    def visualize_results(self):
        """5. 결과 시각화"""
        print("\n[결과 시각화]")

        viz_script = self.project_root / "scripts" / "utils" / "visualize_results.py"

        if not viz_script.exists():
            print("⚠️  시각화 스크립트 없음")
            return

        result = subprocess.run(
            [sys.executable, str(viz_script)], cwd=self.project_root
        )

        if result.returncode == 0:
            print("✓ 시각화 완료")
            print(f"  결과: {self.results_dir / 'visualizations'}")
        else:
            print("⚠️  시각화 중 오류 발생")

    def run(self):
        """전체 파이프라인 실행"""
        print("=" * 80)
        print("로컬 완전 자동화 파이프라인")
        print("=" * 80)
        print(f"모드: {self.mode}")
        print(f"프로젝트: {self.project_root}")
        print("=" * 80)

        # 1. 환경 확인
        try:
            env_ok = self.step(1, 5, "환경 확인", self.check_environment)
            if not env_ok:
                print("\n환경 설정이 필요합니다. 먼저 필수 패키지를 설치하세요.")
                return False
        except Exception as e:
            print(f"환경 확인 실패: {e}")
            return False

        # 2. 데이터 확인
        try:
            data_path = self.step(2, 5, "데이터 확인", self.check_data)
            if not data_path:
                print("\n데이터를 먼저 준비해주세요.")
                return False
        except Exception as e:
            print(f"데이터 확인 실패: {e}")
            return False

        # 3. 데이터 전처리
        try:
            processed_dir = self.step(
                3, 5, "데이터 전처리", lambda: self.preprocess_data(data_path)
            )
            if not processed_dir:
                print("\n데이터 전처리 실패")
                return False
        except Exception as e:
            print(f"데이터 전처리 실패: {e}")
            return False

        # 4. MID 모델 학습 (HSG-Diffusion 핵심)
        try:
            success = self.step(
                4,
                5,
                "MID 모델 학습",
                lambda: self.train_model(processed_dir),
            )
            if not success:
                print("⚠️  MID 학습 실패")
                return False
        except Exception as e:
            print(f"MID 모델 학습 실패: {e}")
            return False

        # 5. 결과 시각화
        try:
            self.step(5, 5, "결과 시각화", self.visualize_results)
        except Exception as e:
            print(f"시각화 실패: {e}")

        print("\n" + "=" * 80)
        print("✓ 전체 파이프라인 완료!")
        print("=" * 80)
        print(f"\n결과 위치:")
        print(f"  체크포인트: checkpoints/mid/")
        print(f"  TensorBoard: runs/mid/")
        print(f"  시각화: results/visualizations/")
        print(f"\nTensorBoard 실행:")
        print(f"  tensorboard --logdir runs/mid")

        return True


def main():
    parser = argparse.ArgumentParser(description="로컬 완전 자동화 파이프라인")
    parser.add_argument(
        "--mode",
        type=str,
        default="fast",
        choices=["ultra_fast", "fast", "full"],
        help="실행 모드 (ultra_fast: 1시간, fast: 2-3시간, full: 6-8시간)",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None, help="데이터 디렉토리 경로"
    )

    args = parser.parse_args()

    pipeline = LocalAutoPipeline(mode=args.mode, data_dir=args.data_dir)

    success = pipeline.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
