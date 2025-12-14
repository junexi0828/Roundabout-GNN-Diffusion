"""
Colab 완전 자동화 파이프라인
하나의 스크립트로 전체 프로세스 자동 실행

사용법:
    !python scripts/colab/colab_auto_pipeline.py --mode fast
    !python scripts/colab/colab_auto_pipeline.py --mode full
"""

import argparse
import sys
import os
from pathlib import Path
import subprocess
import time
from typing import Dict, Optional
import json

# Colab 환경 확인
try:
    import google.colab

    IN_COLAB = True
except ImportError:
    IN_COLAB = False
    print("⚠️  Colab 환경이 아닙니다. 로컬에서 실행됩니다.")


class ColabAutoPipeline:
    """Colab 완전 자동화 파이프라인"""

    def __init__(
        self,
        mode: str = "fast",
        drive_mount: bool = True,
        data_dir: Optional[str] = None,
        github_repo: str = "https://github.com/junexi0828/Roundabout-GNN-Diffusion.git",
    ):
        """
        Args:
            mode: 실행 모드 ('fast' 또는 'full')
            drive_mount: Google Drive 마운트 여부
            data_dir: 데이터 디렉토리 (None이면 자동 감지)
            github_repo: GitHub 저장소 URL
        """
        self.mode = mode
        self.drive_mount = drive_mount and IN_COLAB
        self.github_repo = github_repo

        # 경로 설정
        if IN_COLAB:
            # Colab에서 현재 작업 디렉토리 자동 감지
            current_dir = Path.cwd()
            # 가능한 프로젝트 디렉토리 확인
            possible_roots = [
                current_dir,
                Path("/content/Roundabout_AI"),
                Path("/content/Roundabout-GNN-Diffusion"),
            ]

            self.project_root = None
            for root in possible_roots:
                if root.exists() and (root / "src").exists() and (root / "scripts").exists():
                    self.project_root = root
                    break

            if self.project_root is None:
                # 기본값 사용
                self.project_root = current_dir if (current_dir / "src").exists() else Path("/content/Roundabout_AI")

            self.drive_root = Path("/content/drive/MyDrive")
        else:
            self.project_root = Path(__file__).parent.parent.parent
            self.drive_root = Path.home() / "Google Drive"

        # sys.path에 프로젝트 루트 추가
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))

        self.data_dir = (
            Path(data_dir) if data_dir else self.drive_root / "Roundabout_AI_Data"
        )
        self.results_dir = self.project_root / "results"

        # 모드별 설정
        self.config = self._get_config()

    def _get_config(self) -> Dict:
        """모드별 설정"""
        if self.mode == "fast":
            return {
                "data_sample_ratio": 0.3,  # 30% 데이터만 사용
                "num_epochs": 20,
                "batch_size": 16,
                "eval_every": 5,
                "save_every": 5,
            }
        else:  # full
            return {
                "data_sample_ratio": 1.0,  # 전체 데이터
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

    def setup_environment(self):
        """1. 환경 설정"""
        print("\n[환경 설정]")

        # 시스템 정보
        print(f"Python 버전: {sys.version}")
        print(f"프로젝트 경로: {self.project_root}")

        # 필수 라이브러리 설치
        print("\n[라이브러리 설치]")
        packages = [
            "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
            "torch-geometric",
            "pandas numpy scipy scikit-learn",
            "matplotlib seaborn opencv-python",
            "networkx tqdm pyyaml shapely tensorboard",
            "xxhash aiohttp psutil requests",
        ]

        for pkg in packages:
            print(f"  설치 중: {pkg.split()[0]}...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q"] + pkg.split(), check=True
            )

        print("✓ 라이브러리 설치 완료")

        # GPU 확인
        try:
            import torch

            if torch.cuda.is_available():
                print(f"✓ GPU 사용 가능: {torch.cuda.get_device_name(0)}")
                print(
                    f"  메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
                )
            else:
                print("⚠️  GPU를 사용할 수 없습니다 (CPU 모드)")
        except:
            print("⚠️  PyTorch 확인 실패")

    def clone_repository(self):
        """2. GitHub 저장소 클론"""
        # 이미 클론되어 있고 src 디렉토리가 있으면 건너뛰기
        if self.project_root.exists() and (self.project_root / "src").exists():
            print(f"✓ 프로젝트 이미 존재: {self.project_root}")
            if (self.project_root / ".git").exists():
                # 최신 버전으로 업데이트
                print("  최신 버전으로 업데이트 중...")
                subprocess.run(["git", "pull"], cwd=self.project_root, check=False)
            return

        print(f"\n[GitHub 저장소 클론]")
        print(f"  저장소: {self.github_repo}")
        print(f"  경로: {self.project_root}")

        subprocess.run(
            ["git", "clone", self.github_repo, str(self.project_root)], check=True
        )
        print("✓ 저장소 클론 완료")

    def mount_drive(self):
        """3. Google Drive 마운트"""
        if not self.drive_mount:
            print("⚠️  Google Drive 마운트 건너뜀")
            return

        print("\n[Google Drive 마운트]")
        try:
            from google.colab import drive

            drive.mount("/content/drive")
            print("✓ Google Drive 마운트 완료")
            print(f"  경로: {self.drive_root}")
        except Exception as e:
            print(f"⚠️  Google Drive 마운트 실패: {e}")
            print(
                "  수동 마운트: from google.colab import drive; drive.mount('/content/drive')"
            )

    def prepare_data(self):
        """4. 데이터 준비"""
        print("\n[데이터 준비]")

        # 여러 가능한 경로 확인
        possible_paths = [
            self.data_dir,
            self.drive_root / "Roundabout_AI_Data",
            self.drive_root / "Roundabout_AI" / "data" / "sdd",
            self.project_root / "data" / "sdd" / "converted",
            self.project_root / "data" / "sdd" / "deathCircle",
        ]

        data_path = None
        for path in possible_paths:
            path_obj = Path(path)
            if path_obj.exists():
                # CSV 파일 또는 annotations.txt 확인
                if path_obj.is_dir():
                    csv_files = list(path_obj.glob("*.csv"))
                    ann_files = list(path_obj.glob("**/annotations.txt"))
                    if csv_files or ann_files:
                        print(f"✓ 데이터 디렉토리 발견: {path_obj}")
                        data_path = path_obj
                        break
                elif path_obj.is_file():
                    print(f"✓ 데이터 파일 발견: {path_obj}")
                    data_path = path_obj.parent
                    break

        if data_path is None:
            print(f"⚠️  데이터 디렉토리 없음")
            print("\n📋 수동 작업 필요:")
            print("1. SDD Death Circle 비디오 다운로드")
            print("2. Google Drive에 업로드")
            print(f"3. 가능한 경로:")
            for path in possible_paths:
                print(f"   - {path}")
            print("\n기다리는 중... (30초)")
            time.sleep(30)

            # 다시 확인
            for path in possible_paths:
                path_obj = Path(path)
                if path_obj.exists():
                    csv_files = list(path_obj.glob("*.csv"))
                    ann_files = list(path_obj.glob("**/annotations.txt"))
                    if csv_files or ann_files:
                        data_path = path_obj
                        break

        if data_path:
            return str(data_path)
        else:
            print("⚠️  데이터를 찾을 수 없습니다. 기본 경로 사용")
            return "data/sdd/converted"

    def preprocess_data(self, data_path: str):
        """5. 데이터 전처리"""
        print("\n[데이터 전처리]")

        # 전처리 스크립트 실행
        preprocess_script = self.project_root / "scripts" / "data" / "preprocess_sdd.py"

        if not preprocess_script.exists():
            print("⚠️  전처리 스크립트 없음, 기본 전처리 수행")
            # 간단한 전처리
            try:
                from src.data_processing.preprocessor import TrajectoryPreprocessor
                from src.integration.sdd_data_adapter import SDDDataAdapter
                import pandas as pd

                # 데이터 로드 및 전처리
                adapter = SDDDataAdapter()
                data_path_obj = Path(data_path) if data_path else None
                windows = adapter.load_and_preprocess(data_path_obj)

                # 저장
                output_dir = self.project_root / "data" / "processed"
                output_dir.mkdir(parents=True, exist_ok=True)

                import pickle

                with open(output_dir / "sdd_windows.pkl", "wb") as f:
                    pickle.dump(windows, f)

                print(f"✓ 전처리 완료: {len(windows)}개 윈도우")
                return str(output_dir)
            except Exception as e:
                print(f"⚠️  전처리 실패: {e}")
                print("  이미 전처리된 데이터 사용")
                return data_path

        # 전처리 스크립트 실행
        result = subprocess.run(
            [sys.executable, str(preprocess_script), "--data_dir", data_path],
            cwd=self.project_root,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("✓ 전처리 완료")
            return "data/processed"
        else:
            print(f"⚠️  전처리 오류: {result.stderr}")
            return "data/processed"

    def train_baseline(self, data_dir: str, baseline_name: str = "a3tgcn"):
        """베이스라인 모델 학습"""
        print(f"\n[베이스라인 학습: {baseline_name.upper()}]")

        if baseline_name == "a3tgcn":
            train_script = self.project_root / "scripts" / "training" / "train_a3tgcn.py"
            config_file = self.project_root / "configs" / "a3tgcn_config.yaml"
        else:
            print(f"⚠️  알 수 없는 베이스라인: {baseline_name}")
            return False

        if not train_script.exists():
            print(f"⚠️  학습 스크립트 없음: {train_script}")
            return False

        print(f"  설정 파일: {config_file}")
        print(f"  학습 스크립트: {train_script}")

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
            print(f"\n✓ {baseline_name.upper()} 학습 완료")
            return True
        else:
            print(f"\n⚠️  {baseline_name.upper()} 학습 중 오류 발생")
            return False

    def train_model(self, data_dir: str):
        """6. 모델 학습 (HSG-Diffusion)"""
        print("\n[모델 학습: HSG-Diffusion]")

        # 설정 파일 로드 또는 생성
        import yaml

        config_file = self.project_root / "configs" / f"mid_config_{self.mode}.yaml"

        if config_file.exists():
            print(f"✓ 설정 파일 로드: {config_file}")
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
            config_path = config_file
        else:
            print(f"⚠️  설정 파일 없음, 기본 설정 생성")
            # 설정 파일 생성
            config = {
                "model": {
                    "name": "mid",
                    "obs_steps": 30,
                    "pred_steps": 50,
                    "hidden_dim": 128,
                    "num_diffusion_steps": 100,
                    "use_gnn": True,
                    "node_features": 9,
                },
                "data": {
                    "data_dir": data_dir,
                    "batch_size": self.config["batch_size"],
                    "train_ratio": 0.7,
                    "val_ratio": 0.15,
                    "test_ratio": 0.15,
                },
                "training": {
                    "optimizer": "adamw",
                    "learning_rate": 0.001,
                    "num_epochs": self.config["num_epochs"],
                    "use_amp": True,
                    "early_stopping": {"patience": 15, "min_delta": 0.001},
                },
                "evaluation": {
                    "num_samples": 20,
                    "ddim_steps": 2,
                    "eval_every": self.config["eval_every"],
                },
                "logging": {"log_dir": "runs/mid", "save_dir": "checkpoints/mid"},
            }

            # 설정 파일 저장
            config_path = self.project_root / "configs" / "colab_auto_config.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            print(f"✓ 설정 파일 생성: {config_path}")

        print(f"  에폭: {config['training']['num_epochs']}")
        print(f"  배치 크기: {config['data']['batch_size']}")
        print(f"  Diffusion 스텝: {config['model']['num_diffusion_steps']}")

        # 학습 실행
        train_script = self.project_root / "scripts" / "training" / "train_mid.py"

        print("\n[학습 시작]")
        print("=" * 80)

        result = subprocess.run(
            [
                sys.executable,
                str(train_script),
                "--config",
                str(config_path),
                "--data_dir",
                data_dir,
            ],
            cwd=self.project_root,
        )

        if result.returncode == 0:
            print("\n✓ 학습 완료")
        else:
            print("\n⚠️  학습 중 오류 발생")

        return result.returncode == 0

    def compare_baselines(self):
        """베이스라인 비교 평가"""
        print("\n[베이스라인 비교 평가]")

        # 비교 평가 스크립트 실행
        compare_script = self.project_root / "scripts" / "evaluation" / "compare_baselines.py"

        if not compare_script.exists():
            print("⚠️  비교 평가 스크립트 없음")
            return

        # 체크포인트 경로 확인
        mid_checkpoint = self.project_root / "checkpoints" / "mid" / "best_model.pth"
        a3tgcn_checkpoint = (
            self.project_root / "checkpoints" / "a3tgcn" / "best_model.pth"
        )

        if not mid_checkpoint.exists():
            print(f"⚠️  MID 체크포인트 없음: {mid_checkpoint}")
        if not a3tgcn_checkpoint.exists():
            print(f"⚠️  A3TGCN 체크포인트 없음: {a3tgcn_checkpoint}")

        result = subprocess.run(
            [
                sys.executable,
                str(compare_script),
                "--mid_checkpoint",
                str(mid_checkpoint),
                "--a3tgcn_checkpoint",
                str(a3tgcn_checkpoint),
                "--data_dir",
                "data/processed",
                "--output_dir",
                "results/comparison",
            ],
            cwd=self.project_root,
        )

        if result.returncode == 0:
            print("✓ 비교 평가 완료")
        else:
            print("⚠️  비교 평가 중 오류 발생")

    def visualize_results(self):
        """8. 결과 시각화"""
        print("\n[결과 시각화]")

        # 시각화 스크립트 실행
        viz_script = self.project_root / "scripts" / "utils" / "visualize_results.py"

        if not viz_script.exists():
            print("⚠️  시각화 스크립트 없음")
            return

        result = subprocess.run(
            [sys.executable, str(viz_script)], cwd=self.project_root
        )

        if result.returncode == 0:
            print("✓ 시각화 완료")
        else:
            print("⚠️  시각화 중 오류 발생")

    def save_results(self):
        """9. 결과 저장 (Google Drive)"""
        if not self.drive_mount:
            print("⚠️  Google Drive 마운트 안 됨, 로컬에만 저장")
            return

        print("\n[결과 저장]")

        # 결과 디렉토리 생성
        drive_results = (
            self.drive_root / "Roundabout_AI_Results" / time.strftime("%Y%m%d_%H%M%S")
        )
        drive_results.mkdir(parents=True, exist_ok=True)

        # 결과 복사
        import shutil

        results_to_copy = [
            ("checkpoints", "checkpoints"),
            ("results", "results"),
            ("runs", "runs"),
        ]

        for src_name, dst_name in results_to_copy:
            src = self.project_root / src_name
            dst = drive_results / dst_name

            if src.exists():
                print(f"  복사 중: {src_name} → {dst_name}")
                shutil.copytree(src, dst, dirs_exist_ok=True)

        print(f"✓ 결과 저장 완료: {drive_results}")
        return str(drive_results)

    def run(self):
        """전체 파이프라인 실행"""
        print("=" * 80)
        print("Colab 완전 자동화 파이프라인")
        print("=" * 80)
        print(f"모드: {self.mode}")
        print(f"프로젝트: {self.project_root}")
        print("=" * 80)

        steps = [
            (1, 10, "환경 설정", self.setup_environment),
            (2, 10, "GitHub 저장소 클론", self.clone_repository),
            (3, 10, "Google Drive 마운트", self.mount_drive),
            (4, 10, "데이터 준비", self.prepare_data),
        ]

        # 단계별 실행
        data_path = None
        for step_num, total, name, func in steps:
            try:
                result = self.step(step_num, total, name, func)
                if step_num == 4:  # 데이터 준비
                    data_path = result
            except Exception as e:
                print(f"❌ {name} 실패: {e}")
                return False

        # 데이터 전처리
        try:
            processed_dir = self.step(
                5, 10, "데이터 전처리", lambda: self.preprocess_data(data_path)
            )
        except Exception as e:
            print(f"❌ 데이터 전처리 실패: {e}")
            return False

        # 모델 학습 (HSG-Diffusion)
        try:
            success = self.step(
                6,
                10,
                "모델 학습 (HSG-Diffusion)",
                lambda: self.train_model(processed_dir),
            )
            if not success:
                print("⚠️  학습 실패했지만 계속 진행합니다...")
        except Exception as e:
            print(f"❌ 모델 학습 실패: {e}")
            return False

        # 베이스라인 학습 (A3TGCN)
        try:
            baseline_success = self.step(
                7,
                10,
                "베이스라인 학습 (A3TGCN)",
                lambda: self.train_baseline(processed_dir, "a3tgcn"),
            )
            if not baseline_success:
                print("⚠️  베이스라인 학습 실패했지만 계속 진행합니다...")
        except Exception as e:
            print(f"⚠️  베이스라인 학습 실패: {e}")

        # 비교 평가
        try:
            self.step(8, 10, "베이스라인 비교 평가", self.compare_baselines)
        except Exception as e:
            print(f"⚠️  비교 평가 실패: {e}")

        # 시각화
        try:
            self.step(9, 10, "결과 시각화", self.visualize_results)
        except Exception as e:
            print(f"⚠️  시각화 실패: {e}")

        # 결과 저장
        try:
            self.step(10, 10, "결과 저장", self.save_results)
        except Exception as e:
            print(f"⚠️  결과 저장 실패: {e}")

        print("\n" + "=" * 80)
        print("✓ 전체 파이프라인 완료!")
        print("=" * 80)

        return True


def main():
    parser = argparse.ArgumentParser(description="Colab 완전 자동화 파이프라인")
    parser.add_argument(
        "--mode",
        type=str,
        default="fast",
        choices=["fast", "full"],
        help="실행 모드 (fast: 빠른 테스트, full: 전체 학습)",
    )
    parser.add_argument(
        "--no-drive", action="store_true", help="Google Drive 마운트 안 함"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None, help="데이터 디렉토리 경로"
    )

    args = parser.parse_args()

    pipeline = ColabAutoPipeline(
        mode=args.mode, drive_mount=not args.no_drive, data_dir=args.data_dir
    )

    success = pipeline.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
