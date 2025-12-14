"""
학습 파이프라인 구현
손실 함수, Optimizer, Learning Rate Scheduler, Early Stopping, TensorBoard 로깅
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json


class EarlyStopping:
    """Early Stopping 구현"""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        """
        Args:
            patience: 개선이 없을 때 기다릴 에폭 수
            min_delta: 개선으로 간주할 최소 변화량
            mode: 'min' (손실 최소화) 또는 'max' (점수 최대화)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        if mode == 'min':
            self.is_better = lambda current, best: current < best - min_delta
        else:
            self.is_better = lambda current, best: current > best + min_delta

    def __call__(self, score: float) -> bool:
        """
        Args:
            score: 현재 점수 (손실 또는 메트릭)

        Returns:
            True이면 조기 종료
        """
        if self.best_score is None:
            self.best_score = score
        elif self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class ModelTrainer:
    """모델 학습 클래스"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        config: Dict[str, Any]
    ):
        """
        Args:
            model: 학습할 모델
            train_loader: 학습 데이터 로더
            val_loader: 검증 데이터 로더
            device: 학습 디바이스 (CPU/GPU)
            config: 학습 설정 딕셔너리
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config

        # 손실 함수
        self.criterion = self._create_loss_function()

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Learning Rate Scheduler
        self.scheduler = self._create_scheduler()

        # Early Stopping
        self.early_stopping = EarlyStopping(
            patience=config.get('early_stopping', {}).get('patience', 10),
            min_delta=config.get('early_stopping', {}).get('min_delta', 0.001)
        )

        # TensorBoard Writer
        log_dir = config.get('log_dir', 'runs')
        self.writer = SummaryWriter(log_dir=log_dir)

        # 학습 히스토리
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }

        # 최고 모델 저장 경로
        self.best_model_path = Path(config.get('save_dir', 'checkpoints')) / 'best_model.pth'
        self.best_model_path.parent.mkdir(parents=True, exist_ok=True)

    def _create_loss_function(self) -> nn.Module:
        """손실 함수 생성"""
        loss_type = self.config.get('loss', 'mse')

        if loss_type == 'mse':
            return nn.MSELoss()
        elif loss_type == 'l1':
            return nn.L1Loss()
        elif loss_type == 'smooth_l1':
            return nn.SmoothL1Loss()
        elif loss_type == 'nll':
            # Negative Log Likelihood (확률적 예측용)
            return nn.NLLLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def _create_optimizer(self) -> optim.Optimizer:
        """Optimizer 생성"""
        optimizer_type = self.config.get('optimizer', 'adam').lower()
        lr = self.config.get('learning_rate', 1e-3)
        weight_decay = self.config.get('weight_decay', 0.0)

        if optimizer_type == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            momentum = self.config.get('momentum', 0.9)
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Learning Rate Scheduler 생성"""
        scheduler_type = self.config.get('scheduler', 'reduce_on_plateau')

        if scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )
        elif scheduler_type == 'step':
            step_size = self.config.get('scheduler_step_size', 30)
            gamma = self.config.get('scheduler_gamma', 0.1)
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
        elif scheduler_type == 'cosine':
            T_max = self.config.get('num_epochs', 100)
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=T_max
            )
        else:
            return None

    def train_epoch(self) -> float:
        """한 에폭 학습 (GPU 메모리 최적화 포함)"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, batch in enumerate(pbar):
            # GPU 메모리 정리 (주기적으로)
            if batch_idx % 100 == 0 and self.device.type == 'cuda':
                torch.cuda.empty_cache()

            # 데이터를 디바이스로 이동 (non_blocking으로 비동기 전송)
            obs_data = batch['obs_data'].to(self.device, non_blocking=True)
            pred_target = batch['pred_data'].to(self.device, non_blocking=True)

            # Forward pass
            self.optimizer.zero_grad()

            # 그래프 데이터 처리
            if 'graph' in batch:
                graph = batch['graph'].to(self.device, non_blocking=True)
                pred = self.model(graph.x, graph.edge_index, graph.edge_weight)
            else:
                # 시퀀스 데이터 처리 (A3TGCN 등)
                pred = self.model(obs_data)

            # 손실 계산
            # pred: [batch_size, pred_steps, 2]
            # pred_target: [batch_size, pred_steps, 2]
            loss = self.criterion(pred, pred_target)

            # Backward pass
            loss.backward()

            # Gradient clipping (선택사항)
            max_grad_norm = self.config.get('max_grad_norm', None)
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_grad_norm
                )

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # 진행 상황 업데이트 (GPU 메모리 사용량 포함)
            postfix = {'loss': f'{loss.item():.4f}'}
            if self.device.type == 'cuda':
                memory_mb = torch.cuda.memory_allocated(self.device) / 1e6
                postfix['gpu_mem'] = f'{memory_mb:.0f}MB'
            pbar.set_postfix(postfix)

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self) -> float:
        """검증"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                obs_data = batch['obs_data'].to(self.device)
                pred_target = batch['pred_data'].to(self.device)

                if 'graph' in batch:
                    graph = batch['graph'].to(self.device)
                    pred = self.model(graph.x, graph.edge_index, graph.edge_weight)
                else:
                    pred = self.model(obs_data)

                loss = self.criterion(pred, pred_target)
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def train(self, num_epochs: int):
        """전체 학습 프로세스"""
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)

            # 학습
            train_loss = self.train_epoch()

            # 검증
            val_loss = self.validate()

            # Learning Rate Scheduler 업데이트
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # 현재 Learning Rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # 히스토리 저장
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)

            # TensorBoard 로깅
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)

            # GPU 메모리 사용량 로깅
            if self.device.type == 'cuda':
                memory_allocated = torch.cuda.memory_allocated(self.device) / 1e9
                memory_reserved = torch.cuda.memory_reserved(self.device) / 1e9
                self.writer.add_scalar('GPU/Memory_Allocated_GB', memory_allocated, epoch)
                self.writer.add_scalar('GPU/Memory_Reserved_GB', memory_reserved, epoch)

            # Gradient norm 로깅 (주기적으로)
            if epoch % 10 == 0:
                total_norm = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                self.writer.add_scalar('Training/Gradient_Norm', total_norm, epoch)

            # 결과 출력
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")

            # GPU 메모리 정보 출력
            if self.device.type == 'cuda':
                memory_allocated = torch.cuda.memory_allocated(self.device) / 1e9
                memory_reserved = torch.cuda.memory_reserved(self.device) / 1e9
                print(f"GPU Memory: {memory_allocated:.2f}GB / {memory_reserved:.2f}GB")

            # 최고 모델 저장
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss, is_best=True)
                print(f"✓ 최고 모델 저장 (Val Loss: {val_loss:.4f})")

            # Early Stopping 체크
            if self.early_stopping(val_loss):
                print(f"\nEarly Stopping triggered after {epoch + 1} epochs")
                break

        self.writer.close()
        print(f"\n학습 완료! 최고 검증 손실: {best_val_loss:.4f}")

    def _save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        is_best: bool = False
    ):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
            'history': self.history
        }

        if is_best:
            torch.save(checkpoint, self.best_model_path)

        # 주기적 체크포인트 저장
        if (epoch + 1) % self.config.get('save_every', 10) == 0:
            checkpoint_path = self.best_model_path.parent / f'checkpoint_epoch_{epoch + 1}.pth'
            torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: Path):
        """체크포인트 로드"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        return checkpoint['epoch']


def create_trainer(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: Dict[str, Any]
) -> ModelTrainer:
    """Trainer 생성 헬퍼 함수"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")

    return ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config
    )


def main():
    """테스트용 메인 함수"""
    # 설정 예시
    config = {
        'optimizer': 'adam',
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'scheduler': 'reduce_on_plateau',
        'loss': 'mse',
        'num_epochs': 100,
        'early_stopping': {
            'patience': 10,
            'min_delta': 0.001
        },
        'log_dir': 'runs/test',
        'save_dir': 'checkpoints',
        'max_grad_norm': 1.0
    }

    print("학습 설정:")
    print(json.dumps(config, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

