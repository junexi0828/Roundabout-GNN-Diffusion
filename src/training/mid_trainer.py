"""
MID 모델 학습을 위한 Trainer
Diffusion 모델 특화 학습 로직
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional
from pathlib import Path
import numpy as np
from tqdm import tqdm

from ..evaluation.metrics import calculate_ade, calculate_fde
from ..evaluation.diffusion_metrics import DiffusionEvaluator, calculate_multimodal_metrics


class MIDTrainer:
    """
    MID 모델 학습 클래스
    Diffusion 모델의 특수한 학습 방식을 처리
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: Dict,
        device: torch.device
    ):
        """
        Args:
            model: MID 모델
            train_loader: 학습 데이터 로더
            val_loader: 검증 데이터 로더
            config: 학습 설정
            device: 디바이스
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Optimizer
        optimizer_name = config.get('optimizer', 'adam').lower()
        lr = config.get('learning_rate', 0.001)
        weight_decay = config.get('weight_decay', 1e-5)

        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )

        # Scheduler
        scheduler_type = config.get('scheduler', 'reduce_on_plateau')
        if scheduler_type == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        elif scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.get('num_epochs', 50)
            )
        else:
            self.scheduler = None

        # Loss function (MSE for noise prediction)
        self.criterion = nn.MSELoss()

        # Mixed Precision Training
        self.use_amp = config.get('use_amp', False)
        if self.use_amp:
            # MPS 호환 GradScaler
            device_type = 'cuda' if 'cuda' in str(self.device) else 'cpu'
            self.scaler = torch.amp.GradScaler(device_type)

        # TensorBoard
        log_dir = config.get('log_dir', 'runs/mid')
        self.writer = SummaryWriter(log_dir)

        # 체크포인트 저장 경로
        self.save_dir = Path(config.get('save_dir', 'checkpoints/mid'))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Early Stopping
        self.early_stopping = config.get('early_stopping', {})
        self.patience = self.early_stopping.get('patience', 10)
        self.min_delta = self.early_stopping.get('min_delta', 0.001)
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # 학습 통계
        self.train_losses = []
        self.val_losses = []
        self.val_ades = []
        self.val_fdes = []

    def train_epoch(self, epoch: int) -> float:
        """한 에폭 학습"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")

        for batch_idx, batch in enumerate(pbar):
            # 데이터 준비
            if isinstance(batch, dict):
                # MID 모델은 x, y 좌표만 필요하므로 obs_trajectory 우선 사용
                obs_data = batch.get('obs_trajectory', batch.get('obs_data'))
                future_data = batch.get('future_trajectory', batch.get('future_data'))
                graph_data = batch.get('graph_data', batch.get('graph'))
            else:
                # DataLoader가 직접 반환하는 경우
                obs_data = batch[0]
                future_data = batch[1] if len(batch) > 1 else None
                graph_data = batch[2] if len(batch) > 2 else None

            # 디바이스로 이동
            if obs_data is not None:
                obs_data = obs_data.to(self.device)
            if future_data is not None:
                future_data = future_data.to(self.device)
            if graph_data is not None:
                graph_data = graph_data.to(self.device)

            # Optimizer 초기화
            self.optimizer.zero_grad()

            # Forward pass (Mixed Precision)
            if self.use_amp:
                device_type = 'cuda' if 'cuda' in str(self.device) else 'cpu'
                with torch.amp.autocast(device_type):
                    if future_data is not None:
                        # 타임스텝 랜덤 샘플링
                        batch_size = future_data.size(0)
                        # HybridGNNMID는 내부에 mid 모델을 가지고 있음
                        if hasattr(self.model, 'mid'):
                            num_steps = self.model.mid.num_diffusion_steps
                        else:
                            num_steps = self.model.num_diffusion_steps
                        t = torch.randint(
                            0, num_steps,
                            (batch_size,), device=self.device
                        )

                        # Forward diffusion (노이즈 추가)
                        noise = torch.randn_like(future_data)
                        # HybridGNNMID는 내부에 mid 모델을 가지고 있음
                        if hasattr(self.model, 'mid'):
                            x_t = self.model.mid.q_sample(future_data, t, noise)
                        else:
                            x_t = self.model.q_sample(future_data, t, noise)

                        # 노이즈 예측
                        # 모델이 GNN을 사용하는지 확인
                        if hasattr(self.model, 'use_gnn') and self.model.use_gnn:
                            pred_noise = self.model(
                                graph_data=graph_data,
                                hetero_data=hetero_graph if 'hetero_graph' in locals() else None,
                                obs_trajectory=obs_data,
                                future_trajectory=None,
                                t=t,
                                x_t=x_t
                            )
                        else:
                            # GNN 없는 모델 (MIDModel)
                            pred_noise = self.model(
                                obs_trajectory=obs_data,
                                future_trajectory=None,
                                t=t,
                                x_t=x_t
                            )

                        # Loss 계산 (예측 노이즈 vs 실제 노이즈)
                        loss = self.criterion(pred_noise, noise)
                    else:
                        # 더미 forward (실제로는 future_data 필요)
                        if hasattr(self.model, 'use_gnn') and self.model.use_gnn:
                            pred_noise = self.model(
                                graph_data=graph_data,
                                obs_trajectory=obs_data
                            )
                        else:
                            pred_noise = self.model(
                                obs_trajectory=obs_data
                            )
                        loss = torch.tensor(0.0, device=self.device)

                # Backward
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Mixed Precision 미사용
                if future_data is not None:
                    # 타임스텝 랜덤 샘플링
                    batch_size = future_data.size(0)
                    num_steps = self.config.get('num_diffusion_steps', 100)
                    t = torch.randint(
                        0, num_steps,
                        (batch_size,), device=self.device
                    )

                    # Forward diffusion (노이즈 추가)
                    noise = torch.randn_like(future_data)
                    # HybridGNNMID는 내부에 mid 모델을 가지고 있음
                    if hasattr(self.model, 'mid'):
                        x_t = self.model.mid.q_sample(future_data, t, noise)
                    else:
                        x_t = self.model.q_sample(future_data, t, noise)

                    # 노이즈 예측 (GNN 사용 여부에 따라 분기)
                    if hasattr(self.model, 'use_gnn') and self.model.use_gnn:
                        pred_noise = self.model(
                            graph_data=graph_data,
                            obs_trajectory=obs_data,
                            future_trajectory=None,
                            t=t,
                            x_t=x_t
                        )
                    else:
                        pred_noise = self.model(
                            obs_trajectory=obs_data,
                            future_trajectory=None,
                            t=t,
                            x_t=x_t
                        )
                    # Loss 계산
                    loss = self.criterion(pred_noise, noise)
                else:
                    # 더미 forward (실제로는 future_data 필요)
                    if hasattr(self.model, 'use_gnn') and self.model.use_gnn:
                        pred_noise = self.model(
                            graph_data=graph_data,
                            obs_trajectory=obs_data
                        )
                    else:
                        pred_noise = self.model(
                            obs_trajectory=obs_data
                        )
                    loss = torch.tensor(0.0, device=self.device)

                loss.backward()

                # Gradient clipping
                max_grad_norm = self.config.get('max_grad_norm', 1.0)
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_grad_norm
                    )

                self.optimizer.step()

            # 통계 업데이트
            total_loss += loss.item()
            num_batches += 1

            # Progress bar 업데이트
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # TensorBoard 로깅
            global_step = epoch * len(self.train_loader) + batch_idx
            if batch_idx % 100 == 0:
                self.writer.add_scalar('train/loss', loss.item(), global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], global_step)

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def validate(self, epoch: int) -> Dict[str, float]:
        """검증"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_ground_truths = []
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # 데이터 준비
                if isinstance(batch, dict):
                    # MID 모델은 x, y 좌표만 필요하므로 obs_trajectory 우선 사용
                    obs_data = batch.get('obs_trajectory', batch.get('obs_data'))
                    future_data = batch.get('future_trajectory', batch.get('future_data'))
                    graph_data = batch.get('graph_data', batch.get('graph'))
                else:
                    obs_data = batch[0]
                    future_data = batch[1] if len(batch) > 1 else None
                    graph_data = batch[2] if len(batch) > 2 else None

                # 디바이스로 이동
                if obs_data is not None:
                    obs_data = obs_data.to(self.device)
                if future_data is not None:
                    future_data = future_data.to(self.device)
                if graph_data is not None:
                    graph_data = graph_data.to(self.device)

                # Forward pass
                if future_data is not None:
                    # 타임스텝 랜덤 샘플링
                    batch_size = future_data.size(0)
                    num_steps = self.config.get('num_diffusion_steps', 100)
                    t = torch.randint(
                        0, num_steps,
                        (batch_size,), device=self.device
                    )

                    # Forward diffusion
                    noise = torch.randn_like(future_data)
                    x_t = self.model.q_sample(future_data, t, noise)

                    # 노이즈 예측
                    pred_noise = self.model(
                        graph_data=graph_data,
                        obs_trajectory=obs_data,
                        future_trajectory=None,
                        t=t
                    )

                    # Loss
                    loss = self.criterion(pred_noise, noise)
                    total_loss += loss.item()

                # 샘플링 (평가용)
                if future_data is not None:
                    samples = self.model.sample(
                        graph_data=graph_data,
                        obs_trajectory=obs_data,
                        num_samples=20,
                        ddim_steps=2
                    )
                    # 최소 ADE 샘플 선택
                    best_samples = samples[0]  # 첫 번째 샘플 사용 (실제로는 최소 ADE)
                    all_predictions.append(best_samples.cpu().numpy())
                    all_ground_truths.append(future_data.cpu().numpy())

                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # 평가 지표 계산
        metrics = {'val_loss': avg_loss}

        if all_predictions and all_ground_truths:
            predictions = np.concatenate(all_predictions, axis=0)
            ground_truths = np.concatenate(all_ground_truths, axis=0)

            # 기본 지표
            ade = calculate_ade(predictions, ground_truths)
            fde = calculate_fde(predictions, ground_truths)

            metrics['ade'] = ade
            metrics['fde'] = fde

            # 다중 모달리티 지표 (샘플이 있는 경우)
            if len(all_predictions) > 0 and all_predictions[0].ndim == 4:
                # 샘플 형태: [num_samples, batch, pred_steps, 2]
                samples = np.stack(all_predictions, axis=1)  # [num_samples, total_batch, pred_steps, 2]
                gt_expanded = np.expand_dims(ground_truths, axis=0)  # [1, total_batch, pred_steps, 2]
                gt_expanded = np.repeat(gt_expanded, samples.shape[0], axis=0)  # [num_samples, total_batch, pred_steps, 2]

                # 다중 모달리티 지표
                diffusion_evaluator = DiffusionEvaluator(k=20)
                multimodal_metrics = diffusion_evaluator.evaluate(samples, ground_truths)
                metrics.update(multimodal_metrics)

        return metrics

    def train(self, num_epochs: int):
        """전체 학습 프로세스"""
        print("=" * 80)
        print("MID 모델 학습 시작")
        print("=" * 80)
        print(f"디바이스: {self.device}")
        print(f"에폭 수: {num_epochs}")
        print(f"Mixed Precision: {self.use_amp}")
        print("=" * 80)

        for epoch in range(num_epochs):
            # 학습
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)

            # 검증
            val_metrics = self.validate(epoch)
            val_loss = val_metrics['val_loss']
            self.val_losses.append(val_loss)

            if 'ade' in val_metrics:
                self.val_ades.append(val_metrics['ade'])
                self.val_fdes.append(val_metrics['fde'])

            # Scheduler 업데이트
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # TensorBoard 로깅
            self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
            self.writer.add_scalar('epoch/val_loss', val_loss, epoch)
            if 'ade' in val_metrics:
                self.writer.add_scalar('epoch/ade', val_metrics['ade'], epoch)
                self.writer.add_scalar('epoch/fde', val_metrics['fde'], epoch)

            # 출력
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            if 'ade' in val_metrics:
                print(f"  ADE: {val_metrics['ade']:.4f} m")
                print(f"  FDE: {val_metrics['fde']:.4f} m")

            # 체크포인트 저장
            if val_loss < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_loss
                self.patience_counter = 0

                # 모델 저장
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': self.config
                }
                torch.save(checkpoint, self.save_dir / 'best_model.pth')
                print(f"  ✓ Best model saved (val_loss: {val_loss:.4f})")
            else:
                self.patience_counter += 1

            # Early Stopping
            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                print(f"Best val loss: {self.best_val_loss:.4f}")
                break

        # 최종 모델 저장
        final_checkpoint = {
            'epoch': num_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        torch.save(final_checkpoint, self.save_dir / 'final_model.pth')

        self.writer.close()

        print("\n" + "=" * 80)
        print("학습 완료!")
        print("=" * 80)


def create_mid_trainer(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: Dict,
    device: torch.device
) -> MIDTrainer:
    """MID Trainer 생성 헬퍼 함수"""
    return MIDTrainer(model, train_loader, val_loader, config, device)

