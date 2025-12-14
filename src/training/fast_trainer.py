"""
빠른 학습을 위한 최적화된 Trainer
Mixed Precision Training (FP16), 데이터 샘플링, 모델 경량화 포함
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Dict, Any
from tqdm import tqdm
import numpy as np

from src.training.trainer import ModelTrainer, EarlyStopping


class FastModelTrainer(ModelTrainer):
    """최적화된 빠른 학습 Trainer"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        config: Dict[str, Any],
        use_amp: bool = True,  # Mixed Precision Training
        sample_ratio: float = 1.0  # 데이터 샘플링 비율 (1.0 = 전체 사용)
    ):
        """
        Args:
            use_amp: Mixed Precision Training 사용 (FP16, 약 2배 속도 향상)
            sample_ratio: 학습 데이터 샘플링 비율 (0.1 = 10%만 사용)
        """
        super().__init__(model, train_loader, val_loader, device, config)

        self.use_amp = use_amp and device.type == 'cuda'
        self.sample_ratio = sample_ratio

        # Mixed Precision Training 설정
        if self.use_amp:
            self.scaler = GradScaler()
            print("✓ Mixed Precision Training (FP16) 활성화 - 약 2배 속도 향상")

        # 데이터 샘플링
        if sample_ratio < 1.0:
            self._sample_data()
            print(f"✓ 데이터 샘플링: {sample_ratio*100:.1f}% 사용")

    def _sample_data(self):
        """학습 데이터 샘플링"""
        if hasattr(self.train_loader, 'dataset'):
            dataset = self.train_loader.dataset
            total_size = len(dataset)
            sample_size = int(total_size * self.sample_ratio)

            # 랜덤 샘플링
            indices = np.random.choice(total_size, sample_size, replace=False)
            sampler = torch.utils.data.SubsetRandomSampler(indices)

            # 새로운 DataLoader 생성
            self.train_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.train_loader.batch_size,
                sampler=sampler,
                num_workers=self.train_loader.num_workers,
                pin_memory=self.train_loader.pin_memory
            )

    def train_epoch(self) -> float:
        """최적화된 한 에폭 학습"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc='Training (Fast)')

        for batch_idx, batch in enumerate(pbar):
            # GPU 메모리 정리 (더 자주)
            if batch_idx % 50 == 0 and self.device.type == 'cuda':
                torch.cuda.empty_cache()

            # 데이터를 디바이스로 이동
            obs_data = batch['obs_data'].to(self.device, non_blocking=True)
            pred_target = batch['pred_data'].to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # Mixed Precision Training
            if self.use_amp:
                with autocast():
                    if 'graph' in batch:
                        graph = batch['graph'].to(self.device, non_blocking=True)
                        pred = self.model(graph.x, graph.edge_index, graph.edge_weight)
                    else:
                        pred = self.model(obs_data)

                    loss = self.criterion(pred, pred_target)

                # Scaled backward pass
                self.scaler.scale(loss).backward()

                # Gradient clipping
                max_grad_norm = self.config.get('max_grad_norm', None)
                if max_grad_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_grad_norm
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 일반 학습
                if 'graph' in batch:
                    graph = batch['graph'].to(self.device, non_blocking=True)
                    pred = self.model(graph.x, graph.edge_index, graph.edge_weight)
                else:
                    pred = self.model(obs_data)

                loss = self.criterion(pred, pred_target)
                loss.backward()

                max_grad_norm = self.config.get('max_grad_norm', None)
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_grad_norm
                    )

                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # 진행 상황 업데이트
            postfix = {
                'loss': f'{loss.item():.4f}',
                'avg': f'{total_loss/num_batches:.4f}'
            }
            if self.device.type == 'cuda':
                postfix['gpu_mem'] = f'{torch.cuda.memory_allocated()/1e9:.2f}GB'
            pbar.set_postfix(postfix)

        return total_loss / num_batches if num_batches > 0 else 0.0

    def validate(self) -> float:
        """최적화된 검증"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                obs_data = batch['obs_data'].to(self.device, non_blocking=True)
                pred_target = batch['pred_data'].to(self.device, non_blocking=True)

                if self.use_amp:
                    with autocast():
                        if 'graph' in batch:
                            graph = batch['graph'].to(self.device, non_blocking=True)
                            pred = self.model(graph.x, graph.edge_index, graph.edge_weight)
                        else:
                            pred = self.model(obs_data)
                        loss = self.criterion(pred, pred_target)
                else:
                    if 'graph' in batch:
                        graph = batch['graph'].to(self.device, non_blocking=True)
                        pred = self.model(graph.x, graph.edge_index, graph.edge_weight)
                    else:
                        pred = self.model(obs_data)
                    loss = self.criterion(pred, pred_target)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0


def create_fast_trainer(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: Dict[str, Any],
    use_amp: bool = True,
    sample_ratio: float = 1.0
) -> FastModelTrainer:
    """빠른 Trainer 생성"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return FastModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
        use_amp=use_amp,
        sample_ratio=sample_ratio
    )

