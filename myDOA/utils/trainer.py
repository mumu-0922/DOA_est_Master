"""
训练器：封装训练和验证流程
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
import json
from typing import Optional, Dict
from pathlib import Path

from .metrics import MetricTracker
from .loss_function import CombinedLoss


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


class Trainer:
    """
    DOA模型训练器

    Args:
        model: DOA模型
        device: 训练设备
        save_dir: 保存目录
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        save_dir: str = './results'
    ):
        self.model = model.to(device)
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.loss_fn = None
        self.optimizer = None
        self.scheduler = None
        self.history = {'train_loss': [], 'val_loss': [], 'val_rmse': []}

    def compile(
        self,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        lambda_spectrum: float = 1.0,
        lambda_sparse: float = 0.1,
        lambda_sep: float = 0.1,
        spectrum_loss_type: str = 'mse',
        scheduler_type: str = 'cosine',
        epochs: int = 100
    ):
        """
        配置优化器和损失函数

        Args:
            lr: 学习率（SI-SDR 损失建议使用 5e-4，MSE 损失使用 1e-4）
            weight_decay: 权重衰减
            lambda_spectrum: 空间谱损失权重
            lambda_sparse: 稀疏损失权重
            lambda_sep: 分离损失权重
            spectrum_loss_type: 空间谱损失类型 ('mse', 'bce', 'kl', 'si_sdr')
            scheduler_type: 学习率调度器类型 ('cosine', 'plateau', None)
            epochs: 训练轮数（用于 cosine 调度器）
        """

        self.loss_fn = CombinedLoss(
            lambda_spectrum=lambda_spectrum,
            lambda_sparse=lambda_sparse,
            lambda_sep=lambda_sep,
            spectrum_loss_type=spectrum_loss_type
        )

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        if scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        elif scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=5)
        else:
            self.scheduler = None

    def train_epoch(self, train_loader: DataLoader, k: int = 3) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        metric_tracker = MetricTracker()

        pbar = tqdm(train_loader, desc='Training')
        for batch in pbar:
            inputs = batch['input'].to(self.device)
            target_spectrum = batch['spectrum'].to(self.device)
            target_doa = batch['doa'].to(self.device)

            # 前向传播
            pred_spectrum = self.model(inputs)
            success, pred_doa = self.model.spectrum_to_doa(pred_spectrum, k)

            # 计算损失
            losses = self.loss_fn(pred_spectrum, target_spectrum, pred_doa, target_doa)

            # 反向传播
            self.optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += losses['total'].item()
            metric_tracker.update(pred_doa, target_doa, success)

            pbar.set_postfix({'loss': f"{losses['total'].item():.4f}"})

        metrics = metric_tracker.compute()
        metrics['loss'] = total_loss / len(train_loader)

        return metrics

    @torch.no_grad()
    def validate(self, val_loader: DataLoader, k: int = 3) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        metric_tracker = MetricTracker()

        for batch in val_loader:
            inputs = batch['input'].to(self.device)
            target_spectrum = batch['spectrum'].to(self.device)
            target_doa = batch['doa'].to(self.device)

            pred_spectrum = self.model(inputs)
            success, pred_doa = self.model.spectrum_to_doa(pred_spectrum, k)

            losses = self.loss_fn(pred_spectrum, target_spectrum, pred_doa, target_doa)
            total_loss += losses['total'].item()

            metric_tracker.update(pred_doa, target_doa, success)

        metrics = metric_tracker.compute()
        metrics['loss'] = total_loss / len(val_loader)

        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        k: int = 3,
        early_stopping_patience: int = 15,
        save_best: bool = True
    ) -> Dict:
        """完整训练流程"""

        early_stopping = EarlyStopping(patience=early_stopping_patience)
        best_val_loss = float('inf')

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 40)

            # 训练
            train_metrics = self.train_epoch(train_loader, k)
            self.history['train_loss'].append(train_metrics['loss'])
            print(f"Train - Loss: {train_metrics['loss']:.4f}, RMSE: {train_metrics['rmse']:.4f}")

            # 验证
            val_metrics = self.validate(val_loader, k)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_rmse'].append(val_metrics['rmse'])
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, RMSE: {val_metrics['rmse']:.4f}, "
                  f"Success: {val_metrics['success_rate']:.2%}")

            # 学习率调度
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()

            # 保存最佳模型
            if save_best and val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_checkpoint('best_model.pth')
                print("  -> Saved best model")

            # 早停
            if early_stopping(val_metrics['loss']):
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        # 保存最终模型和历史
        self.save_checkpoint('final_model.pth')
        self.save_history()

        return self.history

    def save_checkpoint(self, filename: str):
        """保存检查点"""
        path = self.save_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, filename: str):
        """加载检查点"""
        path = self.save_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def save_history(self):
        """保存训练历史"""
        path = self.save_dir / 'history.json'
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
