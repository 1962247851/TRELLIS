"""
Volume-conditioned mixin for training models with volume constraints.
"""

from typing import *
import torch
import torch.nn as nn
import numpy as np


class VolumeConditionedMixin:
    """
    体积约束训练的Mixin类
    
    Args:
        volume_embed_dim: 体积嵌入的维度
        volume_range: 体积的范围 (min_volume, max_volume)，单位为立方厘米
        volume_loss_weight: 体积损失的权重
        volume_condition_prob: 在训练期间使用体积条件的概率
    """
    def __init__(
        self, 
        *args, 
        volume_embed_dim: int = 128,
        volume_range: Tuple[float, float] = (1.0, 10000.0),
        volume_loss_weight: float = 0.1,
        volume_condition_prob: float = 0.8,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.volume_embed_dim = volume_embed_dim
        self.volume_range = volume_range
        self.volume_loss_weight = volume_loss_weight
        self.volume_condition_prob = volume_condition_prob
        
        # 体积嵌入网络
        self.volume_embedder = nn.Sequential(
            nn.Linear(1, volume_embed_dim // 2),
            nn.ReLU(),
            nn.Linear(volume_embed_dim // 2, volume_embed_dim),
            nn.ReLU(),
            nn.Linear(volume_embed_dim, volume_embed_dim)
        ).cuda()
        
        # 体积预测头（用于预测生成模型的体积）
        self.volume_predictor = nn.Sequential(
            nn.Linear(self.training_models['denoiser'].model_channels, volume_embed_dim),
            nn.ReLU(),
            nn.Linear(volume_embed_dim, volume_embed_dim // 2),
            nn.ReLU(),
            nn.Linear(volume_embed_dim // 2, 1),
            nn.Sigmoid()  # 输出0-1范围，后续会缩放到实际体积范围
        ).cuda()
        
    def normalize_volume(self, volume: torch.Tensor) -> torch.Tensor:
        """
        将体积值归一化到[0, 1]范围
        
        Args:
            volume: 原始体积值（立方厘米）
            
        Returns:
            归一化后的体积值
        """
        min_vol, max_vol = self.volume_range
        # 对数变换，因为体积值可能跨越很大范围
        log_volume = torch.log(torch.clamp(volume, min=min_vol, max=max_vol))
        log_min = torch.log(torch.tensor(min_vol, device=volume.device))
        log_max = torch.log(torch.tensor(max_vol, device=volume.device))
        
        normalized = (log_volume - log_min) / (log_max - log_min)
        return torch.clamp(normalized, 0.0, 1.0)
    
    def denormalize_volume(self, normalized_volume: torch.Tensor) -> torch.Tensor:
        """
        将归一化的体积值还原到实际范围
        
        Args:
            normalized_volume: 归一化的体积值[0, 1]
            
        Returns:
            实际体积值（立方厘米）
        """
        min_vol, max_vol = self.volume_range
        log_min = torch.log(torch.tensor(min_vol, device=normalized_volume.device))
        log_max = torch.log(torch.tensor(max_vol, device=normalized_volume.device))
        
        log_volume = log_min + normalized_volume * (log_max - log_min)
        volume = torch.exp(log_volume)
        return torch.clamp(volume, min_vol, max_vol)
    
    def encode_volume(self, volume: torch.Tensor) -> torch.Tensor:
        """
        编码体积信息为嵌入向量
        
        Args:
            volume: 体积值（立方厘米）
            
        Returns:
            体积嵌入向量
        """
        normalized_volume = self.normalize_volume(volume)
        # 添加batch维度如果需要
        if normalized_volume.dim() == 0:
            normalized_volume = normalized_volume.unsqueeze(0)
        if normalized_volume.dim() == 1:
            normalized_volume = normalized_volume.unsqueeze(-1)
            
        volume_embed = self.volume_embedder(normalized_volume.float())
        return volume_embed
        
    def get_volume_cond(self, volume: torch.Tensor, apply_dropout: bool = True) -> torch.Tensor:
        """
        获取体积条件嵌入
        
        Args:
            volume: 体积值张量
            apply_dropout: 是否应用条件dropout
            
        Returns:
            体积条件嵌入
        """
        batch_size = volume.shape[0]
        
        # 在训练期间随机dropout体积条件
        if apply_dropout and self.training and self.volume_condition_prob < 1.0:
            mask = torch.rand(batch_size, device=volume.device) < self.volume_condition_prob
            # 对于被dropout的样本，使用平均体积
            avg_volume = torch.tensor(
                np.exp((np.log(self.volume_range[0]) + np.log(self.volume_range[1])) / 2),
                device=volume.device
            )
            volume = torch.where(mask.unsqueeze(-1), volume, avg_volume)
        
        return self.encode_volume(volume)
    
    def compute_volume_loss(self, predicted_features: torch.Tensor, target_volume: torch.Tensor) -> torch.Tensor:
        """
        计算体积预测损失
        
        Args:
            predicted_features: 模型预测的特征
            target_volume: 目标体积值
            
        Returns:
            体积损失
        """
        # 从模型特征预测体积
        predicted_volume_norm = self.volume_predictor(predicted_features.mean(dim=1))  # 平均池化特征
        predicted_volume = self.denormalize_volume(predicted_volume_norm.squeeze(-1))
        
        # 计算MSE损失（在对数空间中，因为体积跨度很大）
        log_pred = torch.log(torch.clamp(predicted_volume, min=self.volume_range[0]))
        log_target = torch.log(torch.clamp(target_volume, min=self.volume_range[0]))
        
        volume_loss = nn.functional.mse_loss(log_pred, log_target)
        return volume_loss
    
    def get_cond(self, cond, volume=None, **kwargs):
        """
        获取包含体积条件的综合条件信息
        
        Args:
            cond: 原始条件（图像等）
            volume: 体积信息
            **kwargs: 其他参数
            
        Returns:
            包含体积条件的条件信息
        """
        # 获取原始条件（如图像条件）
        cond = super().get_cond(cond, **kwargs)
        
        if volume is not None:
            # 获取体积条件嵌入
            volume_cond = self.get_volume_cond(volume)
            
            # 将体积条件与原始条件结合
            # 这里假设cond是一个张量，我们将体积嵌入添加到序列维度
            if isinstance(cond, torch.Tensor):
                # 扩展体积嵌入的序列维度以匹配条件
                volume_cond = volume_cond.unsqueeze(1)  # [B, 1, D]
                cond = torch.cat([cond, volume_cond], dim=1)  # 在序列维度连接
            else:
                # 如果cond是字典或其他格式，添加体积信息
                if isinstance(cond, dict):
                    cond['volume'] = volume_cond
                else:
                    # 创建包含原始条件和体积条件的字典
                    cond = {'image_cond': cond, 'volume_cond': volume_cond}
        
        return cond
    
    def get_inference_cond(self, cond, volume=None, **kwargs):
        """
        获取推理时的条件信息
        """
        # 获取原始条件
        cond = super().get_inference_cond(cond, **kwargs)
        
        if volume is not None:
            # 推理时不应用dropout
            volume_cond = self.get_volume_cond(volume, apply_dropout=False)
            
            if isinstance(cond, torch.Tensor):
                volume_cond = volume_cond.unsqueeze(1)
                cond = torch.cat([cond, volume_cond], dim=1)
            else:
                if isinstance(cond, dict):
                    cond['volume'] = volume_cond
                else:
                    cond = {'image_cond': cond, 'volume_cond': volume_cond}
        
        return cond
