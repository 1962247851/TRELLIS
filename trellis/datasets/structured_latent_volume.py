"""
体积约束的结构化潜在数据集

这个数据集类结合了图像条件和体积信息，用于训练具有体积约束的3D生成模型。
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .components import StandardDatasetBase, ImageConditionedMixin
from .structured_latent import StructuredLatent


class VolumeConditionedMixin:
    """
    为数据集添加体积条件的Mixin类
    """
    def __init__(self, roots, **kwargs):
        super().__init__(roots, **kwargs)
        self.volumes = {}
        
        # 加载体积信息
        for instance in self.instances:
            sha256 = instance[1]
            if 'volume' in self.metadata.loc[sha256]:
                self.volumes[sha256] = float(self.metadata.loc[sha256]['volume'])
            else:
                # 如果没有体积信息，使用默认值
                self.volumes[sha256] = 100.0  # 默认100立方厘米
    
    def filter_metadata(self, metadata):
        """过滤元数据，只保留有体积信息的实例"""
        metadata, stats = super().filter_metadata(metadata)
        
        # 确保有体积信息
        if 'volume' in metadata.columns:
            metadata = metadata[metadata['volume'].notna()]
            metadata = metadata[metadata['volume'] > 0]  # 体积必须大于0
            stats['With volume info'] = len(metadata)
        else:
            # 如果没有体积列，添加默认体积
            metadata['volume'] = 100.0
            stats['Added default volume'] = len(metadata)
        
        return metadata, stats
    
    def get_instance(self, root, instance):
        """获取实例数据，包括体积信息"""
        pack = super().get_instance(root, instance)
        
        # 添加体积信息
        if instance in self.volumes:
            volume = self.volumes[instance]
        else:
            volume = 100.0  # 默认体积
            
        pack['volume'] = torch.tensor(volume, dtype=torch.float32)
        return pack


class ImageVolumeConditionedStructuredLatent(VolumeConditionedMixin, ImageConditionedMixin, StructuredLatent):
    """
    结合图像条件和体积约束的结构化潜在数据集
    
    这个数据集类用于训练能够根据图像生成具有指定体积的3D模型的系统。
    
    Args:
        roots: 数据集根目录路径
        latent_model: 潜在编码器模型名称
        image_size: 图像大小
        min_aesthetic_score: 最小美学评分
        max_num_voxels: 最大体素数量
        normalization: 归一化参数
        pretrained_slat_dec: 预训练的SLAT解码器
        volume_range: 体积范围 (min, max)，单位为立方厘米
        **kwargs: 其他参数
    """
    
    def __init__(
        self, 
        roots, 
        latent_model: str,
        image_size: int = 1024,
        min_aesthetic_score: float = 3.5,
        max_num_voxels: int = 32768,
        normalization: dict = None,
        pretrained_slat_dec: str = None,
        volume_range: tuple = (1.0, 10000.0),
        **kwargs
    ):
        self.volume_range = volume_range
        super().__init__(
            roots=roots,
            latent_model=latent_model,
            image_size=image_size,
            min_aesthetic_score=min_aesthetic_score,
            max_num_voxels=max_num_voxels,
            normalization=normalization,
            pretrained_slat_dec=pretrained_slat_dec,
            **kwargs
        )
        
    def filter_metadata(self, metadata):
        """过滤元数据，确保体积在合理范围内"""
        metadata, stats = super().filter_metadata(metadata)
        
        # 过滤体积范围
        if 'volume' in metadata.columns:
            min_vol, max_vol = self.volume_range
            original_count = len(metadata)
            metadata = metadata[
                (metadata['volume'] >= min_vol) & 
                (metadata['volume'] <= max_vol)
            ]
            stats[f'Volume in range [{min_vol}, {max_vol}]'] = len(metadata)
            if len(metadata) < original_count:
                print(f"Filtered out {original_count - len(metadata)} instances due to volume constraints")
        
        return metadata, stats
    
    def get_instance(self, root, instance):
        """获取实例数据"""
        pack = super().get_instance(root, instance)
        
        # 确保体积在合理范围内
        volume = pack['volume']
        min_vol, max_vol = self.volume_range
        pack['volume'] = torch.clamp(volume, min=min_vol, max=max_vol)
        
        return pack
        
    def __str__(self):
        """返回数据集描述"""
        lines = super().__str__().split('\n')
        lines.insert(-1, f"  Volume range: {self.volume_range} cm³")
        return '\n'.join(lines)


class MetaFood3DVolumeDataset(ImageVolumeConditionedStructuredLatent):
    """
    专门为MetaFood3D数据集设计的体积约束数据集类
    
    这个类专门处理MetaFood3D数据集，包含食物模型的图像和体积信息。
    """
    
    def __init__(self, roots, **kwargs):
        # 为食物模型设置合适的体积范围（1-5000立方厘米）
        kwargs.setdefault('volume_range', (1.0, 5000.0))
        super().__init__(roots, **kwargs)
    
    def filter_metadata(self, metadata):
        """为MetaFood3D数据集应用特定的过滤规则"""
        metadata, stats = super().filter_metadata(metadata)
        
        # 可以添加食物特定的过滤规则
        # 例如：只保留某些食物类别
        if 'category' in metadata.columns:
            # 过滤掉不合适的类别（如果有的话）
            exclude_categories = []  # 可以在这里指定要排除的类别
            if exclude_categories:
                metadata = metadata[~metadata['category'].isin(exclude_categories)]
                stats[f'Excluded categories: {exclude_categories}'] = len(metadata)
        
        return metadata, stats
    
    def get_instance(self, root, instance):
        """获取MetaFood3D实例数据"""
        pack = super().get_instance(root, instance)
        
        # 可以添加食物特定的数据处理
        # 例如：数据增强、特殊的归一化等
        
        return pack
