"""
TRELLIS Training Monitor System
全面的训练监控系统，支持多种指标追踪和可视化
"""

import os
import sys
import json
import time
import psutil
import GPUtil
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import seaborn as sns
from datetime import datetime, timedelta
from collections import deque, defaultdict
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import threading
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ComprehensiveMonitor:
    """
    全面的训练监控器
    支持：损失追踪、学习率监控、梯度分析、资源使用、生成质量评估等
    """

    def __init__(
            self,
            output_dir: str,
            experiment_name: Optional[str] = None,
            config: Optional[Dict] = None,
            enable_tensorboard: bool = True,
            enable_plots: bool = True,
            plot_interval: int = 100,
            save_interval: int = 1000,
            history_size: int = 10000,
            moving_average_window: int = 100
    ):
        self.output_dir = output_dir
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.config = config or {}
        self.enable_tensorboard = enable_tensorboard
        self.enable_plots = enable_plots
        self.plot_interval = plot_interval
        self.save_interval = save_interval
        self.history_size = history_size
        self.ma_window = moving_average_window

        # 创建目录
        self.log_dir = os.path.join(output_dir, 'logs', self.experiment_name)
        self.plot_dir = os.path.join(output_dir, 'plots', self.experiment_name)
        self.metric_dir = os.path.join(output_dir, 'metrics', self.experiment_name)

        for dir_path in [self.log_dir, self.plot_dir, self.metric_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # TensorBoard
        self.writer = SummaryWriter(self.log_dir) if enable_tensorboard else None

        # 历史数据存储
        self.history = defaultdict(lambda: deque(maxlen=history_size))
        self.ma_history = defaultdict(lambda: deque(maxlen=history_size))  # 移动平均

        # 统计信息
        self.stats = {
            'start_time': time.time(),
            'total_steps': 0,
            'total_samples': 0,
            'best_loss': float('inf'),
            'best_step': 0,
            'nan_count': 0,
            'gradient_explosion_count': 0,
            'oom_count': 0
        }

        # 实时数据
        self.current_metrics = {}
        self.alerts = deque(maxlen=50)  # 警告信息

        # 图表设置
        self.setup_plot_style()
        self.fig = None
        self.axes = None

        # 保存配置
        if config:
            self.save_config(config)

    def setup_plot_style(self):
        """设置图表风格"""
        plt.style.use('dark_background')
        sns.set_palette("husl")
        plt.rcParams.update({
            'figure.facecolor': '#1e1e1e',
            'axes.facecolor': '#2d2d2d',
            'axes.edgecolor': '#555555',
            'grid.color': '#404040',
            'text.color': '#ffffff',
            'axes.labelcolor': '#ffffff',
            'xtick.color': '#ffffff',
            'ytick.color': '#ffffff',
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'legend.fontsize': 9,
            'figure.titlesize': 14
        })

    def save_config(self, config: Dict):
        """保存训练配置"""
        config_path = os.path.join(self.metric_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """记录指标"""
        self.stats['total_steps'] = step

        # 更新当前指标
        self.current_metrics.update(metrics)
        self.current_metrics['step'] = step
        self.current_metrics['timestamp'] = time.time()

        # 记录到历史
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.number)):
                self.history[key].append(float(value))
            elif isinstance(value, torch.Tensor):
                self.history[key].append(float(value.detach().cpu().item()))

        self.history['step'].append(step)
        self.history['timestamp'].append(time.time())

        # 计算移动平均
        for key in metrics.keys():
            if key in self.history and len(self.history[key]) > 0:
                window = min(self.ma_window, len(self.history[key]))
                ma_value = np.mean(list(self.history[key])[-window:])
                self.ma_history[f"{key}_ma"].append(ma_value)

        # 检测异常
        self._check_anomalies(metrics, step)

        # TensorBoard记录
        if self.writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float, np.number, torch.Tensor)):
                    self.writer.add_scalar(key, value, step)

        # 更新最佳记录
        if 'loss' in metrics and metrics['loss'] < self.stats['best_loss']:
            self.stats['best_loss'] = metrics['loss']
            self.stats['best_step'] = step

        # 定期绘图
        if self.enable_plots and step % self.plot_interval == 0:
            self.plot_all_metrics()

        # 定期保存
        if step % self.save_interval == 0:
            self.save_metrics()

    def _check_anomalies(self, metrics: Dict, step: int):
        """检测训练异常"""
        # NaN检测
        for key, value in metrics.items():
            if isinstance(value, (float, torch.Tensor)):
                if torch.isnan(torch.tensor(value)):
                    self.stats['nan_count'] += 1
                    self.alerts.append({
                        'step': step,
                        'type': 'NaN',
                        'message': f'NaN detected in {key}'
                    })

        # 梯度爆炸检测
        if 'grad_norm' in metrics and metrics['grad_norm'] > 100:
            self.stats['gradient_explosion_count'] += 1
            self.alerts.append({
                'step': step,
                'type': 'Gradient Explosion',
                'message': f'Gradient norm = {metrics["grad_norm"]:.2f}'
            })

        # 学习率异常
        if 'lr' in metrics and metrics['lr'] < 1e-10:
            self.alerts.append({
                'step': step,
                'type': 'Learning Rate',
                'message': 'Learning rate too small'
            })

    def log_model_stats(self, model: nn.Module, step: int):
        """记录模型统计信息"""
        stats = self._compute_model_stats(model)

        # 记录参数统计
        self.log_metrics({
            'model/total_params': stats['total_params'],
            'model/trainable_params': stats['trainable_params'],
            'model/param_mean': stats['param_mean'],
            'model/param_std': stats['param_std'],
            'model/grad_mean': stats['grad_mean'],
            'model/grad_std': stats['grad_std'],
            'model/weight_norm': stats['weight_norm'],
            'model/grad_norm_total': stats['grad_norm_total']
        }, step)

        # 层级统计
        if self.writer:
            for layer_name, layer_stats in stats['layer_stats'].items():
                for stat_name, value in layer_stats.items():
                    self.writer.add_scalar(
                        f'layers/{layer_name}/{stat_name}',
                        value,
                        step
                    )

    def _compute_model_stats(self, model: nn.Module) -> Dict:
        """计算模型统计信息"""
        stats = {
            'total_params': 0,
            'trainable_params': 0,
            'param_mean': 0,
            'param_std': 0,
            'grad_mean': 0,
            'grad_std': 0,
            'weight_norm': 0,
            'grad_norm_total': 0,
            'layer_stats': {}
        }

        param_values = []
        grad_values = []

        for name, param in model.named_parameters():
            num_params = param.numel()
            stats['total_params'] += num_params

            if param.requires_grad:
                stats['trainable_params'] += num_params

                # 参数统计
                param_data = param.data.detach().cpu().numpy().flatten()
                param_values.extend(param_data)
                stats['weight_norm'] += np.linalg.norm(param_data) ** 2

                # 梯度统计
                if param.grad is not None:
                    grad_data = param.grad.detach().cpu().numpy().flatten()
                    grad_values.extend(grad_data)
                    stats['grad_norm_total'] += np.linalg.norm(grad_data) ** 2

                # 层级统计
                layer_name = name.split('.')[0]
                if layer_name not in stats['layer_stats']:
                    stats['layer_stats'][layer_name] = {
                        'param_mean': 0,
                        'param_std': 0,
                        'grad_mean': 0,
                        'grad_std': 0
                    }

                stats['layer_stats'][layer_name]['param_mean'] = np.mean(param_data)
                stats['layer_stats'][layer_name]['param_std'] = np.std(param_data)

                if param.grad is not None:
                    stats['layer_stats'][layer_name]['grad_mean'] = np.mean(grad_data)
                    stats['layer_stats'][layer_name]['grad_std'] = np.std(grad_data)

        # 全局统计
        if param_values:
            stats['param_mean'] = np.mean(param_values)
            stats['param_std'] = np.std(param_values)

        if grad_values:
            stats['grad_mean'] = np.mean(grad_values)
            stats['grad_std'] = np.std(grad_values)

        stats['weight_norm'] = np.sqrt(stats['weight_norm'])
        stats['grad_norm_total'] = np.sqrt(stats['grad_norm_total'])

        return stats

    def log_system_stats(self, step: int):
        """记录系统资源使用情况"""
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)

        # 内存使用
        memory = psutil.virtual_memory()

        # GPU使用
        gpu_stats = []
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_stats.append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'load': gpu.load * 100,
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal,
                    'memory_percent': gpu.memoryUtil * 100,
                    'temperature': gpu.temperature
                })
        except:
            # 如果GPUtil不可用，使用torch的方法
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_stats.append({
                        'id': i,
                        'memory_used': torch.cuda.memory_allocated(i) / 1024 ** 3,
                        'memory_reserved': torch.cuda.memory_reserved(i) / 1024 ** 3,
                        'memory_percent': (torch.cuda.memory_allocated(i) /
                                           torch.cuda.get_device_properties(i).total_memory * 100)
                    })

        # 记录指标
        metrics = {
            'system/cpu_percent': cpu_percent,
            'system/memory_percent': memory.percent,
            'system/memory_used_gb': memory.used / 1024 ** 3,
        }

        # GPU指标
        for i, gpu in enumerate(gpu_stats):
            prefix = f'system/gpu{i}'
            metrics[f'{prefix}/memory_used_gb'] = gpu.get('memory_used', 0)
            metrics[f'{prefix}/memory_percent'] = gpu.get('memory_percent', 0)
            metrics[f'{prefix}/utilization'] = gpu.get('load', 0)
            if 'temperature' in gpu:
                metrics[f'{prefix}/temperature'] = gpu['temperature']

        self.log_metrics(metrics, step)

    def log_training_speed(self, batch_size: int, step: int, time_elapsed: float):
        """记录训练速度"""
        samples_per_second = batch_size / time_elapsed

        # 计算ETA
        if 'max_steps' in self.config:
            steps_remaining = self.config['max_steps'] - step
            eta_seconds = steps_remaining * time_elapsed
            eta_str = str(timedelta(seconds=int(eta_seconds)))
        else:
            eta_str = "N/A"

        metrics = {
            'speed/samples_per_second': samples_per_second,
            'speed/seconds_per_step': time_elapsed,
            'speed/minutes_per_1k_steps': time_elapsed * 1000 / 60,
        }

        self.log_metrics(metrics, step)
        self.stats['total_samples'] += batch_size

        return eta_str

    def log_3d_metrics(self, metrics: Dict[str, float], step: int):
        """记录3D生成相关的特殊指标"""
        # 3D质量指标
        quality_metrics = {
            'quality/chamfer_distance': metrics.get('chamfer_distance', 0),
            'quality/point_cloud_density': metrics.get('point_cloud_density', 0),
            'quality/mesh_watertight_ratio': metrics.get('mesh_watertight_ratio', 0),
            'quality/surface_smoothness': metrics.get('surface_smoothness', 0),
            'quality/geometric_consistency': metrics.get('geometric_consistency', 0),
        }

        # 生成多样性
        diversity_metrics = {
            'diversity/shape_variance': metrics.get('shape_variance', 0),
            'diversity/texture_variance': metrics.get('texture_variance', 0),
            'diversity/latent_std': metrics.get('latent_std', 0),
        }

        # 条件一致性（用于条件生成）
        if 'condition_consistency' in metrics:
            consistency_metrics = {
                'consistency/image_condition': metrics.get('image_consistency', 0),
                'consistency/text_condition': metrics.get('text_consistency', 0),
                'consistency/view_consistency': metrics.get('view_consistency', 0),
            }
            self.log_metrics(consistency_metrics, step)

        self.log_metrics(quality_metrics, step)
        self.log_metrics(diversity_metrics, step)

    def plot_all_metrics(self):
        """绘制所有指标的综合图表"""
        if not self.enable_plots or len(self.history['step']) < 2:
            return

        # 创建大型综合图表
        if self.fig is None:
            self.fig = plt.figure(figsize=(24, 16))
            self.fig.suptitle(
                f'TRELLIS Training Monitor - {self.experiment_name}',
                fontsize=16,
                y=0.995
            )
        else:
            self.fig.clear()

        # 使用GridSpec创建复杂布局
        gs = gridspec.GridSpec(4, 4, figure=self.fig, hspace=0.3, wspace=0.3)

        # 1. 主要损失曲线（大图）
        ax1 = self.fig.add_subplot(gs[0, :2])
        self._plot_loss_curves(ax1)

        # 2. 学习率调度
        ax2 = self.fig.add_subplot(gs[0, 2])
        self._plot_learning_rate(ax2)

        # 3. 训练速度
        ax3 = self.fig.add_subplot(gs[0, 3])
        self._plot_training_speed(ax3)

        # 4. 梯度统计
        ax4 = self.fig.add_subplot(gs[1, :2])
        self._plot_gradient_stats(ax4)

        # 5. GPU使用情况
        ax5 = self.fig.add_subplot(gs[1, 2:])
        self._plot_gpu_usage(ax5)

        # 6. 模型参数分布
        ax6 = self.fig.add_subplot(gs[2, 0])
        self._plot_parameter_distribution(ax6)

        # 7. 3D质量指标
        ax7 = self.fig.add_subplot(gs[2, 1])
        self._plot_3d_quality_metrics(ax7)

        # 8. 系统资源
        ax8 = self.fig.add_subplot(gs[2, 2])
        self._plot_system_resources(ax8)

        # 9. 训练进度
        ax9 = self.fig.add_subplot(gs[2, 3])
        self._plot_training_progress(ax9)

        # 10. 损失分解（如果有多个损失项）
        ax10 = self.fig.add_subplot(gs[3, :2])
        self._plot_loss_breakdown(ax10)

        # 11. 警告和异常
        ax11 = self.fig.add_subplot(gs[3, 2:])
        self._plot_alerts_summary(ax11)

        # 保存图表
        plt.tight_layout()
        plot_path = os.path.join(self.plot_dir, 'training_monitor.png')
        self.fig.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='#1e1e1e')

        # 生成独立的详细图表
        self._generate_detailed_plots()

    def _plot_loss_curves(self, ax):
        """绘制损失曲线"""
        steps = list(self.history['step'])

        # 绘制所有损失相关的指标
        loss_keys = [k for k in self.history.keys() if 'loss' in k.lower() and k != 'step']

        for key in loss_keys:
            values = list(self.history[key])
            if len(values) > 0:
                # 原始值
                ax.plot(steps, values, alpha=0.3, linewidth=1)

                # 移动平均
                if f"{key}_ma" in self.ma_history:
                    ma_values = list(self.ma_history[f"{key}_ma"])
                    ax.plot(steps[-len(ma_values):], ma_values,
                            label=key, linewidth=2)

        ax.set_xlabel('Steps')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Curves')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # 标记最佳点
        if self.stats['best_step'] > 0:
            ax.axvline(x=self.stats['best_step'], color='red',
                       linestyle='--', alpha=0.5, label='Best')

    def _plot_learning_rate(self, ax):
        """绘制学习率"""
        if 'lr' not in self.history:
            return

        steps = list(self.history['step'])
        lr_values = list(self.history['lr'])

        ax.plot(steps, lr_values, color='green', linewidth=2)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # 显示当前学习率
        if lr_values:
            current_lr = lr_values[-1]
            ax.text(0.95, 0.95, f'Current: {current_lr:.2e}',
                    transform=ax.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))

    def _plot_training_speed(self, ax):
        """绘制训练速度"""
        if 'speed/samples_per_second' not in self.history:
            return

        steps = list(self.history['step'])
        speed = list(self.history['speed/samples_per_second'])

        ax.plot(steps, speed, color='cyan', linewidth=2)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Samples/Second')
        ax.set_title('Training Speed')
        ax.grid(True, alpha=0.3)

        # 显示平均速度
        if speed:
            avg_speed = np.mean(speed[-100:])  # 最近100步的平均
            ax.axhline(y=avg_speed, color='red', linestyle='--', alpha=0.5)
            ax.text(0.95, 0.05, f'Avg: {avg_speed:.1f}',
                    transform=ax.transAxes, ha='right', va='bottom',
                    bbox=dict(boxstyle='round', facecolor='cyan', alpha=0.3))

    def _plot_gradient_stats(self, ax):
        """绘制梯度统计"""
        steps = list(self.history['step'])

        # 梯度范数
        if 'grad_norm' in self.history:
            grad_norm = list(self.history['grad_norm'])
            ax.plot(steps, grad_norm, label='Gradient Norm', color='yellow')

        # 梯度均值和标准差
        if 'model/grad_mean' in self.history:
            grad_mean = list(self.history['model/grad_mean'])
            grad_std = list(self.history['model/grad_std'])

            ax2 = ax.twinx()
            ax2.plot(steps, grad_mean, label='Grad Mean', color='orange', alpha=0.7)
            ax2.plot(steps, grad_std, label='Grad Std', color='red', alpha=0.7)
            ax2.set_ylabel('Gradient Statistics')
            ax2.legend(loc='upper right')

        ax.set_xlabel('Steps')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Analysis')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # 标记梯度爆炸
        if self.stats['gradient_explosion_count'] > 0:
            ax.text(0.02, 0.98, f'Explosions: {self.stats["gradient_explosion_count"]}',
                    transform=ax.transAxes, ha='left', va='top',
                    bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))

    def _plot_gpu_usage(self, ax):
        """绘制GPU使用情况"""
        steps = list(self.history['step'])

        # GPU内存使用
        gpu_keys = [k for k in self.history.keys() if 'gpu' in k and 'memory' in k]

        for key in gpu_keys:
            values = list(self.history[key])
            if len(values) > 0:
                gpu_id = key.split('gpu')[1].split('/')[0]
                ax.plot(steps, values, label=f'GPU {gpu_id}', linewidth=2)

        ax.set_xlabel('Steps')
        ax.set_ylabel('Memory (GB)')
        ax.set_title('GPU Memory Usage')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 添加利用率（如果有）
        if any('utilization' in k for k in self.history.keys()):
            ax2 = ax.twinx()
            util_keys = [k for k in self.history.keys() if 'utilization' in k]
            for key in util_keys:
                values = list(self.history[key])
                if len(values) > 0:
                    ax2.plot(steps, values, '--', alpha=0.5)
            ax2.set_ylabel('Utilization (%)')

    def _plot_parameter_distribution(self, ax):
        """绘制参数分布"""
        if 'model/param_mean' not in self.history:
            ax.text(0.5, 0.5, 'No parameter data',
                    transform=ax.transAxes, ha='center', va='center')
            return

        # 获取最新的参数统计
        param_mean = self.history['model/param_mean'][-1] if self.history['model/param_mean'] else 0
        param_std = self.history['model/param_std'][-1] if self.history['model/param_std'] else 1

        # 生成正态分布
        x = np.linspace(param_mean - 3 * param_std, param_mean + 3 * param_std, 100)
        y = np.exp(-0.5 * ((x - param_mean) / param_std) ** 2) / (param_std * np.sqrt(2 * np.pi))

        ax.fill_between(x, y, alpha=0.5, color='blue')
        ax.axvline(x=param_mean, color='red', linestyle='--', label=f'μ={param_mean:.3f}')
        ax.axvline(x=0, color='green', linestyle='--', alpha=0.5, label='Zero')

        ax.set_xlabel('Parameter Value')
        ax.set_ylabel('Density')
        ax.set_title('Parameter Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_3d_quality_metrics(self, ax):
        """绘制3D质量指标"""
        quality_keys = [k for k in self.history.keys() if 'quality/' in k]

        if not quality_keys:
            ax.text(0.5, 0.5, 'No 3D quality metrics',
                    transform=ax.transAxes, ha='center', va='center')
            return

        # 使用雷达图
        angles = np.linspace(0, 2 * np.pi, len(quality_keys), endpoint=False).tolist()

        # 获取最新值
        values = []
        labels = []
        for key in quality_keys:
            if self.history[key]:
                values.append(self.history[key][-1])
                labels.append(key.split('/')[-1])

        if values:
            # 归一化到0-1
            values = np.array(values)
            if values.max() > 0:
                values = values / values.max()

            # 闭合图形
            angles += angles[:1]
            values = np.concatenate((values, [values[0]]))

            # 绘制
            ax.plot(angles, values, 'o-', linewidth=2, color='lime')
            ax.fill(angles, values, alpha=0.25, color='lime')
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels, size=8)
            ax.set_ylim(0, 1)
            ax.set_title('3D Quality Metrics')
            ax.grid(True, alpha=0.3)

    def _plot_system_resources(self, ax):
        """绘制系统资源使用"""
        if 'system/cpu_percent' not in self.history:
            return

        # 获取最新数据
        cpu = self.history['system/cpu_percent'][-1] if self.history['system/cpu_percent'] else 0
        mem = self.history['system/memory_percent'][-1] if self.history['system/memory_percent'] else 0

        # 绘制仪表盘样式
        categories = ['CPU', 'Memory']
        values = [cpu, mem]
        colors = ['#FF6B6B', '#4ECDC4']

        bars = ax.barh(categories, values, color=colors)
        ax.set_xlim(0, 100)
        ax.set_xlabel('Usage (%)')
        ax.set_title('System Resources')

        # 添加数值标签
        for bar, value in zip(bars, values):
            ax.text(value + 1, bar.get_y() + bar.get_height() / 2,
                    f'{value:.1f}%', va='center')

    def _plot_training_progress(self, ax):
        """绘制训练进度"""
        current_step = self.stats['total_steps']

        if 'max_steps' in self.config:
            max_steps = self.config['max_steps']
            progress = current_step / max_steps * 100

            # 进度条
            ax.barh(['Progress'], [progress], color='#00D9FF', height=0.5)
            ax.set_xlim(0, 100)
            ax.set_xlabel('Completion (%)')
            ax.set_title('Training Progress')

            # 添加文字
            ax.text(50, 0, f'{current_step}/{max_steps} ({progress:.1f}%)',
                    ha='center', va='center', fontsize=12, weight='bold')

            # ETA
            elapsed = time.time() - self.stats['start_time']
            if progress > 0:
                eta = elapsed * (100 - progress) / progress
                eta_str = str(timedelta(seconds=int(eta)))
                ax.text(0.5, -0.5, f'ETA: {eta_str}',
                        transform=ax.transAxes, ha='center', va='top')
        else:
            # 如果没有最大步数，显示已训练时间
            elapsed = time.time() - self.stats['start_time']
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            ax.text(0.5, 0.5, f'Steps: {current_step}\nTime: {elapsed_str}',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=12)
            ax.set_title('Training Status')

        ax.set_yticks([])

    def _plot_loss_breakdown(self, ax):
        """绘制损失分解"""
        steps = list(self.history['step'])

        # 查找所有损失组件
        loss_components = {}
        for key in self.history.keys():
            if 'loss' in key.lower() and key not in ['loss', 'val_loss', 'step']:
                if '/' in key:  # 假设格式为 "loss/component"
                    loss_components[key] = list(self.history[key])

        if not loss_components:
            ax.text(0.5, 0.5, 'No loss breakdown available',
                    transform=ax.transAxes, ha='center', va='center')
            return

        # 堆叠面积图
        y_stack = []
        labels = []
        for key, values in loss_components.items():
            y_stack.append(values)
            labels.append(key.split('/')[-1])

        y_stack = np.array(y_stack)
        ax.stackplot(steps, y_stack, labels=labels, alpha=0.7)

        ax.set_xlabel('Steps')
        ax.set_ylabel('Loss Components')
        ax.set_title('Loss Breakdown')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_alerts_summary(self, ax):
        """绘制警告摘要"""
        if not self.alerts:
            ax.text(0.5, 0.5, 'No alerts',
                    transform=ax.transAxes, ha='center', va='center',
                    color='green', fontsize=14)
            return

        # 统计警告类型
        alert_counts = defaultdict(int)
        for alert in self.alerts:
            alert_counts[alert['type']] += 1

        # 绘制饼图
        if alert_counts:
            labels = list(alert_counts.keys())
            sizes = list(alert_counts.values())
            colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(labels)))

            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                              autopct='%1.0f', startangle=90)
            ax.set_title(f'Alerts Summary (Total: {sum(sizes)})')

            # 显示最近的警告
            recent_alerts = list(self.alerts)[-3:]
            alert_text = '\nRecent:\n'
            for alert in recent_alerts:
                alert_text += f"Step {alert['step']}: {alert['message'][:30]}...\n"

            ax.text(0.5, -0.3, alert_text, transform=ax.transAxes,
                    ha='center', va='top', fontsize=8)

    def _generate_detailed_plots(self):
        """生成详细的单独图表"""
        # 1. 详细的损失曲线对比
        self._save_detailed_loss_plot()

        # 2. 学习率和梯度关系图
        self._save_lr_gradient_plot()

        # 3. 长期趋势图
        self._save_long_term_trends()

        # 4. 参数演化热力图
        self._save_parameter_heatmap()

    def _save_detailed_loss_plot(self):
        """保存详细的损失对比图"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        steps = list(self.history['step'])

        # 原始损失
        ax1.set_title('Raw Loss Values')
        for key in self.history.keys():
            if 'loss' in key.lower() and key != 'step':
                values = list(self.history[key])
                if len(values) > 0:
                    ax1.plot(steps, values, label=key, alpha=0.7)
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 平滑损失
        ax2.set_title('Smoothed Loss Values (Moving Average)')
        for key in self.ma_history.keys():
            if 'loss' in key.lower():
                values = list(self.ma_history[key])
                if len(values) > 0:
                    ax2.plot(steps[-len(values):], values,
                             label=key.replace('_ma', ''), linewidth=2)
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'detailed_loss.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _save_lr_gradient_plot(self):
        """保存学习率和梯度关系图"""
        if 'lr' not in self.history or 'grad_norm' not in self.history:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        lr_values = list(self.history['lr'])
        grad_values = list(self.history['grad_norm'])

        # 确保长度一致
        min_len = min(len(lr_values), len(grad_values))
        lr_values = lr_values[:min_len]
        grad_values = grad_values[:min_len]

        # 散点图
        scatter = ax.scatter(lr_values, grad_values,
                             c=range(min_len), cmap='viridis',
                             alpha=0.6, s=20)

        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Learning Rate vs Gradient Norm')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # 颜色条表示训练进度
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Training Step')

        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'lr_gradient_relation.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _save_long_term_trends(self):
        """保存长期趋势图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        steps = list(self.history['step'])

        # 1. 损失趋势（对数尺度）
        ax = axes[0, 0]
        if 'loss' in self.history:
            loss_values = list(self.history['loss'])
            ax.semilogy(steps, loss_values, alpha=0.5)

            # 添加趋势线
            if len(steps) > 100:
                z = np.polyfit(steps[-1000:], np.log(loss_values[-1000:]), 1)
                p = np.poly1d(z)
                ax.semilogy(steps[-1000:], np.exp(p(steps[-1000:])),
                            'r--', label='Trend')
        ax.set_title('Loss Trend (Log Scale)')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. 训练效率
        ax = axes[0, 1]
        if 'speed/samples_per_second' in self.history:
            speed_values = list(self.history['speed/samples_per_second'])
            ax.plot(steps, speed_values, alpha=0.5)

            # 移动平均
            if len(speed_values) > 100:
                ma = pd.Series(speed_values).rolling(window=100).mean()
                ax.plot(steps, ma, 'r-', linewidth=2, label='MA-100')
        ax.set_title('Training Efficiency')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Samples/Second')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. 资源使用趋势
        ax = axes[1, 0]
        resource_keys = ['system/memory_percent', 'system/gpu0/memory_percent']
        for key in resource_keys:
            if key in self.history:
                values = list(self.history[key])
                label = 'RAM' if 'system/memory' in key else 'GPU'
                ax.plot(steps, values, label=label, alpha=0.7)
        ax.set_title('Resource Usage Trends')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Usage (%)')
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. 模型稳定性
        ax = axes[1, 1]
        stability_keys = ['model/param_std', 'model/grad_std']
        for key in stability_keys:
            if key in self.history:
                values = list(self.history[key])
                label = 'Param Std' if 'param' in key else 'Grad Std'
                ax.plot(steps, values, label=label)
        ax.set_title('Model Stability')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Standard Deviation')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'long_term_trends.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _save_parameter_heatmap(self):
        """保存参数演化热力图"""
        # 收集层级参数统计
        layer_stats = defaultdict(list)

        for key in self.history.keys():
            if key.startswith('layers/'):
                parts = key.split('/')
                if len(parts) >= 3:
                    layer_name = parts[1]
                    stat_type = parts[2]
                    if 'mean' in stat_type:
                        layer_stats[layer_name].extend(list(self.history[key]))

        if not layer_stats:
            return

        # 创建热力图数据
        layers = sorted(layer_stats.keys())
        data = []
        for layer in layers:
            values = layer_stats[layer]
            # 采样以减少数据量
            if len(values) > 100:
                indices = np.linspace(0, len(values) - 1, 100, dtype=int)
                values = [values[i] for i in indices]
            data.append(values)

        if not data:
            return

        data = np.array(data)

        # 绘制热力图
        fig, ax = plt.subplots(figsize=(12, 8))

        im = ax.imshow(data, aspect='auto', cmap='coolwarm',
                       interpolation='nearest')

        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers)
        ax.set_xlabel('Training Progress')
        ax.set_ylabel('Layer')
        ax.set_title('Parameter Evolution Heatmap')

        # 颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Parameter Mean')

        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'parameter_evolution.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    def save_metrics(self):
        """保存所有指标到文件"""
        # 保存为CSV
        df_data = {}
        for key, values in self.history.items():
            if len(values) > 0:
                df_data[key] = list(values)

        if df_data:
            # 确保所有列长度相同
            max_len = max(len(v) for v in df_data.values())
            for key in df_data:
                if len(df_data[key]) < max_len:
                    df_data[key].extend([None] * (max_len - len(df_data[key])))

            df = pd.DataFrame(df_data)
            csv_path = os.path.join(self.metric_dir, 'training_metrics.csv')
            df.to_csv(csv_path, index=False)

        # 保存统计信息
        stats_path = os.path.join(self.metric_dir, 'training_stats.json')
        with open(stats_path, 'w') as f:
            # 转换时间戳为可读格式
            stats_copy = self.stats.copy()
            stats_copy['start_time'] = datetime.fromtimestamp(
                self.stats['start_time']).isoformat()
            stats_copy['current_time'] = datetime.now().isoformat()
            stats_copy['elapsed_time'] = str(timedelta(
                seconds=int(time.time() - self.stats['start_time'])))

            json.dump(stats_copy, f, indent=4)

        # 保存警告日志
        if self.alerts:
            alerts_path = os.path.join(self.metric_dir, 'alerts.json')
            with open(alerts_path, 'w') as f:
                json.dump(list(self.alerts), f, indent=4)

    def generate_report(self):
        """生成训练报告"""
        report = []
        report.append("=" * 80)
        report.append(f"TRELLIS Training Report - {self.experiment_name}")
        report.append("=" * 80)
        report.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # 基本信息
        report.append("## Training Configuration")
        if self.config:
            for key, value in self.config.items():
                report.append(f"  - {key}: {value}")
        report.append("")

        # 训练统计
        report.append("## Training Statistics")
        elapsed = time.time() - self.stats['start_time']
        report.append(f"  - Total Steps: {self.stats['total_steps']}")
        report.append(f"  - Total Samples: {self.stats['total_samples']}")
        report.append(f"  - Training Time: {str(timedelta(seconds=int(elapsed)))}")
        report.append(f"  - Best Loss: {self.stats['best_loss']:.6f} (Step {self.stats['best_step']})")
        report.append(f"  - NaN Count: {self.stats['nan_count']}")
        report.append(f"  - Gradient Explosions: {self.stats['gradient_explosion_count']}")
        report.append("")

        # 最终指标
        report.append("## Final Metrics")
        for key in sorted(self.current_metrics.keys()):
            if key not in ['step', 'timestamp']:
                value = self.current_metrics[key]
                if isinstance(value, float):
                    report.append(f"  - {key}: {value:.6f}")
                else:
                    report.append(f"  - {key}: {value}")
        report.append("")

        # 警告摘要
        if self.alerts:
            report.append("## Alerts Summary")
            alert_counts = defaultdict(int)
            for alert in self.alerts:
                alert_counts[alert['type']] += 1
            for alert_type, count in alert_counts.items():
                report.append(f"  - {alert_type}: {count}")
        else:
            report.append("## No alerts during training")
        report.append("")

        # 保存报告
        report_path = os.path.join(self.metric_dir, 'training_report.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))

        return '\n'.join(report)

    def close(self):
        """关闭监控器"""
        # 生成最终报告
        self.generate_report()

        # 保存最终指标
        self.save_metrics()

        # 生成最终图表
        if self.enable_plots:
            self.plot_all_metrics()

        # 关闭TensorBoard
        if self.writer:
            self.writer.close()

        # 关闭图形
        if self.fig:
            plt.close(self.fig)

        print(f"Training monitor closed. Results saved to: {self.output_dir}")


# 使用示例
def integrate_with_trellis_trainer(trainer, monitor):
    """
    将监控器集成到TRELLIS训练器中
    """
    # 在训练循环中
    for step in range(trainer.start_step, trainer.max_steps):
        start_time = time.time()

        # 执行训练步骤
        loss = trainer.train_step()

        # 计算训练时间
        step_time = time.time() - start_time

        # 收集所有指标
        metrics = {
            'loss': loss.item(),
            'lr': trainer.optimizer.param_groups[0]['lr'],
            'grad_norm': trainer.grad_norm,
            # 添加更多TRELLIS特定的指标
            'loss/reconstruction': trainer.recon_loss.item(),
            'loss/kl_divergence': trainer.kl_loss.item(),
            'loss/perceptual': trainer.perceptual_loss.item(),
            # 3D相关指标
            'quality/chamfer_distance': trainer.chamfer_distance,
            'quality/mesh_quality': trainer.mesh_quality_score,
            # 其他
            'batch_size': trainer.batch_size,
        }

        # 记录到监控器
        monitor.log_metrics(metrics, step)
        monitor.log_system_stats(step)
        monitor.log_training_speed(trainer.batch_size, step, step_time)

        # 定期记录模型统计
        if step % 1000 == 0:
            monitor.log_model_stats(trainer.model, step)

        # 定期记录3D指标
        if step % 5000 == 0:
            monitor.log_3d_metrics(trainer.compute_3d_metrics(), step)


if __name__ == "__main__":
    # 创建监控器
    monitor = ComprehensiveMonitor(
        output_dir="./outputs/metafood3d_training",
        experiment_name="metafood3d_finetuning",
        config={
            'model': 'TRELLIS-L',
            'dataset': 'MetaFood3D',
            'batch_size': 4,
            'max_steps': 100000,
            'learning_rate': 1e-5
        },
        enable_tensorboard=True,
        enable_plots=True,
        plot_interval=100,
        save_interval=1000
    )

    # 模拟训练过程
    for step in range(1000):
        # 模拟指标
        metrics = {
            'loss': np.exp(-step / 1000) + np.random.normal(0, 0.01),
            'lr': 1e-4 * (0.5 ** (step // 200)),
            'grad_norm': np.random.lognormal(0, 0.5),
            'loss/reconstruction': np.exp(-step / 800) * 0.8,
            'loss/kl_divergence': np.exp(-step / 1200) * 0.2,
            'quality/chamfer_distance': 1.0 / (1 + step / 100),
            'speed/samples_per_second': 10 + np.random.normal(0, 1),
        }

        monitor.log_metrics(metrics, step)

        if step % 10 == 0:
            monitor.log_system_stats(step)
            monitor.log_training_speed(4, step, 0.5)

    # 关闭监控器
    monitor.close()