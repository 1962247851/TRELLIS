import os
import torch
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path


def compute_metafood3d_slat_normalization(
        data_dir: str,
        num_channels: int = 8
):
    """
    计算MetaFood3D数据集的SLat normalization参数
    针对稀疏格式的latent数据（包含feats和coords）
    """
    latent_dir = Path(data_dir) / "latents"

    if not latent_dir.exists():
        print(f"错误：找不到latent目录 {latent_dir}")
        return None, None

    # 获取所有.npz文件
    latent_files = list(latent_dir.glob("*.npz"))
    print(f"找到 {len(latent_files)} 个latent文件")

    if len(latent_files) == 0:
        print("错误：没有找到任何.npz文件")
        return None, None

    # 初始化统计量（使用Welford算法）
    n_total_points = 0
    mean = np.zeros(num_channels, dtype=np.float64)
    M2 = np.zeros(num_channels, dtype=np.float64)

    # 用于存储每个文件的统计信息
    file_stats = []

    # 遍历所有文件
    for file_path in tqdm(latent_files, desc="计算统计量"):
        try:
            # 加载数据
            data = np.load(file_path)

            # 检查是否包含必要的键
            if 'feats' not in data:
                print(f"警告：文件 {file_path.name} 不包含 'feats' 键")
                continue

            # 获取特征数据
            feats = data['feats']  # shape: (N, 8)

            # 验证形状
            if feats.shape[1] != num_channels:
                print(f"警告：特征维度不匹配 {feats.shape[1]} != {num_channels}")
                continue

            # 获取当前文件的点数
            n_points = feats.shape[0]

            # 使用Welford算法更新全局统计量
            for i in range(n_points):
                n_total_points += 1
                delta = feats[i] - mean
                mean += delta / n_total_points
                delta2 = feats[i] - mean
                M2 += delta * delta2

            # 记录文件统计信息
            file_stats.append({
                'file': file_path.name,
                'n_points': n_points,
                'feats_mean': feats.mean(axis=0),
                'feats_std': feats.std(axis=0)
            })

        except Exception as e:
            print(f"处理文件 {file_path.name} 时出错: {e}")
            continue

    # 计算最终的标准差
    if n_total_points < 2:
        print("错误：总点数不足")
        return None, None

    variance = M2 / (n_total_points - 1)
    std = np.sqrt(variance)

    # 转换为列表
    mean_list = mean.tolist()
    std_list = std.tolist()

    # 打印统计信息
    print(f"\n=== 统计结果 ===")
    print(f"处理文件数: {len(file_stats)}")
    print(f"总点数: {n_total_points}")
    print(f"平均每个文件的点数: {n_total_points / len(file_stats):.1f}")
    print(f"\n全局统计：")
    print(f"Mean: {[f'{m:.6f}' for m in mean_list]}")
    print(f"Std:  {[f'{s:.6f}' for s in std_list]}")
    print(f"\nMean范围: [{min(mean_list):.6f}, {max(mean_list):.6f}]")
    print(f"Std范围:  [{min(std_list):.6f}, {max(std_list):.6f}]")

    # 检查异常
    if any(s < 1e-6 for s in std_list):
        print("\n警告：存在非常小的std值！")
        for i, s in enumerate(std_list):
            if s < 1e-6:
                print(f"  通道 {i}: std = {s}")

    # 分析各文件的分布
    print(f"\n=== 文件分布分析（前10个）===")
    for i, stats in enumerate(file_stats[:10]):
        print(f"文件 {i + 1}: {stats['n_points']} 点")
        print(f"  Mean: {[f'{m:.3f}' for m in stats['feats_mean']]}")
        print(f"  Std:  {[f'{s:.3f}' for s in stats['feats_std']]}")

    return mean_list, std_list


def verify_normalization(data_dir: str, mean: list, std: list, num_samples: int = 10):
    """
    验证计算的normalization参数
    """
    latent_dir = Path(data_dir) / "latents"
    latent_files = list(latent_dir.glob("*.npz"))[:num_samples]

    print(f"\n=== 验证Normalization（使用{len(latent_files)}个文件）===")

    mean_array = np.array(mean)
    std_array = np.array(std)

    all_normalized = []

    for file_path in latent_files:
        data = np.load(file_path)
        feats = data['feats']

        # 归一化
        normalized = (feats - mean_array) / std_array
        all_normalized.append(normalized)

    # 合并所有归一化后的数据
    all_normalized = np.vstack(all_normalized)

    # 计算归一化后的统计量
    normalized_mean = all_normalized.mean(axis=0)
    normalized_std = all_normalized.std(axis=0)

    print("归一化后的统计量：")
    print(f"Mean: {[f'{m:.6f}' for m in normalized_mean]}")
    print(f"Std:  {[f'{s:.6f}' for s in normalized_std]}")
    print("（理想情况应该接近0和1）")

    # 检查范围
    print(f"\n归一化后的值范围：")
    print(f"Min: {all_normalized.min(axis=0)}")
    print(f"Max: {all_normalized.max(axis=0)}")


def save_normalization_config(mean: list, std: list, output_file: str = "metafood3d_normalization.json"):
    """
    保存normalization配置
    """
    # 创建完整的配置更新
    config_update = {
        "dataset": {
            "args": {
                "normalization": {
                    "mean": mean,
                    "std": std
                }
            }
        }
    }

    # 保存到文件
    with open(output_file, "w") as f:
        json.dump(config_update, f, indent=4)

    print(f"\n配置已保存到: {output_file}")

    # 同时生成可以直接复制到配置文件的格式
    print("\n可以直接复制到训练配置文件的内容：")
    print('"normalization": {')
    print(f'    "mean": {mean},')
    print(f'    "std": {std}')
    print('}')


# 主函数
if __name__ == "__main__":
    data_dir = "./datasets/MetaFood3D"  # 修改为您的实际路径

    # 计算normalization参数
    mean, std = compute_metafood3d_slat_normalization(data_dir)

    if mean is not None and std is not None:
        # 验证结果
        verify_normalization(data_dir, mean, std)

        # 保存配置
        save_normalization_config(mean, std)

        # 额外分析
        print("\n=== 额外分析 ===")

        # 检查哪些通道变化最大
        std_array = np.array(std)
        sorted_indices = np.argsort(std_array)[::-1]
        print("\n通道按标准差排序（从大到小）：")
        for i, idx in enumerate(sorted_indices):
            print(f"  通道 {idx}: std = {std[idx]:.6f}, mean = {mean[idx]:.6f}")
