import os
import torch
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
import pickle


def compute_slat_normalization(
        data_dir: str,
        subset: str = "MetaFood3D",
        num_channels: int = 8
):
    """
    从已编码的SLat数据计算normalization参数

    Args:
        data_dir: 数据集根目录（如 datasets/MetaFood3D）
        subset: 数据集子集名称
        num_channels: latent通道数（默认8）

    Returns:
        mean_list: 各通道均值
        std_list: 各通道标准差
    """

    # 1. 定位latent文件
    latent_dir = Path(data_dir) / "latents"
    if not latent_dir.exists():
        print(f"错误：找不到latent目录 {latent_dir}")
        print("请确保已运行 encode_latent.py")
        return None, None

    # 获取所有latent文件
    latent_files = list(latent_dir.glob("*.pt"))
    if not latent_files:
        latent_files = list(latent_dir.glob("*.pkl"))
    if not latent_files:
        latent_files = list(latent_dir.glob("*.npz"))

    print(f"找到 {len(latent_files)} 个latent文件")

    # 2. 初始化统计量（使用Welford算法）
    n_samples = 0
    mean = torch.zeros(num_channels, dtype=torch.float64)
    M2 = torch.zeros(num_channels, dtype=torch.float64)

    # 3. 遍历所有latent文件
    for file_path in tqdm(latent_files, desc="计算统计量"):
        try:
            # 加载latent数据
            if file_path.suffix == '.pt':
                latent = torch.load(file_path, map_location='cpu')
            elif file_path.suffix == '.pkl':
                with open(file_path, 'rb') as f:
                    latent = pickle.load(f)
                    if isinstance(latent, np.ndarray):
                        latent = torch.from_numpy(latent)
            elif file_path.suffix == '.npz':
                data = np.load(file_path)
                latent = torch.from_numpy(data['latent'])

            # 确保是torch tensor
            if not isinstance(latent, torch.Tensor):
                latent = torch.tensor(latent, dtype=torch.float32)

            # 检查形状
            if latent.dim() == 1:
                # 如果是1D，确保长度是8
                assert latent.shape[0] == num_channels, f"期望{num_channels}通道，得到{latent.shape[0]}"
                sample_mean = latent
                n_points = 1
            elif latent.dim() == 2:
                # 如果是2D，第一维应该是通道数
                if latent.shape[0] == num_channels:
                    sample_mean = latent.mean(dim=1)
                    n_points = latent.shape[1]
                elif latent.shape[1] == num_channels:
                    sample_mean = latent.mean(dim=0)
                    n_points = latent.shape[0]
                else:
                    print(f"警告：跳过形状不匹配的文件 {file_path}: {latent.shape}")
                    continue
            else:
                print(f"警告：跳过维度不支持的文件 {file_path}: {latent.dim()}D")
                continue

            # Welford算法更新
            n_samples += n_points
            delta = sample_mean.double() - mean
            mean += delta * n_points / n_samples
            delta2 = sample_mean.double() - mean
            M2 += delta * delta2 * n_points

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            continue

    # 4. 计算最终的std
    if n_samples < 2:
        print("错误：样本数量不足")
        return None, None

    variance = M2 / (n_samples - 1)
    std = torch.sqrt(variance)

    # 5. 转换为列表
    mean_list = mean.numpy().tolist()
    std_list = std.numpy().tolist()

    return mean_list, std_list


def verify_normalization(data_dir: str, mean: list, std: list, num_samples: int = 100):
    """
    验证计算的normalization参数
    """
    latent_dir = Path(data_dir) / "latents"
    latent_files = list(latent_dir.glob("*.pt"))[:num_samples]

    mean_tensor = torch.tensor(mean)
    std_tensor = torch.tensor(std)

    normalized_values = []

    for file_path in latent_files:
        latent = torch.load(file_path, map_location='cpu')
        if latent.dim() == 2:
            latent = latent.mean(dim=1) if latent.shape[0] == 8 else latent.mean(dim=0)

        normalized = (latent - mean_tensor) / std_tensor
        normalized_values.append(normalized)

    all_normalized = torch.stack(normalized_values)

    print("\n归一化后的统计量验证：")
    print(f"均值: {all_normalized.mean(dim=0).tolist()}")
    print(f"标准差: {all_normalized.std(dim=0).tolist()}")
    print("（应该接近0和1）")


# 主函数
if __name__ == "__main__":
    # 配置
    data_dir = "./datasets/MetaFood3D"  # 您的数据集路径

    # 计算normalization
    print("开始计算SLat normalization参数...")
    mean, std = compute_slat_normalization(data_dir)

    if mean is not None and std is not None:
        print("\n计算完成！")
        print(f"Mean: {mean}")
        print(f"Std: {std}")
        print(f"Mean范围: [{min(mean):.4f}, {max(mean):.4f}]")
        print(f"Std范围: [{min(std):.4f}, {max(std):.4f}]")

        # 验证
        verify_normalization(data_dir, mean, std)

        # 创建完整的配置
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
        output_file = "metafood3d_normalization.json"
        with open(output_file, "w") as f:
            json.dump(config_update, f, indent=4)

        print(f"\n配置已保存到: {output_file}")

        # 生成更新后的完整配置文件
        print("\n在您的训练配置文件中，更新normalization部分为：")
        print(json.dumps(config_update["dataset"]["args"]["normalization"], indent=4))
