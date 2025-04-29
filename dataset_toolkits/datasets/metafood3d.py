# 文件路径: dataset_toolkits/datasets/metafood3d.py
import os
import argparse
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from utils import get_file_hash


def add_args(parser: argparse.ArgumentParser):
    """添加特定于MetaFood3D数据集的参数"""
    pass


def get_metadata(output_dir, **kwargs):
    """为MetaFood3D数据集创建或加载元数据"""
    metadata_path = os.path.join(output_dir, 'raw_metadata.csv')

    # 如果元数据已存在，直接加载
    if os.path.exists(metadata_path):
        return pd.read_csv(metadata_path)

    print("构建MetaFood3D数据集的元数据...")

    # 假设原始数据集位于"metafood3d"目录
    dataset_dir = os.path.expanduser("/root/autodl-tmp/metafood3d")  # 调整为实际路径
    metadata = []

    # 扫描所有类别
    for category in sorted(os.listdir(dataset_dir)):
        category_path = os.path.join(dataset_dir, category)
        if not os.path.isdir(category_path):
            continue

        # 扫描每个类别下的实例
        for instance in sorted(os.listdir(category_path)):
            instance_path = os.path.join(category_path, instance)
            if not os.path.isdir(instance_path):
                continue

            # 查找OBJ文件
            obj_files = [f for f in os.listdir(instance_path) if f.endswith('.obj')]
            if not obj_files:
                continue

            obj_file = obj_files[0]  # 假设每个实例只有一个OBJ文件
            obj_path = os.path.join(instance_path, obj_file)

            # 计算SHA256哈希值作为唯一标识符
            sha256 = get_file_hash(obj_path)

            # 构建文件标识符
            file_identifier = f"{category}/{instance}/{obj_file}"

            # 创建带有描述的记录
            record = {
                "sha256": sha256,
                "file_identifier": file_identifier,
                "captions": f"A 3D model of {category.lower()} - {instance.replace('_', ' ')}",
                "aesthetic_score": 5.0,  # 默认美学评分
                "category": category,
                "instance": instance
            }
            metadata.append(record)

    # 转换为DataFrame并保存
    df = pd.DataFrame(metadata)
    df.to_csv(metadata_path, index=False)
    print(f"元数据已保存到 {metadata_path}，共 {len(df)} 个模型")

    return df


def download(metadata, output_dir, **kwargs):
    """将数据集中的原始OBJ文件复制到指定目录"""
    os.makedirs(os.path.join(output_dir, 'raw'), exist_ok=True)

    dataset_dir = os.path.expanduser("/root/autodl-tmp/metafood3d")  # 调整为实际路径
    downloaded = {}

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor, \
            tqdm(total=len(metadata), desc="复制模型文件") as pbar:

        def copy_file(row):
            try:
                file_id = row["file_identifier"]
                sha256 = row["sha256"]

                # 源文件路径
                src_path = os.path.join(dataset_dir, file_id)

                # 目标文件路径
                dst_dir = os.path.join(output_dir, 'raw', sha256)
                os.makedirs(dst_dir, exist_ok=True)
                dst_path = os.path.join(dst_dir, "model.obj")

                # 复制文件
                if not os.path.exists(dst_path):
                    import shutil
                    shutil.copy2(src_path, dst_path)

                # 复制相关的材质和纹理文件
                src_dir = os.path.dirname(src_path)
                for f in os.listdir(src_dir):
                    if f.endswith('.mtl') or f.endswith('.png') or f.endswith('.jpg'):
                        shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f))

                # 记录相对路径
                rel_path = os.path.join('raw', sha256, "model.obj")
                downloaded[sha256] = rel_path

                pbar.update()
                return sha256, rel_path
            except Exception as e:
                print(f"处理 {file_id} 时出错: {e}")
                pbar.update()
                return None

        results = list(executor.map(copy_file, metadata.to_dict('records')))
        results = [r for r in results if r is not None]

    return pd.DataFrame(results, columns=['sha256', 'local_path'])


def foreach_instance(metadata, output_dir, func, max_workers=None, desc='处理对象'):
    """对数据集中的每个实例应用函数"""
    metadata_records = metadata.to_dict('records')
    records = []
    max_workers = max_workers or os.cpu_count()

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor, \
                tqdm(total=len(metadata_records), desc=desc) as pbar:

            def worker(metadatum):
                try:
                    local_path = metadatum['local_path']
                    sha256 = metadatum['sha256']
                    file = os.path.join(output_dir, local_path)
                    record = func(file, sha256)
                    if record is not None:
                        records.append(record)
                    pbar.update()
                except Exception as e:
                    print(f"处理对象 {sha256} 时出错: {e}")
                    pbar.update()

            executor.map(worker, metadata_records)
            executor.shutdown(wait=True)
    except Exception as e:
        print(f"处理过程中发生错误: {e}")

    return pd.DataFrame.from_records(records)
