import os

# 设置环境变量
os.environ['http_proxy'] = 'http://127.0.0.1:8889'
os.environ['https_proxy'] = 'http://127.0.0.1:8889'
os.environ['CUDA_HOME'] = '/usr/local/cuda-11.8'
os.environ['ATTN_BACKEND'] = 'xformers'  # 可以是 'flash-attn' 或 'xformers'，默认是 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'  # 可以是 'native' 或 'auto'，默认是 'auto'
# 'auto' 更快但会在开始时进行基准测试
# 如果只运行一次，建议设置为 'native'

import datetime
import glob
import numpy as np
import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

# 创建时间戳目录
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"./{timestamp}"
os.makedirs(output_dir, exist_ok=True)
print(f"结果将保存到: {output_dir}")

# 加载模型
pipeline = TrellisImageTo3DPipeline.from_pretrained("/home/public/PycharmProjects/TRELLIS/test")
pipeline.cuda()


# 加载图片函数
def load_images(image_paths):
    """加载多张图片"""
    if isinstance(image_paths, str):
        # 如果输入是一个目录，则加载所有图片
        if os.path.isdir(image_paths):
            image_files = glob.glob(os.path.join(image_paths, "*.jpg")) + \
                          glob.glob(os.path.join(image_paths, "*.jpeg")) + \
                          glob.glob(os.path.join(image_paths, "*.png"))
            print(f"从目录 {image_paths} 找到 {len(image_files)} 张图片")
        else:
            # 单张图片路径
            image_files = [image_paths]
    else:
        # 多张图片路径列表
        image_files = image_paths

    images = []
    image_names = []

    for file_path in image_files:
        try:
            img = Image.open(file_path)
            images.append(img)
            image_names.append(os.path.splitext(os.path.basename(file_path))[0])
            print(f"已加载图片: {file_path}")
        except Exception as e:
            print(f"无法加载图片 {file_path}: {e}")

    return images, image_names


# 可以修改这里指定一个包含多张图片的目录
# 或者直接指定多张图片的路径列表
image_path = "assets/example_image"  # 这里可以是目录或图片路径列表
images, image_names = load_images(image_path)

if not images:
    print("没有找到有效图片，退出程序")
    exit(1)

print(f"开始处理 {len(images)} 张图片...")

# 运行模型生成3D资产
outputs = pipeline.run_multi_image(
    images,
    seed=1,
    # 可选参数
    # sparse_structure_sampler_params={
    #     "steps": 12,
    #     "cfg_strength": 7.5,
    # },
    # slat_sampler_params={
    #     "steps": 12,
    #     "cfg_strength": 3,
    # },
)

# 保存所有模型生成的资产
print("正在生成和保存结果...")

# 生成组合视频 (可选)
if len(images) > 1:
    try:
        video_gaussian = render_utils.render_video(outputs['gaussian'][0])['color']
        video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
        combined_video = [np.concatenate([frame_gs, frame_mesh], axis=1) for frame_gs, frame_mesh in
                          zip(video_gaussian, video_mesh)]
        imageio.mimsave(os.path.join(output_dir, "combined_video.mp4"), combined_video, fps=30)
        print(f"已保存组合视频")
    except Exception as e:
        print(f"生成组合视频时出错: {e}")

# 保存每个模型的单独输出
for i, (gaussian, radiance_field, mesh) in enumerate(
        zip(outputs['gaussian'], outputs['radiance_field'], outputs['mesh'])):
    file_prefix = image_names[i] if i < len(image_names) else f"object_{i + 1}"
    print(f"处理对象 {i + 1}/{len(outputs['gaussian'])}: {file_prefix}")

    # 保存高斯渲染视频
    video = render_utils.render_video(gaussian)['color']
    video_output_path = os.path.join(output_dir, f"{file_prefix}_gaussian.mp4")
    imageio.mimsave(video_output_path, video, fps=30)
    print(f"  已保存高斯视频: {video_output_path}")

    # 保存辐射场渲染视频
    video = render_utils.render_video(radiance_field)['color']
    video_output_path = os.path.join(output_dir, f"{file_prefix}_radiance_field.mp4")
    imageio.mimsave(video_output_path, video, fps=30)
    print(f"  已保存辐射场视频: {video_output_path}")

    # 保存网格渲染视频
    video = render_utils.render_video(mesh)['normal']
    video_output_path = os.path.join(output_dir, f"{file_prefix}_mesh.mp4")
    imageio.mimsave(video_output_path, video, fps=30)
    print(f"  已保存网格视频: {video_output_path}")

    # 保存PLY文件
    ply_output_path = os.path.join(output_dir, f"{file_prefix}.ply")
    gaussian.save_ply(ply_output_path)
    print(f"  已保存PLY文件: {ply_output_path}")

    # 保存GLB文件
    try:
        glb = postprocessing_utils.to_glb(
            gaussian,
            mesh,
            simplify=0.95,  # 简化过程中要移除的三角形比例
            texture_size=1024,  # GLB使用的纹理大小
        )
        glb_output_path = os.path.join(output_dir, f"{file_prefix}.glb")
        glb.export(glb_output_path)
        print(f"  已保存GLB文件: {glb_output_path}")
    except Exception as e:
        print(f"  生成GLB文件时出错: {e}")

print(f"\n所有结果已保存到: {output_dir}")
