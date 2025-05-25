import os
import time
import datetime
import glob
from pathlib import Path

# 设置环境变量
os.environ['http_proxy'] = 'http://127.0.0.1:8889'
os.environ['https_proxy'] = 'http://127.0.0.1:8889'
os.environ['CUDA_HOME'] = '/usr/local/cuda-11.8'
os.environ['ATTN_BACKEND'] = 'xformers'  # 可以是 'flash-attn' 或 'xformers'，默认为 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'  # 可以是 'native' 或 'auto'，默认为 'auto'
# 'auto' 速度更快但会在开始时进行基准测试
# 如果只运行一次，推荐设置为 'native'

import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils


def load_model(model_path="/home/public/PycharmProjects/TRELLIS/test"):
    """加载模型并返回，确保只加载一次"""
    print(f"正在加载模型 {model_path}...")
    model_load_start = time.time()
    pipeline = TrellisImageTo3DPipeline.from_pretrained(model_path)
    pipeline.cuda()
    model_load_time = time.time() - model_load_start
    print(f"模型加载完成，耗时 {model_load_time:.2f} 秒")
    return pipeline


def process_images(pipeline, input_dir="assets/example_image", seed=1):
    """
    使用预加载的模型处理指定目录下的所有图片

    参数:
    pipeline: 预加载的模型
    input_dir: 输入图片目录
    seed: 随机种子
    """
    # 记录总开始时间
    total_start_time = time.time()

    # 创建以当前时间命名的主文件夹
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # 查找所有图片文件
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(input_dir, ext.upper())))

    if not image_paths:
        print(f"在 {input_dir} 中未找到图片")
        return

    print(f"找到 {len(image_paths)} 张图片需要处理")

    # 处理每个图片
    successful = 0
    failed = 0

    for i, image_path in enumerate(image_paths):
        # 记录单个图片处理开始时间
        img_start_time = time.time()

        # 获取图片名（不含扩展名）作为子目录名
        image_name = Path(image_path).stem
        image_output_dir = os.path.join(output_dir, image_name)
        os.makedirs(image_output_dir, exist_ok=True)

        print(f"[{i + 1}/{len(image_paths)}] 正在处理图片: {image_path}")

        try:
            # 加载图片
            image = Image.open(image_path)

            # 运行pipeline
            outputs = pipeline.run(
                image,
                seed=seed,
                # 可选参数可以在这里添加
            )

            # 渲染并保存结果
            video = render_utils.render_video(outputs['gaussian'][0])['color']
            imageio.mimsave(os.path.join(image_output_dir, f"{image_name}_gs.mp4"), video, fps=30)

            video = render_utils.render_video(outputs['radiance_field'][0])['color']
            imageio.mimsave(os.path.join(image_output_dir, f"{image_name}_rf.mp4"), video, fps=30)

            video = render_utils.render_video(outputs['mesh'][0])['normal']
            imageio.mimsave(os.path.join(image_output_dir, f"{image_name}_mesh.mp4"), video, fps=30)

            # 保存GLB文件
            glb = postprocessing_utils.to_glb(
                outputs['gaussian'][0],
                outputs['mesh'][0],
                simplify=0.95,
                texture_size=1024,
            )
            glb.export(os.path.join(image_output_dir, f"{image_name}.glb"))

            # 保存Gaussians作为PLY文件
            outputs['gaussian'][0].save_ply(os.path.join(image_output_dir, f"{image_name}.ply"))

            # 计算处理时间
            img_elapsed_time = time.time() - img_start_time
            print(f"✓ 成功处理 {image_path}, 耗时 {img_elapsed_time:.2f} 秒")
            successful += 1

        except Exception as e:
            print(f"✗ 处理 {image_path} 失败: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1

    # 计算总处理时间
    total_elapsed_time = time.time() - total_start_time
    hours, remainder = divmod(total_elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"\n处理完成: {successful} 个成功, {failed} 个失败")
    print(f"总处理时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
    print(f"结果保存在: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    # 如果需要命令行参数，可以取消下面注释并添加argparse相关代码
    """
    import argparse
    parser = argparse.ArgumentParser(description="使用TRELLIS处理多张图片")
    parser.add_argument("--input_dir", type=str, default="assets/example_image", 
                        help="包含输入图片的目录")
    parser.add_argument("--model_path", type=str, default="/home/public/PycharmProjects/TRELLIS/test", 
                        help="TRELLIS模型路径")
    parser.add_argument("--seed", type=int, default=1, 
                        help="随机种子，用于可重现性")

    args = parser.parse_args()

    # 只加载一次模型
    pipeline = load_model(args.model_path)
    process_images(pipeline, args.input_dir, args.seed)
    """

    # 使用默认参数
    model_path = "/home/public/PycharmProjects/TRELLIS/test"
    input_dir = "assets/example_image"
    seed = 1

    # 只加载一次模型
    pipeline = load_model(model_path)
    # 使用加载好的模型处理所有图片
    process_images(pipeline, input_dir, seed)