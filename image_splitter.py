import os
import argparse
from PIL import Image
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def split_and_save_image(args):
    """
    处理单个图像的分割和保存
    """
    input_path, input_root, output_root = args
    try:
        with Image.open(input_path) as img:
            width, height = img.size
            mid_x = width // 2
            mid_y = height // 2

            regions = [
                (0, 0, mid_x, mid_y),
                (mid_x, 0, width, mid_y),
                (0, mid_y, mid_x, height),
                (mid_x, mid_y, width, height)
            ]

            rel_path = os.path.relpath(input_path, input_root)
            dir_part, filename = os.path.split(rel_path)
            base_name = os.path.splitext(filename)[0]
            output_dir = os.path.join(output_root, dir_part, base_name)
            
            os.makedirs(output_dir, exist_ok=True)

            for i, box in enumerate(regions, 1):
                output_path = os.path.join(output_dir, f"{i}.png")
                img.crop(box).save(output_path, "PNG",  optimize=True)
        return True  # 成功标记
    except Exception as e:
        print(f"\nError processing {input_path}: {str(e)}")
        return False  # 失败标记

def process_images(input_root, output_root, num_processes):
    """
    处理整个目录树的图像
    """
    # 收集所有图像文件路径
    image_paths = []
    for root, _, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_paths.append(os.path.join(root, file))

    # 准备任务参数
    tasks = [(path, input_root, output_root) for path in image_paths]

    # 使用带进度条的进程池
    with Pool(processes=num_processes) as pool:
        # 创建进度条
        with tqdm(total=len(tasks), desc="Processing images", unit="img") as pbar:
            # 使用imap_unordered获取最快完成的任务
            for result in pool.imap_unordered(split_and_save_image, tasks):
                if result:
                    pbar.update(1)  # 成功时更新进度
                else:
                    pbar.total -= 1  # 失败时调整总数
                    pbar.refresh()  # 刷新显示

            pbar.close()  # 确保进度条关闭

    # 打印统计信息
    print(f"\nProcessed {len(image_paths)} files")
    print(f"Success: {len(image_paths) - pbar.n} | Failed: {pbar.total - pbar.n}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="带进度条的批量图像分割处理器")
    parser.add_argument("--input", required=True, help="输入目录路径")
    parser.add_argument("--output", required=True, help="输出目录路径")
    parser.add_argument("--processes", type=int, default=cpu_count(), 
                       help="使用的进程数（默认CPU核心数）")
    
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        raise ValueError(f"输入目录不存在: {args.input}")
    os.makedirs(args.output, exist_ok=True)

    process_images(args.input, args.output, args.processes)