import os
import random
from PIL import Image
import multiprocessing

def collect_image_files(source_dir):
    """收集源目录中所有支持的图像文件"""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_files = []
    
    for root, _, files in os.walk(source_dir):
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext in valid_extensions:
                image_files.append(os.path.join(root, filename))
    return image_files

def process_single_image(args):
    """处理单个图像文件（供多进程调用）"""
    file_path, index, target_dir, crop_size = args
    
    try:
        with Image.open(file_path) as img:
            # 获取图像尺寸
            width, height = img.size
            
            # 跳过尺寸不足的图像
            if width < crop_size or height < crop_size:
                print(f"Skipping {file_path} (minimum size {crop_size}x{crop_size})")
                return
            
            # 生成随机裁剪位置
            left = random.randint(0, width - crop_size)
            upper = random.randint(0, height - crop_size)
            right = left + crop_size
            lower = upper + crop_size
            
            # 执行裁剪并保存
            cropped_img = img.crop((left, upper, right, lower))
            output_path = os.path.join(target_dir, f"{index:05d}.png")
            cropped_img.save(output_path)
            
            print(f"Processed: {file_path} -> {output_path}")
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

def process_images_parallel(source_dir, target_dir, crop_size=256, workers=None):
    """并行处理图像的主函数"""
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)
    
    # 收集所有图像文件并编号
    image_files = collect_image_files(source_dir)
    print(f"Found {len(image_files)} image files to process")
    
    # 准备多进程参数
    task_args = [
        (file_path, idx, target_dir, crop_size)
        for idx, file_path in enumerate(image_files)
    ]
    
    # 创建进程池并行处理
    with multiprocessing.Pool(processes=workers) as pool:
        pool.map(process_single_image, task_args)

if __name__ == "__main__":
    # 配置参数（按需修改）
    SOURCE_DIR = "/home/ubuntu/ssd/lsdir/train"
    TARGET_DIR = "/home/ubuntu/ssd/lsdir_patches"
    CROP_SIZE = 384       # 可调整裁剪尺寸
    WORKERS = 16        # None表示使用全部CPU核心
    
    process_images_parallel(
        source_dir=SOURCE_DIR,
        target_dir=TARGET_DIR,
        crop_size=CROP_SIZE,
        workers=WORKERS
    )
    print("All images processed.")