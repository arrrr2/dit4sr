import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def sliding_window_crop(image: Image.Image, stride, overlap) -> list[tuple[Image.Image, int, int]]:
    """
    使用滑动窗口裁剪图像。

    :param image: PIL.Image对象
    :param window_size: 窗口大小（像素）
    :param stride: 步幅（像素）
    :param min_overlap: 最小重叠大小（像素）
    :return: 裁剪后的图像列表
    """
    width, height = image.size
    crops = []

    min_overlap = overlap * 2
    
    # 横向裁剪
    y = 0
    while y + stride <= height:
        x = 0
        while x + stride <= width:
            box = (x, y, x + stride, y + stride)
            crop = image.crop(box)
            crops.append((crop, x, y))
            x += stride
            # x -= overlap
        
        # 处理右边缘
        remaining_x = width - x
        if remaining_x > min_overlap:
            box = (x, y, width, y + stride)
            crop = image.crop(box)
            crops.append((crop, x, y))
        y += stride
        # y -= overlap
    
    # 处理下边缘
    remaining_y = height - y
    if remaining_y > overlap:
        x = 0
        while x + stride <= width:
            box = (x, y, x + stride, height)
            crop = image.crop(box)
            crops.append((crop, x, y))
            x += stride
            # x -= overlap
        
        # 处理右下角
        remaining_x = width - (x)
        if remaining_x > min_overlap:
            box = (x, y, width, height)
            crop = image.crop(box)
            crops.append((crop, x, y))
    
    return crops

def process_single_image(image_path, input_dir, output_dir, window_size, overlap):
    """
    处理单张图片：进行滑动窗口裁剪并保存裁剪后的图片。

    :param image_path: 图片路径
    :param input_dir: 输入目录路径
    :param output_dir: 输出目录路径
    :param window_size: 窗口大小（像素）
    :param overlap: 重叠大小（像素）
    :return: 成功处理的标志（True/False）
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            
            # 跳过小于窗口大小的图片
            if width < window_size or height < window_size:
                return False
            
            stride = window_size
            
            # 获取裁剪后的图片
            crops = sliding_window_crop(img, stride, overlap)
            
            if not crops:
                return False
            
            # # 计算相对路径以保留子文件夹结构
            # relative_path = image_path.relative_to(input_dir)
            # relative_dir = relative_path.parent
            
            # # 创建对应的输出子目录
            # save_dir = output_dir / relative_dir
            # save_dir.mkdir(parents=True, exist_ok=True)
            
            # # 保存裁剪后的图片
            # base_name = image_path.stem
            # ext = image_path.suffix
            # for crop, x, y in crops:
            #     crop_name = f"{base_name}_x{x}_y{y}{ext}"
            #     crop_path = save_dir / crop_name
            #     crop.save(crop_path, compress_level=0)
        
        return True
    except Exception as e:
        print(f"处理 {image_path} 时出错: {e}")
        return False

def process_images(input_dir, output_dir, window_size=512, overlap=128, num_processes=32):
    """
    处理输入目录中的所有图片，进行滑动窗口裁剪，并保存裁剪后的图片。

    :param input_dir: 输入目录路径
    :param output_dir: 输出目录路径
    :param window_size: 窗口大小（像素）
    :param overlap: 重叠大小（像素）
    :param num_processes: 使用的进程数
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # 支持的图片格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    # 递归获取所有图片路径
    image_paths = [p for p in input_dir.rglob('*') if p.suffix.lower() in image_extensions]
    
    total_images = len(image_paths)
    if total_images == 0:
        print("未找到符合条件的图片。")
        return
    
    # 使用ProcessPoolExecutor进行多进程处理
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        for image_path in image_paths:
            futures.append(executor.submit(process_single_image, image_path, input_dir, output_dir, window_size, overlap))
        
        # 使用tqdm显示进度条
        for _ in tqdm(as_completed(futures), total=total_images, desc='正在处理图片'):
            pass

if __name__ == "__main__":
    import argparse

    # 设置命令行参数
    parser = argparse.ArgumentParser(description='使用滑动窗口裁剪图片并保留文件夹结构。')
    parser.add_argument('input_dir', type=str, help='输入目录路径')
    parser.add_argument('output_dir', type=str, help='输出目录路径')
    parser.add_argument('--window_size', type=int, default=512, help='裁剪窗口大小（默认: 512）')
    parser.add_argument('--overlap', type=int, default=128, help='窗口重叠大小（默认: 128）')
    parser.add_argument('--num_processes', type=int, default=128, help='使用的进程数（默认: 4）')
    
    args = parser.parse_args()
    
    # 调用处理函数
    process_images(args.input_dir, args.output_dir, args.window_size, args.overlap, args.num_processes)