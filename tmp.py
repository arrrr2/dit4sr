import time
import numpy as np
from PIL import Image
from io import BytesIO
import statistics
import cv2

def load_source_image(source_path):
    print(f"加载源图像：{source_path}...")
    return Image.open(source_path)

def save_compressed_images(img, levels=range(10)):
    print("生成不同压缩级别的PNG文件...")
    for level in levels:
        img.save(f'compressed_{level}.png', compress_level=level)

def load_images_to_memory(source_path, levels=range(10)):
    print("预加载图片数据到内存...")
    file_data = {}
    
    # 加载原始图像数据
    with open(source_path, 'rb') as f:
        file_data['original'] = f.read()
    
    # 加载压缩版本数据
    for level in levels:
        with open(f'compressed_{level}.png', 'rb') as f:
            file_data[level] = f.read()
    
    return file_data

def pil_reader(data):
    with Image.open(BytesIO(data)) as img:
        img.load()

def opencv_reader(data):
    np_array = np.frombuffer(data, dtype=np.uint8)
    cv2.imdecode(np_array, cv2.IMREAD_COLOR)

def run_profiling(file_data, reader_func, levels, iterations):
    results = {}
    
    for level in levels:
        data = file_data[level]
        times = []
        
        # 预热阶段（不记录时间）
        for _ in range(iterations):
            reader_func(data)
        
        # 正式测试
        for _ in range(iterations):
            start = time.perf_counter_ns()
            reader_func(data)
            end = time.perf_counter_ns()
            times.append((end - start) / 1e9)  # 转换为秒
            
        results[level] = {
            'avg': statistics.mean(times),
            'std': statistics.stdev(times),
            'min': min(times),
            'max': max(times)
        }
    
    return results

def perform_comparison(file_data, levels, iterations=50):
    print("开始对比测试...")
    return {
        'PIL': run_profiling(file_data, pil_reader, levels, iterations),
        'OpenCV': run_profiling(file_data, opencv_reader, levels, iterations)
    }

def print_comparison(results):
    print("\n对比测试结果（单位：秒）：")
    headers = f"{'压缩级别':<12} | {'库名称':<8} | {'平均时间':<10} | {'标准差':<10} | {'最小值':<10} | {'最大值':<10}"
    print(headers)
    print("-" * 75)
    
    # 排序键：original优先，数字按大小排序
    sorted_keys = sorted(results['PIL'].keys(), 
                        key=lambda x: (x != 'original', x if isinstance(x, int) else 0))
    
    for level in sorted_keys:
        for lib in ['PIL', 'OpenCV']:
            res = results[lib][level]
            level_str = str(level).ljust(12) if isinstance(level, int) else level.ljust(12)
            print(f"{level_str} | {lib:<8} | {res['avg']:.6f} | {res['std']:.6f} | {res['min']:.6f} | {res['max']:.6f}")

if __name__ == "__main__":
    source_image_path = "/home/ubuntu/ssd/imgnet_gen_selected/n02115641_8319_5.png"  # 需要替换为实际文件路径
    
    # 初始化测试环境
    img = load_source_image(source_image_path)
    save_compressed_images(img)
    file_data = load_images_to_memory(source_image_path)
    
    # 执行对比测试（包含原始图像）
    comparison_results = perform_comparison(
        file_data, 
        levels=['original'] + list(range(10)),
        iterations=50
    )
    print_comparison(comparison_results)

    # 清理生成的测试文件（可选）
    # import os
    # for level in range(10):
    #     os.remove(f'compressed_{level}.png')