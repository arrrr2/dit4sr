import os
import sys
from PIL import Image
import matplotlib.pyplot as plt

def get_image_sizes(root_dir):
    """获取所有图片尺寸（宽度和高度合并统计）"""
    sizes = []
    valid_ext = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
    
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(valid_ext):
                try:
                    with Image.open(os.path.join(root, file)) as img:
                        w, h = img.size
                        sizes.extend([w, h])
                except Exception as e:
                    print(f"Skipped {file}: {str(e)}")
    return sizes

def plot_simple_histogram(data, output_file):
    """生成简洁直方图"""
    plt.figure(figsize=(10, 6))
    
    # 基础直方图配置
    plt.hist(data, 
             bins=100,         # 固定分箱数量
             color='steelblue', 
             edgecolor='black')
    
    # 基础样式设置
    plt.title('Image Size Distribution', fontsize=14)
    plt.xlabel('Pixel Value (Width & Height)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # 自动优化布局
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()

def main():
    if len(sys.argv) != 2:
        print("Usage: python simple_hist.py <image_folder>")
        sys.exit(1)
    
    target_dir = sys.argv[1]
    if not os.path.isdir(target_dir):
        print(f"Error: {target_dir} is not a valid directory")
        sys.exit(1)
    
    print("Processing images...")
    sizes = get_image_sizes(target_dir)
    
    if not sizes:
        print("No images found")
        return
    
    output_path = os.path.join(os.getcwd(), "simple_size_histogram.png")
    plot_simple_histogram(sizes, output_path)
    print(f"Done! Result saved to: {output_path}")

if __name__ == "__main__":
    main()