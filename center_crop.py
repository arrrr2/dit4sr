import os
import argparse
from PIL import Image

def center_crop_image(input_folder, output_folder, target_size):
    """
    Resize and center crop images to a square of target_size x target_size pixels.

    Args:
        input_folder (str): Path to the input folder containing images.
        output_folder (str): Path to the output folder to save processed images.
        target_size (int): Target size for the shorter edge and cropped square.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, _, files in os.walk(input_folder):
        relative_path = os.path.relpath(root, input_folder)
        target_dir = os.path.join(output_folder, relative_path)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                input_file_path = os.path.join(root, file)
                output_file_path = os.path.join(target_dir, os.path.splitext(file)[0] + ".png")

                try:
                    with Image.open(input_file_path) as img:
                        # Resize image to maintain aspect ratio with shorter side = target_size
                        width, height = img.size
                        if width < height:
                            new_width = target_size
                            new_height = int(target_size * height / width)
                        else:
                            new_height = target_size
                            new_width = int(target_size * width / height)

                        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)

                        # Center crop to target_size x target_size
                        left = (new_width - target_size) / 2
                        top = (new_height - target_size) / 2
                        right = left + target_size
                        bottom = top + target_size

                        img = img.crop((left, top, right, bottom))

                        # Save to output folder in webp format with quality 99
                        img.save(output_file_path)
                        print(f"Processed: {output_file_path}")

                except Exception as e:
                    print(f"Failed to process {input_file_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Center crop and resize images.")
    parser.add_argument("input_folder", type=str, help="Path to the input folder containing images.")
    parser.add_argument("output_folder", type=str, help="Path to the output folder to save processed images.")
    parser.add_argument("target_size", type=int, help="Target size for the shorter edge and cropped square.")

    args = parser.parse_args()
    center_crop_image(args.input_folder, args.output_folder, args.target_size)

if __name__ == "__main__":
    main()
