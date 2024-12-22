from PIL import Image
import time
import numpy as np
png_1 = "/home/ubuntu/data/datasets/lsdir/train/hr/0004000/0003987.png"


# test decode time
for _ in range(10):
    img = Image.open(png_1)
    img = img.convert("RGB")
    img = np.array(img)
    # img = img.resize((256, 256), Image.BICUBIC)
    # img = img.convert("RGB")
start = time.time()
for _ in range(1000):
    img = Image.open(png_1)
    img = img.convert("RGB")
    img = np.array(img)
    # img = img.resize((256, 256), Image.BICUBIC)
    # img = img.convert("RGB")
end = time.time()

print("Time: ", end - start)




img = Image.open(png_1)
img.save("compress_level_0.png", format="PNG", compress_level=0)



png_2 = "compress_level_0.png"
for _ in range(10):
    img = Image.open(png_2)
    img = img.convert("RGB")
    img = np.array(img)
    # img = img.resize((256, 256), Image.BICUBIC)
    # img = img.convert("RGB")

start = time.time()
for _ in range(1000):
    img = Image.open(png_2)
    img = img.convert("RGB")
    img = np.array(img)
    # img = img.resize((256, 256), Image.BICUBIC)
    # img = img.convert("RGB")
end = time.time()

print("Time: ", end - start)


