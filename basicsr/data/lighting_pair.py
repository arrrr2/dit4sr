import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
from pathlib import Path

import albumentations

import torch.nn.functional as F
from torch.utils import data as data
from omegaconf import OmegaConf

from basicsr.utils import DiffJPEG
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt

def readline_txt(txt_file):
    txt_file = [txt_file, ] if isinstance(txt_file, str) else txt_file
    out = []
    for txt_file_current in txt_file:
        with open(txt_file_current, 'r') as ff:
            out.extend([x[:-1] for x in ff.readlines()])

    return out

@DATASET_REGISTRY.register(suffix='basicsr')
class LightingPairataset(data.Dataset):
    """Dataset used for Real-ESRGAN model:
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It loads gt (Ground-Truth) images, and augments them.
    It also generates blur kernels and sinc kernels for generating low-quality images.
    Note that the low-quality images are processed in tensors on GPUS for faster processing.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
            Please see more options in the codes.
    """

    def __init__(self, conf, mode='training'):
        super(LightingPairataset, self).__init__()
        conf = conf.data.train.params
        self.opt = conf
        self.file_client = None
        self.io_backend_opt = conf['io_backend']


        
        self.gt_paths = []
        if 'gt_dir_paths' in conf:
            for current_dir in conf['gt_dir_paths']:
                for current_ext in conf['im_exts']:
                    self.gt_paths.extend(sorted([str(x) for x in Path(current_dir).glob(f'**/*.{current_ext}')]))
        if 'gt_txt_file_path' in conf:
            for current_txt in conf['gt_txt_file_path']:
                self.gt_paths.extend(readline_txt(current_txt))
        if 'gt_length' in conf:
            self.gt_paths = random.sample(self.gt_paths, conf['gt_length'])

        print(len(self.gt_paths))
        print(self.gt_paths[:10])

    

        self.lq_paths = []
        if 'lq_dir_paths' in conf:
            for current_dir in conf['lq_dir_paths']:
                for current_ext in conf['im_exts']:
                    self.lq_paths.extend(sorted([str(x) for x in Path(current_dir).glob(f'**/*.{current_ext}')]))
        if 'lq_txt_file_path' in conf:
            for current_txt in conf['lq_txt_file_path']:
                self.lq_paths.extend(readline_txt(current_txt))
        if 'lq_length' in conf:
            self.lq_paths = random.sample(self.lq_paths, conf['lq_length'])

        self.gt_size = conf['gt_size']
        self.mode = mode
        self.conditional = conf['cond'] 

        print(conf)



    @torch.no_grad()
    def __getitem__(self, index):
        self.gt_size = self.opt['gt_size']

        gt_size = self.gt_size

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.gt_paths[index]
        # avoid errors caused by high latency in reading files
        retry = 100000
        while retry > 0:
            try:
                img_bytes = self.file_client.get(gt_path, 'gt')
                img_gt = imfrombytes(img_bytes, float32=True)
                if self.conditional: 
                    img_lq = imfrombytes(self.file_client.get(self.lq_paths[index], 'lq'), float32=True)
                    print(f"lq: {self.lq_paths[index]}")
                    print(f"gt: {gt_path}")
                else: img_lq = np.random.rand(*img_gt.shape).astype(np.float32)
                if img_gt.shape[0] < gt_size or img_gt.shape[1] < gt_size:
                    raise ValueError(f'GT image is too small: {gt_path}, {img_gt.shape}')
            # except (IOError, OSError, AttributeError) as e:
            except:
                # logger = get_root_logger()
                # logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                index = random.randint(0, self.__len__())
                gt_path = self.gt_paths[index]
                time.sleep(0.001)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1

        if self.mode == 'testing':
            if not hasattr(self, 'test_aug'):
                self.test_aug = albumentations.Compose([
                    albumentations.SmallestMaxSize(max_size=gt_size),
                    albumentations.CenterCrop(gt_size, gt_size),
                    ])
            img_gt = self.test_aug(image=img_gt)['image']
        elif self.mode == 'training':
            pass
        else:
            raise ValueError(f'Unexpected value {self.mode} for mode parameter')

        if self.mode == 'training':
            # print("got one")
            crop_pad_size = self.opt['crop_pad_size']

            h, w = img_gt.shape[0:2]
            # 1st: crop to not larger than crop_pad_size
            if img_gt.shape[0] > crop_pad_size and img_gt.shape[1] > crop_pad_size:
                h, w = img_gt.shape[0:2]
                # randomly choose top and left coordinates
                top = random.randint(0, h - crop_pad_size)
                left = random.randint(0, w - crop_pad_size)
                top, left = max(0, top), max(0, left)
                img_gt = img_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...]
                img_lq = img_lq[top:top + crop_pad_size, left:left + crop_pad_size, ...]
            # print("1st finished")

            # 2nd: pad to target
            h, w = img_gt.shape[0:2]
            while h < crop_pad_size or w < crop_pad_size:
                pad_h = min(max(0, crop_pad_size - h), h)
                pad_w = min(max(0, crop_pad_size - w), w)
                img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
                img_lq = cv2.copyMakeBorder(img_lq, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
                h, w = img_gt.shape[0:2]
            # print("2nd finished")

            # 3rd: choose a value between gt_size and crop_pad size, and select a region of this size to randomly crop to this value
            if crop_pad_size > gt_size:
                rnd_h = random.randint(0, crop_pad_size - gt_size)
                rnd_w = random.randint(0, crop_pad_size - gt_size)
                img_gt = img_gt[rnd_h:rnd_h + gt_size, rnd_w:rnd_w + gt_size, ...]
                img_lq = img_lq[rnd_h:rnd_h + gt_size, rnd_w:rnd_w + gt_size, ...] 
            # print("3rd finished")

            # 4th: if crop_pad is larger than gt_size, resize (antialiasing)
            if crop_pad_size > gt_size:
                img_gt = cv2.resize(img_gt, (gt_size, gt_size), interpolation=cv2.INTER_AREA)
                img_lq = cv2.resize(img_lq, (gt_size, gt_size), interpolation=cv2.INTER_AREA)
            # print("4th finished")


        elif self.mode == 'testing':
            pass
        else:
            raise ValueError(f'Unexpected value {self.mode} for mode parameter')

        # hwc - chw, numpy to torch
        if not self.conditional: img_lq = self.fun(img_gt) 



        gt = torch.from_numpy(np.ascontiguousarray(np.transpose(img_gt, (2, 0, 1)))).float()
        lq = torch.from_numpy(np.ascontiguousarray(np.transpose(img_lq, (2, 0, 1)))).float()

        # print("got one")

        # gt, lq = gt  * 2.0 - 1.0, lq * 2.0 - 1.0
        return gt, lq

    def __len__(self):
        return len(self.gt_paths)
    


    def fun(self, x):
        """
        综合方法对图像进行模拟错误曝光处理，更平衡地模拟欠曝和过曝。
        假设x为HWC格式的[0,1]范围的RGB图像(np.float32)。
        """
        # 拷贝图像以进行调整
        x_adjusted = x.copy()

        # 1. 全局曝光调整 (更平衡的范围)
        if random.random() < 0.5:
            exposure_factor = random.uniform(0.2, 3.0)  # 扩展范围，包含更强的过曝
            x_adjusted = np.clip(x_adjusted * exposure_factor, 0, 1)

        # 2. 局部曝光调整（随机局部mask，包含更强的过曝）
        if random.random() < 0.4:
            h, w, _ = x_adjusted.shape
            mask = np.random.uniform(0.5, 2.0, size=(h, w, 1))  # 允许局部过曝更强
            mask = np.repeat(mask, 3, axis=2)
            mask = cv2.GaussianBlur(mask, (51, 51), 0)
            x_adjusted = np.clip(x_adjusted * mask, 0, 1)

        # 3. Gamma校正 (更广的范围)
        if random.random() < 0.5:
            gamma = random.uniform(0.3, 2.5)  # 更广的gamma范围
            x_adjusted = np.clip(x_adjusted ** gamma, 0, 1)

        # 4. 高光剪裁与阴影剪裁
        if random.random() < 0.4:
            # 高光剪裁 (更低的阈值模拟更明显的过曝)
            highlight_threshold = random.uniform(0.5, 0.95)
            x_adjusted[x_adjusted > highlight_threshold] = 1.0

        if random.random() < 0.4:
            # 阴影剪裁
            shadow_threshold = random.uniform(0.05, 0.4) # 稍微调整范围
            x_adjusted[x_adjusted < shadow_threshold] = 0.0

        # 5. 添加随机噪声 (稍微提高幅度)
        if random.random() < 0.3: # 稍微提高概率
            noise_type = random.choice(["gaussian", "salt_pepper"])
            if noise_type == "gaussian":
                noise = np.random.normal(0, 0.03, x_adjusted.shape).astype(np.float32) # 稍微提高噪声幅度
                x_adjusted = np.clip(x_adjusted + noise, 0, 1)
            else:
                prob = random.uniform(0.01, 0.05) # 稍微提高椒盐噪声概率
                salt_pepper = np.random.choice([0, 1], size=x_adjusted.shape, p=[1 - prob, prob])
                x_adjusted = np.clip(x_adjusted + salt_pepper * random.uniform(-0.3, 0.3), 0, 1) # 扩展亮度变化范围

        # 6. 对比度调整 (更广的范围)
        if random.random() < 0.4:
            alpha = random.uniform(0.3, 2.0)  # 更广的对比度调整范围
            x_adjusted = np.clip((x_adjusted - 0.5) * alpha + 0.5, 0, 1)

        # 7. 模拟过曝的白色漂白效果
        if random.random() < 0.2:
            white_level = random.uniform(0.8, 1.2) # 允许超过1.0来模拟漂白
            x_adjusted = np.clip(x_adjusted * white_level, 0, 1)

        # 8. 模拟欠曝的黑色拉伸效果
        if random.random() < 0.2:
            black_level = random.uniform(0.0, 0.2)
            x_adjusted = np.clip(x_adjusted - black_level, 0, 1)

        # 9. 随机保存原图和调整后图像，便于调试
        if random.random() < 0.001:
            cv2.imwrite('example_original.jpg', cv2.cvtColor((x * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            cv2.imwrite('example_adjusted.jpg', cv2.cvtColor((x_adjusted * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

        return x_adjusted