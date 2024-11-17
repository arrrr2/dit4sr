# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from basicsr.data.realesrgan_dataset import RealESRGANDataset
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import torch.nn.functional as ff
from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from omegaconf import OmegaConf


#################################################################################
#                             Training Helper Functions                         #
#################################################################################


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    # dist.init_process_group("nccl")
    # assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    # rank = dist.get_rank()
    # device = rank % torch.cuda.device_count()
    # seed = args.global_seed * dist.get_world_size() + rank
    # torch.manual_seed(seed)
    # torch.cuda.set_device(device)
    # print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    device = "cuda:2"



    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."


    vae = AutoencoderKL.from_pretrained(f"./vae").to(device)


    dataset_conf = OmegaConf.load("./baseline.yaml")
    dataset = RealESRGANDataset(dataset_conf)
 

    # sampler = DistributedSampler(
    #     dataset,
    #     num_replicas=dist.get_world_size(),
    #     rank=rank,
    #     shuffle=True,
    #     seed=args.global_seed
    # )
    batch_size = int(args.global_batch_size)
    loader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle=False,
        # sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    for epoch in range(args.epochs):
        # sampler.set_epoch(epoch)

        for x, y in loader:
            x = x.to(device) 
            y = y.to(device)
            with torch.autocast("cuda"):
                y = ff.interpolate(y, x.size(2), mode="nearest")
                with torch.no_grad():
                    # Map input images to latent space 
                    x = vae.encode(x).latent_dist.mode()
                    y = vae.encode(y).latent_dist.sample()

                    # resample to image for visual
                    x = vae.decode(x).sample
                    y = vae.decode(y).sample

                    # generate PIL images with a 8*8 grid (now 64 images)

                    grid_x = torch.cat([torch.cat([x[j * 8 + i] for i in range(8)], dim=1)  for j in range(8)], dim=2)
                    grid_y = torch.cat([torch.cat([y[j * 8 + i] for i in range(8)], dim=1)  for j in range(8)], dim=2)
                    grid = torch.cat([grid_x, grid_y], dim=1)
                    grid = grid / 2 + 0.5
                    grid = (grid * 255).round().clamp(0, 255).permute(1, 2, 0).cpu().numpy()
                    grid = grid.astype(np.uint8)
                    grid_image = Image.fromarray(grid)
                    grid_image.save(f"epoch_{epoch}.png")




 


            log_steps += 1
            # 收集 x, y 到主进程以便于处理。




    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="latents")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--global-batch-size", type=int, default=64)

    args = parser.parse_args()
    main(args)
