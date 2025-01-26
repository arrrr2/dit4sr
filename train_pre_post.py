# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark=True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from basicsr.data.realesrgan_dataset import RealESRGANDataset
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
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
from torch.amp import autocast, GradScaler
from sd3_impls import SD3LatentFormat
import random
# torch.set_num_threads(1)
# torch.set_num_interop_threads(1)

scaler_pre = GradScaler("cuda", enabled=True)
scaler_post = GradScaler("cuda", enabled=True)
#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

t32 = lambda x: x.to(torch.float32)
t16 = lambda x: x.to(torch.float16)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")],
            force=True
        )
        logger = logging.getLogger(__name__)
        print(f"logger set, name={str(logger)}")
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger



def non_uniform_sampler(batch_size, T, t_delta, p, device):
    """
    Samples timesteps non-uniformly based on specified probabilities for significant and non-significant intervals.
    
    Parameters:
    - batch_size: Number of timesteps to sample.
    - T: Total number of timesteps.
    - t_delta: Timestep at which to split intervals.
    - p: Probability weight for the significant interval.
    - device: Device to use for tensor operations.
    
    Returns:
    - A tensor of sampled timesteps.
    """
    weight_first = t_delta * p
    weight_second = (T - t_delta) * (1 - p)
    total_weight = weight_first + weight_second
    prob_first = weight_first / total_weight if total_weight > 0 else 0.0
    prob_second = weight_second / total_weight if total_weight > 0 else 1.0
    
    choices = torch.rand(batch_size, device=device) < prob_first
    t_significant = torch.randint(0, t_delta, (batch_size,), device=device)
    t_non_significant = torch.randint(t_delta, T, (batch_size,), device=device)
    t = torch.where(choices, t_significant, t_non_significant)
    return t


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    print(str(args))
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        logger.info(str(args))
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model_pre = DiT_models[args.model](input_size=latent_size,)
    model_post = DiT_models[args.model](input_size=latent_size,)
    # Note that parameter initialization is done within the DiT constructor
    ema_pre = deepcopy(model_pre).to(device)  # Create an EMA of the model for use after training
    ema_post = deepcopy(model_post).to(device)
    requires_grad(ema_pre, False)
    requires_grad(ema_post, False)
    
    model_pre = DDP(model_pre.to(device), device_ids=[rank])
    model_post = DDP(model_post.to(device), device_ids=[rank])
    diffusion_pre = create_diffusion(timestep_respacing="", diffusion_steps=1000)  # default: 1000 steps, linear noise schedule
    diffusion_post = create_diffusion(timestep_respacing="", diffusion_steps=1000)  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"./vae", torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model_post.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt_pre = torch.optim.AdamW(model_pre.parameters(), lr=1e-4, weight_decay=0)
    warmup_scheduler_pre = LinearLR(opt_pre, start_factor=0.1, end_factor=1.0, total_iters=20000)
    annealing_scheduler_pre = CosineAnnealingLR(opt_pre, T_max=400000, eta_min=0.00001)
    sch_pre = SequentialLR(opt_pre, schedulers=[warmup_scheduler_pre, annealing_scheduler_pre], milestones=[20000])

    opt_post = torch.optim.AdamW(model_post.parameters(), lr=1e-4, weight_decay=0)
    warmup_scheduler_post = LinearLR(opt_post, start_factor=0.1, end_factor=1.0, total_iters=20000)
    annealing_scheduler_post = CosineAnnealingLR(opt_post, T_max=400000, eta_min=0.00001)
    sch_post = SequentialLR(opt_post, schedulers=[warmup_scheduler_post, annealing_scheduler_post], milestones=[20000])

    dataset_conf = OmegaConf.load(args.dataset)
    dataset = RealESRGANDataset(dataset_conf)
    
    if args.compile:
        model_pre = torch.compile(model_pre,  mode="max-autotune")
        model_post = torch.compile(model_post, mode="max-autotune")
        vae.encode = torch.compile(vae.encode, fullgraph=True, mode="max-autotune")

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )


    # Prepare models for training:
    update_ema(ema_pre, model_pre.module, decay=0)  # Ensure EMA is initialized with synced weights
    model_pre.train()  # important! This enables embedding dropout for classifier-free guidance
    ema_pre.eval()  # EMA model should always be in eval mode

    update_ema(ema_post, model_post.module, decay=0)  # Ensure EMA is initialized with synced weights
    model_post.train()  # important! This enables embedding dropout for classifier-free guidance
    ema_post.eval()  # EMA model should always be in eval mode

    pin = SD3LatentFormat().process_in
    pout = SD3LatentFormat().process_out


    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss_pre = 0
    running_loss_post = 0
    start_time = time()

    dataset.jpeger = dataset.jpeger.to(device)



    tok = time()
    logger.info(f"Training for {args.iters} iterss...")
    for epoch in range(100000000000):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for batch in loader:
            tik = time()

            # print(f"Time to load batch: {tik - tok}")

            with torch.autocast("cuda"): imgs = dataset.degrade_fun(batch['gt'].to(device, non_blocking=True), batch['kernel1'].to(device, non_blocking=True),\
                                        batch['kernel2'].to(device, non_blocking=True), batch['sinc_kernel'].to(device, non_blocking=True))
            x, y_post = imgs['gt'], imgs['lq']
            # x = x.to(device, non_blocking=True)
            # y = y.to(device, non_blocking=True)
            opt_pre.zero_grad(), opt_post.zero_grad()
            
            with torch.no_grad():
                # Map input images to latent space 
                x, y_post = t16(x), t16(y_post)
                y_pre = ff.interpolate(y_post, x.size(2), mode="nearest")
                if random.random() < 0.001:
                    saving =y_post[0]
                    saving = (saving * 0.5 + 0.5).clamp(0, 1)
                    saving = (saving * 255).to(torch.uint8).cpu().numpy().transpose(1, 2, 0)
                    saving = Image.fromarray(saving)
                    saving.save(f"test.png")
                if args.no_sample: x, y_post, y_pre = vae.encode(x).latent_dist.mode().clone(), vae.encode(y_pre).latent_dist.mode().clone(), 
                else: x, y_post, y_pre = vae.encode(x).latent_dist.sample().clone(), vae.encode(y_post).latent_dist.sample().clone(), vae.encode(y_pre).latent_dist.mode().clone()
                y_post = ff.interpolate(y_post, x.size(2), mode="nearest")
                x, y_post, y_pre = pin(x), pin(y_post), pin(y_pre)
                x, y_post, y_pre = t32(x), t32(y_post), t32(y_pre)
            with torch.autocast("cuda", enabled=True):
                
                # t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
                t = non_uniform_sampler(x.shape[0], diffusion_pre.num_timesteps, args.bernoulli_mid, args.bernoulli_p, device)
                
                model_kwargs_pre = dict(y=y_pre)
                model_kwargs_post = dict(y=y_post)


                loss_dict_pre = diffusion_pre.training_losses(model_pre, x, t, model_kwargs_pre)
                loss_dict_post = diffusion_post.training_losses(model_post, x, t, model_kwargs_post)

                loss_pre = loss_dict_pre["loss"].mean()
                loss_post = loss_dict_post["loss"].mean()

            # Backpropagate and update for model_pre
            scaler_pre.scale(loss_pre).backward()
            scaler_pre.step(opt_pre)
            scaler_pre.update()
            sch_pre.step()
            update_ema(ema_pre, model_pre.module)

            # Backpropagate and update for model_post
            scaler_post.scale(loss_post).backward()
            scaler_post.step(opt_post)
            scaler_post.update()
            sch_post.step()
            update_ema(ema_post, model_post.module)

            # Log loss values:
            running_loss_pre += loss_pre.item()
            running_loss_post += loss_post.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                # Calculate average loss for pre and post separately
                avg_loss_pre = torch.tensor(running_loss_pre / log_steps, device=device)
                avg_loss_post = torch.tensor(running_loss_post / log_steps, device=device)
                
                # Reduce the losses across all processes
                dist.all_reduce(avg_loss_pre, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_loss_post, op=dist.ReduceOp.SUM)
                
                # Calculate the final average loss for pre and post
                avg_loss_pre = avg_loss_pre.item() / dist.get_world_size()
                avg_loss_post = avg_loss_post.item() / dist.get_world_size()
                
                # Log the losses
                logger.info(f"(step={train_steps:07d}) Pre Train Loss: {avg_loss_pre:.4f}, Post Train Loss: {avg_loss_post:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                running_loss_pre = 0
                running_loss_post = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    # Save pre model checkpoint
                    checkpoint_pre = {
                        "model": model_pre.module.state_dict(),
                        "ema": ema_pre.state_dict(),
                        "opt": opt_pre.state_dict(),
                        "args": args
                    }
                    checkpoint_path_pre = f"{checkpoint_dir}/{train_steps:07d}_pre.pt"
                    torch.save(checkpoint_pre, checkpoint_path_pre)
                    logger.info(f"Saved pre checkpoint to {checkpoint_path_pre}")

                    # Save post model checkpoint
                    checkpoint_post = {
                        "model": model_post.module.state_dict(),
                        "ema": ema_post.state_dict(),
                        "opt": opt_post.state_dict(),
                        "args": args
                    }
                    checkpoint_path_post = f"{checkpoint_dir}/{train_steps:07d}_post.pt"
                    torch.save(checkpoint_post, checkpoint_path_post)
                    logger.info(f"Saved post checkpoint to {checkpoint_path_post}")
                dist.barrier()

            # print(f"Time to process batch: {tok - tik}")
        if train_steps >= args.iters:
            break


    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--dataset", type=str, default="baseline.yaml")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-S/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--iters", type=int, default=600_000)
    parser.add_argument("--compile", action="store_true", help="Enable compilation")
    parser.add_argument("--no_sample", action="store_true", help="not using sample mode")
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=12222)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=100_000)
    parser.add_argument("--bernoulli-mid", type=int, default=500)
    parser.add_argument("--bernoulli-p", type=float, default=0.5)
    args = parser.parse_args()
    main(args)
