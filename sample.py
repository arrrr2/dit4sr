# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.utils
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
from torch.utils.data import DataLoader
import argparse
import torchvision
from PIL import Image
import os




def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XXS/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
    out_path = args.out_path


    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, root_dir, transform=None):
            self.root_dir = root_dir
            self.transform = transform
            self.image_paths = []
            for root, _, files in os.walk(root_dir):
                for file in files:
                    if file.endswith(('png', 'jpg', 'jpeg')):
                        self.image_paths.append(os.path.join(root, file))

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return {"img": image, "pat": img_path}

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x * 2 - 1)
    ])

    dataset = CustomDataset(args.in_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict['model'])
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"./vae").to(device)
    t2p = torchvision.transforms.ToPILImage()

    # Labels to condition the model with (feel free to change):
    # Create sampling noise:
    for batch in dataloader:
        images = batch['img'].to(device)
        paths = batch['pat']
        images = torch.nn.functional.interpolate(images, scale_factor=4, mode='nearest')
        latents = vae.encode(images).latent_dist.sample()

        z = torch.randn(len(images), 16, latent_size, latent_size, device=device)
        model_kwargs = dict(y=latents)
        # Sample images:
        with torch.no_grad():
            samples = diffusion.p_sample_loop(
                model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
            )
            samples = vae.decode(samples).sample / 2 + 0.5
            samples = samples.clamp(0, 1)

        for i, sample in enumerate(samples):
            img: Image = t2p(sample)
            img.save(os.path.join(out_path, paths[i].split("/")[-1]))
            print(paths[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XXS/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default="./results/000-DiT-XXS-2/checkpoints/0300000.pt",
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--in-path", type=str, default="/home/ubuntu/data/repos/ResShift/testdata/Val_SR/lq")
    parser.add_argument("--out-path", type=str, default="/home/ubuntu/data/repos/ResShift/testdata/Val_SR/dit300k")
    args = parser.parse_args()
    main(args)
