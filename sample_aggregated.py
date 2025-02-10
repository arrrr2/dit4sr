import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.utils
from torchvision.utils import save_image
from diffusion import create_diffusion # Keep original create_diffusion import
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
from torch.utils.data import DataLoader
import argparse
import torchvision
from PIL import Image
from PIL import ImageOps
from tqdm import tqdm
import os
from sd3_impls import SD3LatentFormat
import numpy as np # Keep numpy import if gaussian_kernel is still used from diffusion.py

import torch.nn.functional as ff

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XXS/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
    out_path = args.out_path
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)


    @torch.no_grad()
    def pad_to_factor(image:torch.Tensor, factor=1):
        # 获取原始图片的宽度和高度
        w, h = image.shape[1], image.shape[2]

        # 计算需要填充到的宽度和高度
        
        new_w = (w + factor - 1) // factor * factor
        new_h = (h + factor - 1) // factor * factor

        
        # 如果需要填充
        if new_w != w or new_h != h:
            # 计算需要填充的宽度和高度
            pad_w = new_w - w
            pad_h = new_h - h
            # 使用反射填充
            new_image = torch.nn.functional.pad(image, (0, pad_h, 0, pad_w), mode='reflect')

            return new_image

        return image




    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, root_dir, transform=None):
            self.root_dir = root_dir
            self.transform = transform
            self.image_paths = []
            for root, _, files in os.walk(root_dir):
                for file in files:
                    if file.endswith(('png', 'jpg', 'jpeg', 'JPG')):
                        self.image_paths.append(os.path.join(root, file))

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert("RGB")
            size = image.size
            size = image.size
            if self.transform:
                image = self.transform(image)
            return {"img": image, "pat": img_path, "size":size}
            return {"img": image, "pat": img_path, "size":size}


    transform = torchvision.transforms.Compose([
        # torchvision.transforms.Lambda(lambda x: crop_devided_by(x, 16)),
        # torchvision.transforms.Lambda(lambda x: crop_or_pad_to_tar(x, 1024)),
        # torchvision.transforms.Lambda(lambda x: center_crop_image(x, args.image_size)),
        # torchvision.transforms.Resize(args.image_size, interpolation=torchvision.transforms.InterpolationMode.NEAREST), 
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: pad_to_factor(x, 4)),
        torchvision.transforms.Lambda(lambda x: x * 2 - 1)
    ])

    dataset = CustomDataset(args.in_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False) # Load checkpoint to specified device
    model.load_state_dict(state_dict['ema'])
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps)) # Use original create_diffusion
    vae = AutoencoderKL.from_pretrained(f"./vae", torch_dtype=torch.float16).to(device) # Load VAE in float16 and move to device
    t2p = torchvision.transforms.ToPILImage()

    pout = SD3LatentFormat().process_out
    pin = SD3LatentFormat().process_in

    # Labels to condition the model with (feel free to change):
    # Create sampling noise:
    for batch in tqdm(dataloader):

        # Sample images:
        with torch.no_grad():
            images = batch['img'].to(device)
            paths = batch['pat']
            # images = torch.nn.functional.interpolate(images, scale_factor=4, mode='nearest')
            images = images.to(torch.float16) # Convert images to float16 before encoding
            images = ff.interpolate(images, scale_factor=4)

            latents = vae.encode(images).latent_dist.mode()
            latents = pin(latents)
            latents = latents.to(torch.float32) # Keep latents in float32 for DiT
            z = torch.randn(len(images), 16, latents.shape[2], latents.shape[3], device=device, dtype=torch.float32) # Ensure z is float32 and on device
            model_kwargs = dict(y=latents)
            with torch.autocast(device_type='cuda', dtype=torch.float16): # Use autocast for potential speedup
                samples = diffusion.ddim_sample_loop_progressive_with_patch_aggregation( # Keep patch aggregation sampling
                    model, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device,
                    patch_size=latent_size, stride=latent_size // 2,  batch_size_patch=8# Keep patch aggregation parameters
                )
                for sample_dict in samples: # Iterate to get final sample
                    samples = sample_dict['sample']
                samples = pout(samples)
                samples = samples.to(torch.float16) # Convert samples to float16 before decoding
                samples = vae.decode(samples).sample

                samples = samples / 2  + 0.5
                samples = samples.clamp(0, 1)


        for i, sample in enumerate(samples):
            image_size = batch['size']
            sample = sample.cpu().float()
            sample = sample[:, :image_size[1] * 4, :image_size[0] * 4]
            img: Image = t2p(sample.cpu()) # Move sample to CPU before converting to PIL Image
            img.save(os.path.join(out_path, paths[i].split("/")[-1]))
            # print(paths[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XS/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512, 1024], default=256)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default="./results/000-DiT-XS-2/checkpoints/0200000.pt",
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--in-path", type=str, default="/home/ubuntu/data/repos/ResShift/testdata/Val_SR/lq")
    parser.add_argument("--out-path", type=str, default="/home/ubuntu/data/repos/ResShift/testdata/Val_SR/ditxs0")
    args = parser.parse_args()
    main(args)