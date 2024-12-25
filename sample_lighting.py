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
from PIL import ImageChops
import os
from sd3_impls import SD3LatentFormat



def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XXS/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
    out_path = args.out_path
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)

    def crop_divided_by(image, divisor):
        w, h = image.size
        w = w // divisor * divisor
        h = h // divisor * divisor
        image = image.crop((0, 0, w, h))
        return image

    def crop_or_pad_to_tar(image, target_size=1024):

        w, h = image.size

        # Calculate cropping coordinates if necessary
        if w > target_size or h > target_size:
            left = (w - target_size) // 2 if w > target_size else 0
            top = (h - target_size) // 2 if h > target_size else 0
            right = left + target_size if w > target_size else w
            bottom = top + target_size if h > target_size else h
            image = image.crop((left, top, right, bottom))

        # Calculate padding if necessary
        if w < target_size or h < target_size:
            new_image = Image.new(image.mode, (target_size, target_size), (0, 0, 0)) # Default to black background
            # Calculate offset to center the original image
            offset = ((target_size - w) // 2, (target_size - h) // 2)
            new_image.paste(image, offset)
            image = new_image

        return image
    
    def center_crop_image(image, target_size):

        w, h = image.size

        # Resize the image so the shorter side matches the target size
        if w < h:
            new_h = int(target_size * h / w)
            resized_image = image.resize((target_size, new_h), Image.Resampling.BICUBIC)
        else:
            new_w = int(target_size * w / h)
            resized_image = image.resize((new_w, target_size), Image.Resampling.BICUBIC)

        # Perform center crop
        rw, rh = resized_image.size
        left = (rw - target_size) // 2
        top = (rh - target_size) // 2
        right = left + target_size
        bottom = top + target_size
        cropped_image = resized_image.crop((left, top, right, bottom))

        return cropped_image


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
            shape = image.size

            # pad image with reflection padding with 128 on right and bottom
            padded_image = torchvision.transforms.functional.pad(image, padding=(512, 512, 512, 512), padding_mode='reflect')
            if self.transform:
                image = self.transform(padded_image)
            return {"img": image, "pat": img_path, "ori_shape": shape}
        
    

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda x: crop_divided_by(x, 8)),
        # torchvision.transforms.Lambda(lambda x: crop_or_pad_to_tar(x, 1024)),
        # torchvision.transforms.Lambda(lambda x: center_crop_image(x, args.image_size)),
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Lambda(lambda x: x * 2 - 1)
    ])

    dataset = CustomDataset(args.in_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

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
    vae = AutoencoderKL.from_pretrained(f"./vae", torch_dtype=torch.float16).to(device)
    t2p = torchvision.transforms.ToPILImage()

    pout = SD3LatentFormat().process_out
    pin = SD3LatentFormat().process_in

    # Labels to condition the model with (feel free to change):
    # Create sampling noise:
    for batch in dataloader:
        images = batch['img'].to(device)
        paths = batch['pat']
        ori_shape = batch['ori_shape']


        # model_kwargs = dict(y=latents) # 不再需要预先初始化一个和整张图一样大小的噪声，因为已经针对patch进行处理
        # Sample images:
        with torch.no_grad():   
            # vae = vae.to(device)
            images = images.to(torch.float16)
            latents = vae.encode(images).latent_dist.mode()
            latents = pin(latents)
            latents = latents.to(torch.float32).to(device)
            # vae = vae.to("cpu")
            model = model.to(device)
            # print(device)

            with torch.autocast("cuda"):
                
                model_kwargs = dict(y=latents) # model_kwargs 应当在 with torch.no_grad() 内定义
                samples = diffusion.ddim_sample_loop_with_patch_aggregation( # 或 ddim_sample_loop_with_patch_aggregation
                    model.forward, latents.shape, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device,
                    patch_size=latent_size, stride=latent_size // 2, sigma=8.0
                )
                samples = pout(samples)
                samples = samples.to(torch.float16)
                samples = vae.decode(samples).sample
                samples = samples.clamp(0, 1)

        for i, sample in enumerate(samples):
            w, h = ori_shape[0], ori_shape[1]
            w, h = h.item(), w.item()
            img: Image.Image = t2p(sample[:, 512:w+512, 512:h+512])
            
            img.save(os.path.join(out_path, paths[i].split("/")[-1]))
            print(paths[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XXS/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512, 1024], default=256)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default="./results/000-DiT-XXS-2/checkpoints/0300000.pt",
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--in-path", type=str, default="/home/ubuntu/data/repos/ResShift/testdata/Val_SR/lq")
    parser.add_argument("--out-path", type=str, default="/home/ubuntu/data/repos/ResShift/testdata/Val_SR/dit300k")
    args = parser.parse_args()
    main(args)