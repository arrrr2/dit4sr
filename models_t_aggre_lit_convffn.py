# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.vision_transformer import  Mlp
from lit import LinearAttention, dwc_ffn
from typing import Union, Tuple

from torch.utils.checkpoint import checkpoint

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            patch_size: Union[int, Tuple[int, int]] = 16,
            stride: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            flatten: bool = True,
            bias: bool = True,
    ):
        super().__init__()
        self.patch_size = (patch_size,) * 2 if isinstance(patch_size, int) else patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, bias=bias)
        self.flatten = flatten



    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        return x



def modulate(x, shift, scale):
    return x * (1 + scale) + shift


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = LinearAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        # self.attn = GeneralizedLinearAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # approx_gelu = lambda: nn.GELU(approximate="tanh")
        # self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        self.mlp = dwc_ffn(hidden_size, mlp_ratio=mlp_ratio, num_heads=num_heads)

        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size ** 0.5)
        # self.adaLN_modulation = nn.Sequential(
        #     nn.SiLU(),
        #     nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        # )

    def forward(self, x, c, image_shape):
        B, N, C = x.shape
        # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = c.chunk(6, dim=1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + c.reshape(B, 6, -1)).chunk(6, dim=1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), image_shape)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class ConvBlock(nn.Module):
    def __init__(self, hidden_size, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.LeakyReLU()
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x, input_shape):
        x = self.norm(x)
        x = self.unpatchify(x, input_shape)
        x = self.conv(x)
        x = self.patchify(x)
        x = self.relu(x)
        
        return x
    
    def unpatchify(self, x, input_shape):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = x.shape[-1]
        h, w = input_shape
        x = x.reshape(shape=(x.shape[0], h, w, c))
        x = torch.einsum('nhwc->nchw', x)

        return x

    def patchify(self, x):
        """
        imgs: (N, C, H, W)
        x: (N, T, patch_size**2 * C)
        """
        n, c, h, w = x.shape
        x = torch.einsum('nchw->nhwc', x)
        x = x.reshape(shape=(n, h * w, c))
        return x   

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, kernel_size=3):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.last_conv = nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, padding=kernel_size // 2)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c, HW=None):
        if HW is None:
            h = w = int(x.shape[1] ** 0.5)
        else: 
            h, w = HW

        
        shift, scale = self.adaLN_modulation(c).unsqueeze(1).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)

        x = x.reshape(x.shape[0], h, w, -1).permute(0, 3, 1, 2)
        x = self.last_conv(x)
        x = x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[1])
        x = self.linear(x)
        return x

class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=32,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=True,
        grad_checkpoint=False,
        **kwargs
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels  if learn_sigma else in_channels // 2
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_checkpoint = grad_checkpoint

        self.x_embedder = PatchEmbed(patch_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)


        self.ditblocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )
        # self.convblocks = nn.ModuleList(
        #     [ConvBlock(hidden_size, 3, 1, 1) for _ in range(depth)]
        # )

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )


        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

        # Zero-out adaLN modulation layers in DiT blocks:
        # for block in self.blocks:
        #     nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
        #     nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        # nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        # nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, hw=None):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        if hw is None: h = w = int(x.shape[1] ** 0.5)
        else: h, w = hw
        assert h * w == x.shape[1], f"h={h}, w={w}, x.shape[1]={x.shape[1]}"

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs 

    def patchify(self, imgs):
        """
        imgs: (N, C, H, W)
        x: (N, T, patch_size**2 * C)
        """
        p = self.x_embedder.patch_size[0]
        c = self.out_channels
        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0, "Image dimensions must be divisible by the patch size."
        
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p * p * c))
        return x    


    def forward(self, x, t, **kwargs):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        lq = kwargs['y']
        x = torch.cat([x, lq], dim=1)
        image_shape = (x.shape[2] // self.patch_size, x.shape[3] // self.patch_size)
        # x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        x = self.x_embedder(x)
        t = self.t_embedder(t)                   # (N, D)
        # y = self.y_embedder(y, self.training)    # (N, D)
        # c = t + y  
        c = self.adaLN_modulation(t)
        for _ in range(len(self.ditblocks)):
            if self.use_checkpoint:
                x = checkpoint(self.ditblocks[_], x, c, image_shape)
            else:
                x = self.ditblocks[_](x, c, image_shape)
            # x = self.convblocks[_](x, image_shape)  
        x = self.final_layer(x, t)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x, image_shape)                   # (N, out_channels, H, W)
        return x




#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_M_2(**kwargs):
    return DiT(depth=24, hidden_size=384, patch_size=2, num_heads=8, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)

def DiT_S_1(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=1, num_heads=6, **kwargs)

def DiT_XS_2_D(**kwargs):
    return DiT(depth=32, hidden_size=128, patch_size=2, num_heads=2, **kwargs)

def DiT_XS_2(**kwargs):
    return DiT(depth=8, hidden_size=256, patch_size=2, num_heads=4, **kwargs)

def DiT_XS_1(**kwargs):
    return DiT(depth=8, hidden_size=256, patch_size=1, num_heads=4, **kwargs)

def DiT_XS_4(**kwargs):
    return DiT(depth=8, hidden_size=256, patch_size=4, num_heads=4, **kwargs)

def DiT_XS_8(**kwargs):
    return DiT(depth=8, hidden_size=256, patch_size=8, num_heads=4, **kwargs)

def DiT_XXS_2(**kwargs):
    return DiT(depth=6, hidden_size=256, patch_size=2, num_heads=4, **kwargs)

def DiT_XXS_1(**kwargs):
    return DiT(depth=6, hidden_size=256, patch_size=1, num_heads=4, **kwargs)

DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8, 
    'DiT-XS/2-D': DiT_XS_2_D,
    'DiT-XS/2': DiT_XS_2, 'DiT-XS/4': DiT_XS_4, 'DiT-XS/8': DiT_XS_8, 'DiT-XS/1': DiT_XS_1,
    'DiT-XXS/2': DiT_XXS_2,  'DiT-XXS/1': DiT_XXS_1, 'DiT-M/2': DiT_M_2,
}

if __name__=="__main__":

    device = "cuda:3"
    model = DiT_S_1(input_size=64)
    from torch.utils.flop_counter import FlopCounterMode
    

    input = torch.randn(1, 16, 240, 135)
    y = torch.randn(1, 16, 240, 135)
    t = torch.randn(1)

    print(model)
    with FlopCounterMode(depth=4):
        model(input, t, y=y)
    
    # 计算参数量
    def count_parameters(model):
        total_params = 0
        for param in model.parameters():
            if param.requires_grad:  # 只计算可训练的参数
                total_params += param.numel()  # 累加每个参数的参数量
        return total_params

    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params}")

    module_param_counts = []
    total_trainable_params = 0

    for name, module in model.named_modules():
        # Skip the top-level module (model itself)
        if name == '':
            continue

        trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total_trainable_params += trainable_params

        module_param_counts.append({"Module": name, "Parameters": trainable_params})

    # --- Improved part: Parameter count per module ---
    print("\n--- Parameters per Module ---")
    module_params = {}
    for name, module in model.named_children(): # Use named_children to get module names
        module_param_count = count_parameters(module)
        module_params[name] = module_param_count
        print(f"Module '{name}': {module_param_count} parameters")


    # --- Memory Usage Measurement ---
    print("\n--- Memory Usage ---")

    # --- GPU memory (if available) ---
    if torch.cuda.is_available():
        initial_gpu_memory = torch.cuda.memory_allocated(device=device) # Get initial GPU memory usage
        torch.cuda.synchronize(device=device)
        model, input, t, y = model.to(device=device), input.to(device=device), t.to(device=device), y.to(device=device) # Move model and input to GPU
        # with torch.no_grad(): 
        #     with torch.autocast("cuda"): model(input, t, y=y) # Run the model to allocate memory
        with torch.autocast("cuda"): 
            model(input, t, y=y)
        torch.cuda.synchronize(device=device)
        final_gpu_memory = torch.cuda.memory_allocated(device=device) # Get GPU memory usage after model run
        gpu_memory_usage = final_gpu_memory - initial_gpu_memory
        print(f"GPU Memory Usage: {gpu_memory_usage / (1024**2):.2f} MB") # Convert to MB

        max_gpu_memory_allocated = torch.cuda.max_memory_allocated(device=device)
        print(f"Max GPU Memory Allocated: {max_gpu_memory_allocated / (1024**2):.2f} MB") # Convert to MB
        torch.cuda.reset_peak_memory_stats(device=device) # Reset peak memory stats for future measurements if needed

    else:
        print("CUDA is not available, skipping GPU memory measurement.")

    import psutil
    # --- System RAM memory ---
    initial_ram_usage = psutil.Process().memory_info().rss # in bytes
    model(input, t, y=y) # Run the model again to measure RAM usage (if not already run above)
    final_ram_usage = psutil.Process().memory_info().rss # in bytes
    ram_memory_usage = final_ram_usage - initial_ram_usage
    print(f"RAM Memory Usage: {ram_memory_usage / (1024**2):.2f} MB") # Convert to MB