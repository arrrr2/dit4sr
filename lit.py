import torch
import torch.nn as nn

class LinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.,
        proj_drop=0.,
        kernel_function=nn.ReLU,
        kernel_size=5,

    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
        groups=head_dim, padding=kernel_size // 2)
        self.kernel_function = kernel_function() 
    def forward(self, x, HW=None):
        B, N, C = x.shape

        if HW is None:
            H = W = int(N ** 0.5)
        else:
            H, W = HW

        q = self.q(x) # (B, N, D)
        dtype = q.dtype
        kv = self.kv(x).reshape(B, N, 2, C).permute(2, 0, 1, 3) # (2, B, N, D)
        k, v = kv[0], kv[1] # (B, N, D)
        q = self.kernel_function(q) + 1e-6 # (B, N, D)
        k = self.kernel_function(k) + 1e-6 # (B, N, D)
        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3).to(dtype) # (B, h, N, D/h)
        k = k.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3).to(dtype) # (B, h, N, D/h)
        v = v.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3).to(dtype) # (B, h, N, D/h)
        

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6) # (B, h, N, 1)
        kv = (k.transpose(-2, -1) * (N ** -0.5)) @ (v * (N ** -0.5)) # (B, h, D/h, N) @ (B, h, N, D/h) = (B, h, D/h, D/h)
        x = q @ kv * z # (B, h, N, D/h) @ (B, H, D/h, D/h) * (B, h, N, 1) = (B, h, N, D/h)
        x = x.transpose(1, 2).reshape(B, N, C) # (B, N, D)
        v = v.reshape(B * self.num_heads, H, W, -1).permute(0, 3, 1, 2) # (B*h, D/h, H, W)
        x = x + self.dwc(v).reshape(B, C, N).permute(0, 2, 1) # (B, N, D)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

class dwc_ffn(nn.Module):
    def __init__(self, 
                 hidden_size, 
                 mlp_ratio=4.0,
                 num_heads=4,
                 kernel_function=nn.ReLU,
                 kernel_size=3,
                 ):
        super().__init__()

        mid_size = int(hidden_size * mlp_ratio)

        self.mlp_conv = nn.Conv2d(in_channels=hidden_size, out_channels=mid_size, kernel_size=1)

        self.dwconv = nn.Conv2d(in_channels=hidden_size, out_channels=mid_size, kernel_size=kernel_size,
                                groups=hidden_size // num_heads, padding=kernel_size // 2)
        
        self.kernel_function = kernel_function()

        self.after_conv = nn.Conv2d(in_channels=mid_size, out_channels=hidden_size, kernel_size=1)

    def forward(self, x, HW=None):

        if HW is None:
            H = W = int(x.shape[1] ** 0.5)
        else:
            H, W = HW

        x = x.view(x.shape[0], H, W, -1).permute(0, 3, 1, 2)


        x0 = self.mlp_conv(x)
        x1 = self.dwconv(x)

        x = x0 * (1 + self.kernel_function(x1))

        x = self.after_conv(x)

        x = x.view(x.shape[0], -1, H * W).permute(0, 2, 1)


        return x
        
