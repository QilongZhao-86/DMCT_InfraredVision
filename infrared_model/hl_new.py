import math
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from guided_filter_pytorch.guided_filter import ConvGuidedFilter
from guided_filter_pytorch.guided_filter import GuidedFilter
    
class LayerNorm2d(nn.Module):
    """LN for (B,C,H,W) across spatial dims per channel."""
    def __init__(self, c: int, eps: float = 1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(c))
        self.b = nn.Parameter(torch.zeros(c))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.w[:, None, None] + self.b[:, None, None]


class FFTBlock(nn.Module):
    def __init__(self, cutoff_ratio: float = 0.25, mode: str = "low"):
        super().__init__()
        assert mode in ("low", "high")
        self.cutoff_ratio = cutoff_ratio
        self.mode = mode

    def forward(self, x: Tensor) -> Tensor:
        assert x.dim() == 4, "FFTBlock expects (B,C,H,W)"
        B, C, H, W = x.shape

        fft = torch.fft.fft2(x, norm="ortho")
        fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))

        cutoff = int(min(H, W) * self.cutoff_ratio / 2.0)
        mask_low = torch.zeros_like(fft_shifted)
        cx, cy = H // 2, W // 2
        if cutoff > 0:
            mask_low[:, :, cx - cutoff:cx + cutoff, cy - cutoff:cy + cutoff] = 1.0

        if self.mode == "low":
            fft_f = fft_shifted * mask_low
        else:
            fft_f = fft_shifted * (1.0 - mask_low)

        out = torch.fft.ifft2(torch.fft.ifftshift(fft_f, dim=(-2, -1)), norm="ortho").real
        return out

    
class trans(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x,y):
        return y-x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        # x: [B, C, H, W]
        avg_out = torch.mean(x, dim=1, keepdim=True)          # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)        # [B, 1, H, W]
        x_cat = torch.cat([avg_out, max_out], dim=1)          # [B, 2, H, W]
        attn = self.conv(x_cat)                               # [B, 1, H, W]
        attn = torch.sigmoid(attn)                            # (0,1)
        return attn

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False)
        )

    def forward(self, x):
        """
        x: [B, C, H, W]
        return: w: [B, C, 1, 1]
        """
        B, C, H, W = x.shape

        gap = F.adaptive_avg_pool2d(x, 1).view(B, C)   # [B, C]

        w = self.mlp(gap)                              # [B, C]
        w = torch.sigmoid(w).view(B, C, 1, 1)          # [B, C, 1, 1]

        return w
# class EfficientSpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super().__init__()
#         self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         # x: [B, C, H, W]
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         res = torch.cat([avg_out, max_out], dim=1)
#         res = self.conv(res)
#         return self.sigmoid(res)
    

class HighLowPyramid(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int = 32,
        radiux=(2, 4, 6),
        eps_list=(1e-4, 1e-3),
        sizes=((64, 64), (32, 32), (16, 16), (8, 8)),
        use_attention: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.radiux = list(radiux)
        self.eps_list = list(eps_list)
        self.sizes = sizes
        self.use_attention = use_attention

        self.num_branches = len(self.radiux) * len(self.eps_list)
        cat_channels = channels * self.num_branches  # LF_cat/HF_cat channels

        self.pre_conv = nn.Conv2d(channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.pre_act = nn.ReLU(inplace=True)

        if use_attention:
            self.spatial_attn = SpatialAttention()
            self.channel_attn = ChannelAttention(channels=out_channels, reduction=8)

        self.proj_l = nn.Conv2d(cat_channels, out_channels, kernel_size=1, bias=True)
        self.proj_h = nn.Conv2d(cat_channels, out_channels, kernel_size=1, bias=True)

        self._conv_gf_bank = nn.ModuleDict()
        for r in self.radiux:
            self._conv_gf_bank[str(r)] = ConvGuidedFilter(radius=r)

    @staticmethod
    def get_residue(x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W)  residue: (B,1,H,W)
        max_c = torch.max(x, dim=1, keepdim=True)[0]
        min_c = torch.min(x, dim=1, keepdim=True)[0]
        return max_c - min_c

    def _guided_filter(self, guide: torch.Tensor, src: torch.Tensor, radius: int, eps: float) -> torch.Tensor:
        """
        guide: (B,1,H,W)  residue
        src:   (B,C,H,W)  x
        return: LF (B,C,H,W)
        """
        gf = GuidedFilter(radius, eps)       # 用位置参数
        return gf(guide, src)
    
    def decomposition(self, x, C):
        LF_list = []
        HF_list = []
        res = self.get_residue(x)
        for radius in self.radiux:
            for eps in self.eps_list:
                gf = GuidedFilter(radius, eps)   #用位置参数
                LF = gf(res, x)
                LF_list.append(LF)
                HF_list.append(x - LF)
        LF = torch.cat(LF_list, dim=1)
        HF = torch.cat(HF_list, dim=1)
        return LF, HF
    
    def forward(self, x: torch.Tensor):
        """
          每个元素 (B,out_channels,h_i,w_i)
        """
        B, C, H, W = x.shape
        assert C == self.channels, f"expected channels={self.channels}, got {C}"

        LF_cat, HF_cat = self.decomposition(x, C=x.shape[1])
        low = self.proj_l(LF_cat)   # (B,out_channels,H,W)
        high = self.proj_h(HF_cat)  # (B,out_channels,H,W)

        if self.use_attention:
            feat = self.pre_act(self.pre_conv(x))  # (B,out_channels,H,W)
            e = self.spatial_attn(feat)   # (B,1,H,W)
            w = self.channel_attn(feat)   # (B,out_channels,1,1)

            high = e * high
            low = (1.0 - e) * low * w

        low_list, high_list = [], []
        for hh, ww in self.sizes:
            low_list.append(F.interpolate(low, size=(hh, ww), mode="area"))
            high_list.append(F.interpolate(high, size=(hh, ww), mode="bilinear", align_corners=False))

        return low_list, high_list

class FiLMFromFHFL(nn.Module):
    def __init__(self, hl_channels: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * hl_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * hl_channels), 
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, fd_p: Tensor, fh_p: Tensor, fl_p: Tensor) -> Tensor:
        # GAP: (B,C,H,W) -> (B,C)
        cond = torch.cat(
            [fh_p.mean(dim=(2, 3)), fl_p.mean(dim=(2, 3))],
            dim=1
        )  # (B,2C)

        gb = self.mlp(cond)              # (B,2C)
        gamma, beta = gb.chunk(2, dim=1) # each (B,C)

        gamma = gamma[:, :, None, None]
        beta  = beta[:, :, None, None]

        return fd_p * (1.0 + gamma) + beta



class MultiScaleAdapter(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hl_channels: int = 32,
        hidden_dim: Optional[int] = None,
        num_heads_t: int = 4,
        num_heads_s: int = 4,
        cutoff_ratio: float = 0.25,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hl_channels = hl_channels
        self.hidden_dim = hidden_dim or in_channels
        
        self.film = FiLMFromFHFL(hl_channels, hidden=128)##新加的FiLM门控的定义

        self.low_path = nn.Sequential(
            nn.Conv2d(hl_channels, hl_channels, kernel_size=1, bias=True),
            FFTBlock(cutoff_ratio=cutoff_ratio,mode="low"),
            nn.ReLU(inplace=True),
        )
        self.high_path = nn.Sequential(
            nn.Conv2d(hl_channels, hl_channels, kernel_size=1, bias=True),
            FFTBlock(cutoff_ratio=cutoff_ratio,mode="high"),
            nn.ReLU(inplace=True),
        )
        
        self.fd_proj = nn.Conv2d(in_channels, hl_channels, kernel_size=1, bias=True)
        
        self.beta_l = nn.Parameter(torch.tensor(0.0))
        self.beta_h = nn.Parameter(torch.tensor(0.0))
        self.gamma = nn.Parameter(torch.tensor(0.0))
        
        self.fuse = nn.Sequential(
            nn.Conv2d(hl_channels, self.hidden_dim, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim, in_channels, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.fuse[-1].weight)
        nn.init.zeros_(self.fuse[-1].bias)
        self.align_h = None
        self.align_l = None

    def forward(self, fd: Tensor, fh: Tensor, fl: Tensor) -> Tensor:
        """
        fd: (B, in_channels, H, W)
        fh: (B, hl_channels, h, w)  (来自 high_list[inc_num])
        fl: (B, hl_channels, h, w)  (来自 low_list[inc_num])
        """
        assert fd.dim() == 4, "fd must be (B,C,H,W)"
        B, C, H, W = fd.shape
        assert C == self.in_channels, f"expected fd C={self.in_channels}, got {C}"

        if fh.shape[-2:] != (H, W):
            fh = F.interpolate(fh, size=(H, W), mode="bilinear", align_corners=False)
        if fl.shape[-2:] != (H, W):
            fl = F.interpolate(fl, size=(H, W), mode="bilinear", align_corners=False)

        fl_p = self.low_path(fl)
        fh_p = self.high_path(fh)
        fd_p = self.fd_proj(fd)

        fd_mod = self.film(fd_p, fh_p, fl_p)   #FiLM调制在这里
        delta = self.fuse(fd_mod)
        return fd + delta

