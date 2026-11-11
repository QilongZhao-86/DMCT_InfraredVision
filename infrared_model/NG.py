# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import kornia.filters as KF
# import kornia.augmentation as KA

# # =========================
# # Selector (no strength head)
# # =========================

# class DegradationSelector(nn.Module):
#     """
#     Predicts:
#     - p: whether to degrade
#     - g: which degradation (one-hot)
#     """
#     def __init__(self, num_types):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(3, 32, 1),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(32, num_types + 1, 1),
#         )

#     def forward(self, x, tau=0.7):
#         out = self.net(x).squeeze(-1).squeeze(-1)

#         logits = out[:, :-1]
#         raw_p  = out[:, -1]

#         g = F.gumbel_softmax(logits, tau=tau, hard=True)
#         p = torch.sigmoid(raw_p)

#         return g, p
    
# def soft_clamp(x):
#     return torch.clamp(x, 0.0, 1.0)


# # class NG(nn.Module):
# #     def __init__(self):
# #         super().__init__()

# #         self.selector = DegradationSelector(num_types=2)

# #         self.ops = nn.ModuleList([
# #             KF.GaussianBlur2d(
# #                 kernel_size=(5, 5),
# #                 sigma=(0.4, 0.4)
# #             ),
# #             KA.RandomGaussianNoise(
# #                 mean=0.0,
# #                 std=5e-4,
# #                 p=1.0
# #             ),
# #         ])

# #         self.strength = 5e-4
# #         self.max_apply_prob = 0.01

# #     def forward(self, x):
# #         # print("正在使用ng")
# #         g, p = self.selector(x)
# #         if x.max() > 1.0:
# #             x = x / 255.0
# #         apply = (torch.rand_like(p) < p).float()

# #         y = x

# #         for i, op in enumerate(self.ops):
# #             y_i = op(x)
# #             delta = y_i - x
# #             delta = delta - delta.mean(dim=[2, 3], keepdim=True)

# #             y = y + (
# #                 apply[:, None, None, None]
# #                 * g[:, i, None, None, None]
# #                 * self.strength
# #                 * delta
# #             )
# #         # print(f"Before clamp - Max: {y.max().item()}, Min: {y.min().item()}")
# #         return soft_clamp(y)

# #可以生效的但是强度还是太大

# # class NG(nn.Module):
# #     def __init__(self):
# #         super().__init__()
# #         self.selector = DegradationSelector(num_types=2)
# #         self.ops = nn.ModuleList([
# #             KF.GaussianBlur2d(kernel_size=(5, 5), sigma=(1.0, 1.0)),
# #             KA.RandomGaussianNoise(mean=0.0, std=0.05, p=1.0),
# #         ])
# #         self.strength = 0.001 
# #         self.max_apply_prob = 0.01 

# #     def forward(self, x):
# #         g, p = self.selector(x)
# #         if x.max() > 1.0: x = x / 255.0

# #         actual_p = p * self.max_apply_prob
# #         apply = (torch.rand_like(actual_p) < actual_p).float()

# #         y = x.clone()
# #         for i, op in enumerate(self.ops):
# #             y_i = op(x)
# #             delta = y_i - x
            
# #             delta = delta - delta.mean(dim=[1, 2, 3], keepdim=True)
            
# #             rms = torch.sqrt(torch.mean(delta**2, dim=[1, 2, 3], keepdim=True) + 1e-8)
# #             delta = delta / rms 

# #             y = y + (
# #                 apply[:, None, None, None]
# #                 * g[:, i, None, None, None]
# #                 * self.strength 
# #                 * delta
# #             )
# #         return torch.clamp(y, 0.0, 1.0)


# class NG(nn.Module):
#     def __init__(self, strength=0.01):
#         super().__init__()

#         self.selector = DegradationSelector(num_types=3) 
#         self.strength = strength * 20 

#     def debug_perturbation(self, x, y, name="NG_Stage"):
#         with torch.no_grad():
#             diff = torch.abs(y - x)
#             max_diff = diff.max().item()
#             mean_diff = diff.mean().item()
#             rel_perturb = (mean_diff / (torch.abs(x).mean() + 1e-8)) * 100
#             shift = (y.mean() - x.mean()).item()
#             mse = torch.mean((y - x) ** 2)
#             psnr = 10 * torch.log10(1 / (mse + 1e-10))

#             print(f"--- [{name}] Diagnosis ---")
#             print(f"Max Perturb: {max_diff:.2e} | Rel Intensity: {rel_perturb:.6f}%")
#             print(f"Signal Integrity (PSNR): {psnr.item():.2f} dB | Mean Shift: {shift:.2e}")

#     def forward(self, x):
#         g, p = self.selector(x)
#         # 批量概率掩码
#         apply = (torch.rand(x.size(0), device=x.device) < p).float()
#         y = x.clone()
        
#         # 动态捕捉特征量级
#         feat_std = x.std().item() + 1e-6

#         with torch.no_grad():
#             # 保护小目标：提取局部最大值
#             max_pool = F.max_pool2d(x, kernel_size=7, stride=1, padding=3)
#             # 保护区域掩码：目标及其周围 0.8 倍亮度区不加噪声
#             protect_mask = (x < (max_pool * 0.8)).float()

#         # --- 算子 0：横向条纹 (Horizontal Stripes) ---
#         if g[:, 0].any():
#             # 模拟行扫描或读出电路噪声
#             # 下采样再插值是为了产生“有厚度”的条纹，而不是像素毛刺
#             stripe_h = torch.randn(x.size(0), 1, x.size(2)//4, 1, device=x.device)
#             stripe_h = F.interpolate(stripe_h, size=(x.size(2), 1), mode='nearest')
#             y = y + apply[:,None,None,None] * g[:,0,None,None,None] * (self.strength * feat_std) * stripe_h * protect_mask

#         # --- 算子 1：纵向条纹 (Vertical Stripes) ---
#         if g[:, 1].any():
#             # 模拟列非均匀性噪声 (红外中最常见的噪声)
#             stripe_v = torch.randn(x.size(0), 1, 1, x.size(3)//4, device=x.device)
#             stripe_v = F.interpolate(stripe_v, size=(1, x.size(3)), mode='nearest')
#             y = y + apply[:,None,None,None] * g[:,1,None,None,None] * (self.strength * feat_std) * stripe_v * protect_mask

#         # --- 算子 2：分辨率大幅度下采样 (Low Res) ---
#         if g[:, 2].any():
#             h, w = x.size()[2:]
#             # 模拟光学系统模糊或远距离低分辨率
#             low_res = F.interpolate(x, scale_factor=0.4, mode='bilinear', align_corners=False)
#             res_back = F.interpolate(low_res, size=(h, w), mode='bilinear', align_corners=False)
#             delta = res_back - x
#             y = y + apply[:,None,None,None] * g[:,2,None,None,None] * self.strength * delta * protect_mask

#         # self.debug_perturbation(x, y)
#         return y
import os
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from PIL import Image
import torchvision.transforms.functional as TF


# =========================
# Degradation Hyperparams
# =========================

# ---------------- Gaussian Noise ----------------
GN_BASE = 0.0
GN_SCALE = 1.0

# ---------------- Stripe Noise ----------------
SN_BASE = 0.0
SN_SCALE = 0.5

# ---------------- Shot Noise ----------------
SHOT_BASE = 0.0
SHOT_SCALE = 1.0

# ---------------- Optical Noise ----------------
ON_BASE = 0.0
ON_SCALE = 0.3
ON_DOWNSAMPLE = 2

# ---------------- Blur ----------------
BLUR_BASE = 0.0
BLUR_SCALE = 1.0
BLUR_POOL_KERNEL = 3

# ---------------- Contrast ----------------
CONTRAST_BASE = 1.0
CONTRAST_DECAY = 1.0
CONTRAST_MIN = 0.85

# ---------------- Resolution ----------------
RESOLUTION_BASE = 0.0
RESOLUTION_SCALE = 1.0
RESOLUTION_DOWN_FACTOR = 0.5


def _reshape_strength(s, x):
    """
    将 strength reshape 成可广播到 [B,C,H,W] 的形式。
    支持:
      scalar
      [B]
      [B,1]
    """
    if not torch.is_tensor(s):
        s = torch.tensor(s, device=x.device, dtype=x.dtype)
    s = s.to(device=x.device, dtype=x.dtype)

    if s.dim() == 0:
        s = s.view(1, 1, 1, 1).expand(x.size(0), 1, 1, 1)
    elif s.dim() == 1:
        s = s.view(-1, 1, 1, 1)
    elif s.dim() == 2:
        s = s.view(s.size(0), s.size(1), 1, 1)
    else:
        raise ValueError(f"Unsupported strength shape: {s.shape}")

    return s


# =========================
# Degradation Functions
# =========================

def gaussian_noise(x, s):
    s = _reshape_strength(s, x)
    std = GN_BASE + GN_SCALE * s
    noise = torch.randn_like(x)
    return x + noise * std


def stripe_noise(x, s):
    s = _reshape_strength(s, x)
    b, c, h, w = x.shape

    amp = SN_BASE + SN_SCALE * s
    col = torch.randn(b, 1, 1, w, device=x.device, dtype=x.dtype)
    noise = col.expand(b, 1, h, w).expand(b, c, h, w)
    return x + amp * noise


def shot_noise(x, s):
    s = _reshape_strength(s, x)
    x_clamped = torch.clamp(x, 0, 1)
    amp = SHOT_BASE + SHOT_SCALE * s
    noise = torch.randn_like(x_clamped) * torch.sqrt(torch.clamp(x_clamped, min=1e-6))
    return x_clamped + amp * noise


def optical_noise(x, s):
    s = _reshape_strength(s, x)
    b, c, h, w = x.shape

    hh = max(1, h // ON_DOWNSAMPLE)
    ww = max(1, w // ON_DOWNSAMPLE)

    low_noise = torch.randn(b, 1, hh, ww, device=x.device, dtype=x.dtype)
    low_noise = F.interpolate(low_noise, size=(h, w), mode="bilinear", align_corners=False)
    low_noise = low_noise.expand(b, c, h, w)

    amp = ON_BASE + ON_SCALE * s
    return x + amp * low_noise


def blur(x, s):
    s = _reshape_strength(s, x)
    alpha = torch.clamp(BLUR_BASE + BLUR_SCALE * s, 0.0, 1.0)

    pad = BLUR_POOL_KERNEL // 2
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    x_blur = F.avg_pool2d(x_pad, kernel_size=BLUR_POOL_KERNEL, stride=1)

    return x * (1.0 - alpha) + x_blur * alpha


def contrast(x, s):
    s = _reshape_strength(s, x)
    factor = torch.clamp(CONTRAST_BASE - CONTRAST_DECAY * s, min=CONTRAST_MIN, max=1.0)
    mean = x.mean(dim=(2, 3), keepdim=True)
    return mean + factor * (x - mean)


def resolution(x, s):
    s = _reshape_strength(s, x)
    alpha = torch.clamp(RESOLUTION_BASE + RESOLUTION_SCALE * s, 0.0, 1.0)

    b, c, h, w = x.shape
    h2 = max(1, int(h * RESOLUTION_DOWN_FACTOR))
    w2 = max(1, int(w * RESOLUTION_DOWN_FACTOR))

    x_low = F.interpolate(x, size=(h2, w2), mode="bilinear", align_corners=False)
    x_low = F.interpolate(x_low, size=(h, w), mode="bilinear", align_corners=False)

    return x * (1.0 - alpha) + x_low * alpha


_FUNC = {
    "GaussianNoise": gaussian_noise,
    "StripeNoise": stripe_noise,
    "ShotNoise": shot_noise,
    "OpticalNoise": optical_noise,
    "Blur": blur,
    "Contrast": contrast,
    "Resolution": resolution,
}


class NG(nn.Module):
    """
    以第一段代码为主：
      - 网络结构
      - ng_types
      - max_strength
      - delta_cap
      - forward_learnable / forward_degrade / apply_all / apply_single / apply_subset

    同时加入第二段代码的能力：
      - 返回 raw/norm/mapped 强度信息
      - inspect 单张图
      - 保存退化图
      - 打印各项强度
    """

    def __init__(self,**KWARGS):
        super().__init__()

        self.ng_types = [
            "StripeNoise",
            "OpticalNoise",
            "Blur",
            "Contrast",
            "Resolution",
        ]
        self.type_num = len(self.ng_types)

        self.encoder = nn.Sequential(
            nn.Upsample(size=(256, 256), mode="bilinear", align_corners=False),

            nn.Conv2d(3, 32, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv2d(32, 64, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv2d(64, 128, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=False),

            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(128, self.type_num),
        )

        # 不要在这里强绑 device='cuda'
        self.log_lambda = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        self.register_buffer(
            "max_strength",
            torch.tensor([
                0.0200,  # StripeNoise
                0.0200,  # OpticalNoise
                0.2500,  # Blur
                0.2000,  # Contrast
                0.2500,  # Resolution
            ], dtype=torch.float32)
        )

        self.register_buffer("delta_cap", torch.tensor(0.20, dtype=torch.float32))

    def get_strength_info(self, img: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        输入:
            img: [B,C,H,W]  (C 可以是 1 或 3)

        返回:
            raw_strength:   [B,T]
            norm_strength:  [B,T] in (0,1)
            strength:       [B,T] 真正用于退化的物理强度
        """

        # ========= 通道对齐 =========
        if img.dim() != 4:
            raise ValueError(f"Expected [B,C,H,W], got {img.shape}")

        # encoder[0] 是 Upsample
        # encoder[1] 是 Conv2d
        in_ch = self.encoder[1].in_channels

        if in_ch == 1:
            if img.size(1) == 3:
                img = img.mean(dim=1, keepdim=True)   # RGB → 灰度
            elif img.size(1) != 1:
                raise ValueError(f"Unsupported input channels: {img.size(1)}")

        elif in_ch == 3:
            if img.size(1) == 1:
                img = img.repeat(1, 3, 1, 1)          # 灰度 → 伪RGB
            elif img.size(1) != 3:
                raise ValueError(f"Unsupported input channels: {img.size(1)}")

        else:
            raise ValueError(f"Unsupported encoder input channels: {in_ch}")

        # ========= Forward =========
        feat = self.encoder(img).view(img.size(0), -1)   # [B,128]

        raw_strength = self.mlp(feat)                    # [B,T]
        norm_strength = torch.sigmoid(raw_strength)      # [B,T]
        strength = norm_strength * self.max_strength.unsqueeze(0)

        return {
            "raw_strength": raw_strength,
            "norm_strength": norm_strength,
            "strength": strength,
        }
    def forward(
        self,
        img: torch.Tensor,
        return_strength: bool = False,
        return_info_dict: bool = False,
        **kwargs
    ):
        """
        img: [B,1,H,W]

        return_strength=True:
            返回 (out, strength, norm_strength, raw_strength)

        return_info_dict=True:
            返回 (out, info_dict)

        默认:
            只返回 out
        """
        info = self.get_strength_info(img)
        out = self.apply_all(img, info["strength"])

        if return_info_dict:
            return out, info

        if return_strength:
            return out, info["strength"], info["norm_strength"], info["raw_strength"]

        return out

    @torch.no_grad()
    def get_degradation_stats(self, img_input):
        """
        兼容你 trainer 当前的统计接口。

        返回格式：
        stats = {
            ng_type: {
                "avg_weight": ...,
                "avg_gamma": ...,
                "effective_strength": ...,
                ...
            }
        }

        说明：
        - 第一段代码结构里没有独立的 weight/gamma 分解
        - 这里只有 raw -> sigmoid -> mapped_strength
        - 所以这里做兼容映射：
            avg_weight        := mean(norm_strength)
            avg_gamma         := mean(norm_strength)
            effective_strength:= mean(mapped_strength)
        """
        info = self.get_strength_info(img_input)
        norm_strength = info["norm_strength"]   # [B, T]
        strength = info["strength"]             # [B, T]

        mean_norm = norm_strength.mean(dim=0)   # [T]
        mean_strength = strength.mean(dim=0)    # [T]

        stats = {}
        for ind, ng_type in enumerate(self.ng_types):
            item = {
                "avg_weight": mean_norm[ind].item(),
                "avg_gamma": mean_norm[ind].item(),
                "effective_strength": mean_strength[ind].item(),
                "norm_strength": mean_norm[ind].item(),
                "mapped_strength": mean_strength[ind].item(),
                "max_strength": self.max_strength[ind].item(),
            }

            # 为了和你原来的 print 风格一致，加一些更直观的物理解释
            if ng_type == "Blur":
                item["alpha"] = mean_strength[ind].item()

            elif ng_type == "Contrast":
                contrast_s = mean_strength[ind].item()
                contrast_factor = max(CONTRAST_MIN, min(1.0, CONTRAST_BASE - CONTRAST_DECAY * contrast_s))
                item["contrast_factor"] = contrast_factor

            elif ng_type == "Resolution":
                item["alpha"] = mean_strength[ind].item()
                item["down_factor"] = RESOLUTION_DOWN_FACTOR

            elif ng_type == "StripeNoise":
                stripe_amp = SN_BASE + SN_SCALE * mean_strength[ind].item()
                item["stripe_amplitude"] = stripe_amp

            elif ng_type == "OpticalNoise":
                optical_amp = ON_BASE + ON_SCALE * mean_strength[ind].item()
                item["optical_amplitude"] = optical_amp

            stats[ng_type] = item

        return stats
    # =========================
    # Non-learnable degrade
    # =========================
    @torch.no_grad()
    def forward_degrade(self, img, mode="all", strength=0.1, num_select=1, return_info=False):
        """
        非学习退化模式:
          - mode='all': 随机选 num_select 个退化串联
          - mode in self.ng_types: 单一退化

        return_info=True 时，额外返回每个样本实际使用的退化信息
        """
        b = img.size(0)
        device = img.device

        if not isinstance(mode, str):
            raise RuntimeError(f"mode must be a string, but got type={type(mode)}")

        if mode == "all":
            out_batch = []
            info_batch = []
            k = max(1, min(num_select, self.type_num))

            for i in range(b):
                x = img[i:i+1]
                per_img_info = []

                idxs = torch.randperm(self.type_num, device=device)[:k]
                for idx in idxs:
                    t = self.ng_types[idx.item()]
                    rand_strength = torch.rand(1, device=device, dtype=img.dtype) * strength
                    x = _FUNC[t](x, rand_strength.view(1, 1))
                    x = torch.clamp(x, 0, 1)

                    per_img_info.append({
                        "type": t,
                        "strength": float(rand_strength.item())
                    })

                out_batch.append(x)
                info_batch.append(per_img_info)

            out = torch.cat(out_batch, dim=0)
            if return_info:
                return out, info_batch
            return out

        # _FUNC also exposes GaussianNoise and ShotNoise as explicit audit
        # modes; they are intentionally not part of the learned five-op path.
        if mode in _FUNC:
            out = self.apply_single(img, mode, strength)
            if return_info:
                info_batch = [[{"type": mode, "strength": float(strength)}] for _ in range(b)]
                return out, info_batch
            return out

        raise RuntimeError(f"Unsupported mode: {mode}. Expected 'all' or one of {self.ng_types}")

    # =========================
    # Apply helpers
    # =========================
    def apply_all(self, img, strength_vec):
        x = img

        for j, t in enumerate(self.ng_types):
            s = strength_vec[:, j:j+1]   # [B,1]
            x_old = x
            x_new = _FUNC[t](x_old, s)

            delta = x_new - x_old
            delta = torch.clamp(delta, -self.delta_cap, self.delta_cap)

            x = torch.clamp(x_old + delta, 0, 1)

        return x

    @torch.no_grad()
    def apply_single(self, img, mode, strength):
        if mode not in _FUNC:
            raise ValueError(f"Unknown mode: {mode}")

        b = img.size(0)
        s = torch.full((b, 1), float(strength), device=img.device, dtype=img.dtype)
        out = _FUNC[mode](img, s)
        return torch.clamp(out, 0, 1)

    @torch.no_grad()
    def apply_subset(self, img, modes, strength):
        b = img.size(0)
        s = torch.full((b, 1), float(strength), device=img.device, dtype=img.dtype)
        x = img

        for m in modes:
            if m not in _FUNC:
                raise ValueError(f"Unknown mode in subset: {m}")
            x = _FUNC[m](x, s)
            x = torch.clamp(x, 0, 1)

        return x

    # =========================
    # Inspect / Print Utilities
    # =========================
    @torch.no_grad()
    def inspect_single_image_and_save(
        self,
        img_tensor: torch.Tensor,
        epoch: Optional[int] = None,
        save_dir: Optional[str] = None,
        save_input: bool = False,
        file_prefix: Optional[str] = None,
        verbose: bool = True,
        learnable: bool = True,
        degrade_mode: str = "all",
        degrade_strength: float = 0.1,
        degrade_num_select: int = 1,
    ):
        """
        参考第二段代码的 inspect 功能，但适配第一段代码逻辑。

        img_tensor:
            支持 [H,W], [1,H,W], [B,1,H,W]
        """
        self.eval()

        if not torch.is_tensor(img_tensor):
            raise TypeError(f"img_tensor 必须是 torch.Tensor, 但得到的是 {type(img_tensor)}")

        # 统一成 [1,1,H,W]
        if img_tensor.dim() == 2:
            img_tensor = img_tensor.unsqueeze(0)  # [1,H,W]
        elif img_tensor.dim() == 4:
            if img_tensor.size(0) != 1:
                raise ValueError(f"如果传入 4 维 tensor，batch size 必须为 1，但现在 shape={img_tensor.shape}")
            img_tensor = img_tensor.squeeze(0)    # [1,H,W]
        elif img_tensor.dim() != 3:
            raise ValueError(f"img_tensor 应该是 [H,W] / [1,H,W] / [1,1,H,W]，但现在 shape={img_tensor.shape}")

        if img_tensor.dim() != 3 or img_tensor.size(0) != 1:
            raise ValueError(f"当前 NG 网络按第一段代码要求输入单通道 [1,H,W]，但现在 shape={img_tensor.shape}")

        model_device = next(self.parameters()).device
        img_tensor = img_tensor.to(model_device).float()

        # 如果像素范围像 0~255，则归一化到 0~1
        if img_tensor.max() > 1.5:
            img_tensor = img_tensor / 255.0

        img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
        x = img_tensor.unsqueeze(0)  # [1,1,H,W]

        if learnable:
            info = self.get_strength_info(x)
            out = self.apply_all(x, info["strength"])

            raw_strength = info["raw_strength"]      # [1,T]
            norm_strength = info["norm_strength"]    # [1,T]
            strength = info["strength"]              # [1,T]

            info_dict = {
                "mode": "learnable",
                "raw_strength": raw_strength[0].detach().cpu(),
                "norm_strength": norm_strength[0].detach().cpu(),
                "mapped_strength": strength[0].detach().cpu(),
                "type_to_strength": {
                    t: float(strength[0, i].item()) for i, t in enumerate(self.ng_types)
                }
            }

        else:
            out, degrade_info = self.forward_degrade(
                x,
                mode=degrade_mode,
                strength=degrade_strength,
                num_select=degrade_num_select,
                return_info=True
            )
            info_dict = {
                "mode": "degrade",
                "degrade_mode": degrade_mode,
                "degrade_strength": degrade_strength,
                "degrade_num_select": degrade_num_select,
                "degrade_info": degrade_info[0],
            }

        out = torch.clamp(out, 0.0, 1.0)
        delta = (out - x).abs()
        mean_delta = delta.mean().item()
        max_delta = delta.max().item()

        save_path = None
        input_path = None
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

            if file_prefix is None:
                if epoch is None:
                    file_prefix = "inspect"
                else:
                    file_prefix = f"epoch_{epoch:03d}"

            save_path = os.path.join(save_dir, f"{file_prefix}_degraded.png")
            vutils.save_image(out.cpu(), save_path)

            if save_input:
                input_path = os.path.join(save_dir, f"{file_prefix}_input.png")
                vutils.save_image(x.cpu(), input_path)

        if verbose:
            print("\n===== NG Inspect =====")
            if epoch is not None:
                print(f"Epoch: {epoch}")
            print(f"input shape: {tuple(x.shape)}")
            print(f"mean_delta={mean_delta:.6f} | max_delta={max_delta:.6f}")

            if save_path is not None:
                print(f"saved degraded image to: {save_path}")
            if input_path is not None:
                print(f"saved input image to:    {input_path}")

            if learnable:
                print("---- Learnable Strength Info ----")
                for i, t in enumerate(self.ng_types):
                    print(
                        f"{t:<12} | "
                        f"logit={info_dict['raw_strength'][i].item(): .6f} | "
                        f"norm={info_dict['norm_strength'][i].item(): .6f} | "
                        f"mapped={info_dict['mapped_strength'][i].item(): .6f}"
                    )
            else:
                print("---- Degrade Info ----")
                for item in info_dict["degrade_info"]:
                    print(f"{item['type']:<12} | strength={item['strength']:.6f}")

            print(f"input mean/std:  {x.mean().item():.6f} / {x.std().item():.6f}")
            print(f"output mean/std: {out.mean().item():.6f} / {out.std().item():.6f}")

        return out, info_dict

    @torch.no_grad()
    def inspect_image_path_and_save(
        self,
        image_path: str,
        epoch: Optional[int] = None,
        save_dir: Optional[str] = None,
        save_input: bool = False,
        resize_to: Optional[tuple] = None,
        verbose: bool = True,
        learnable: bool = True,
        degrade_mode: str = "all",
        degrade_strength: float = 0.1,
        degrade_num_select: int = 1,
    ):
        """
        从路径读取灰度图并 inspect。
        resize_to: (H, W)
        """
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"找不到图片文件: {image_path}")

        img = Image.open(image_path).convert("L")

        if resize_to is not None:
            if not (isinstance(resize_to, (tuple, list)) and len(resize_to) == 2):
                raise ValueError(f"resize_to 应该是 (H, W)，现在是: {resize_to}")
            img = img.resize((resize_to[1], resize_to[0]), Image.BILINEAR)

        img_tensor = TF.to_tensor(img)  # [1,H,W]

        stem = os.path.splitext(os.path.basename(image_path))[0]
        if epoch is None:
            file_prefix = f"{stem}"
        else:
            file_prefix = f"epoch_{epoch:03d}_{stem}"

        return self.inspect_single_image_and_save(
            img_tensor=img_tensor,
            epoch=epoch,
            save_dir=save_dir,
            save_input=save_input,
            file_prefix=file_prefix,
            verbose=verbose,
            learnable=learnable,
            degrade_mode=degrade_mode,
            degrade_strength=degrade_strength,
            degrade_num_select=degrade_num_select,
        )

    @torch.no_grad()
    def print_batch_strength_info(self, img: torch.Tensor, max_items: int = 4):
        """
        打印 batch 中前 max_items 个样本的 learnable 强度信息。
        img: [B,1,H,W]
        """
        info = self.get_strength_info(img)
        raw_strength = info["raw_strength"]
        norm_strength = info["norm_strength"]
        strength = info["strength"]

        b = img.size(0)
        n = min(b, max_items)

        print("\n===== Batch Strength Info =====")
        for i in range(n):
            print(f"[Sample {i}]")
            for j, t in enumerate(self.ng_types):
                print(
                    f"  {t:<12} | "
                    f"logit={raw_strength[i, j].item(): .6f} | "
                    f"norm={norm_strength[i, j].item(): .6f} | "
                    f"mapped={strength[i, j].item(): .6f}"
                )

    def get_lambda(self, lambda_max: float = 0.4):
        """
        如果你训练时还在用 lambda = lambda_max * sigmoid(log_lambda)
        可以直接调这个函数。
        """
        return lambda_max * torch.sigmoid(self.log_lambda)

def _reshape_strength(s, x):
    if not torch.is_tensor(s):
        s = torch.tensor(s, device=x.device, dtype=x.dtype)
    s = s.to(device=x.device, dtype=x.dtype)

    if s.dim() == 0:
        s = s.view(1, 1, 1, 1).expand(x.size(0), 1, 1, 1)
    elif s.dim() == 1:
        s = s.view(-1, 1, 1, 1)
    elif s.dim() == 2:
        s = s.view(s.size(0), s.size(1), 1, 1)
    else:
        raise ValueError(f"Unsupported strength shape: {s.shape}")

    return s


def gaussian_noise(x, s):
    s = _reshape_strength(s, x)
    std = GN_BASE + GN_SCALE * s
    noise = torch.randn_like(x)
    return x + noise * std


def stripe_noise(x, s):
    s = _reshape_strength(s, x)
    b, c, h, w = x.shape

    amp = SN_BASE + SN_SCALE * s
    col = torch.randn(b, 1, 1, w, device=x.device, dtype=x.dtype)
    noise = col.expand(b, 1, h, w).expand(b, c, h, w)
    return x + amp * noise


def shot_noise(x, s):
    s = _reshape_strength(s, x)
    x_clamped = torch.clamp(x, 0, 1)
    amp = SHOT_BASE + SHOT_SCALE * s
    noise = torch.randn_like(x_clamped) * torch.sqrt(torch.clamp(x_clamped, min=1e-6))
    return x_clamped + amp * noise


def optical_noise(x, s):
    s = _reshape_strength(s, x)
    b, c, h, w = x.shape

    hh = max(1, h // ON_DOWNSAMPLE)
    ww = max(1, w // ON_DOWNSAMPLE)

    low_noise = torch.randn(b, 1, hh, ww, device=x.device, dtype=x.dtype)
    low_noise = F.interpolate(low_noise, size=(h, w), mode="bilinear", align_corners=False)
    low_noise = low_noise.expand(b, c, h, w)

    amp = ON_BASE + ON_SCALE * s
    return x + amp * low_noise


def blur(x, s):
    s = _reshape_strength(s, x)
    alpha = torch.clamp(BLUR_BASE + BLUR_SCALE * s, 0.0, 1.0)

    pad = BLUR_POOL_KERNEL // 2
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    x_blur = F.avg_pool2d(x_pad, kernel_size=BLUR_POOL_KERNEL, stride=1)

    return x * (1.0 - alpha) + x_blur * alpha


def contrast(x, s):
    s = _reshape_strength(s, x)
    factor = torch.clamp(CONTRAST_BASE - CONTRAST_DECAY * s, min=CONTRAST_MIN, max=1.0)
    mean = x.mean(dim=(2, 3), keepdim=True)
    return mean + factor * (x - mean)


def resolution(x, s):
    s = _reshape_strength(s, x)
    alpha = torch.clamp(RESOLUTION_BASE + RESOLUTION_SCALE * s, 0.0, 1.0)

    b, c, h, w = x.shape
    h2 = max(1, int(h * RESOLUTION_DOWN_FACTOR))
    w2 = max(1, int(w * RESOLUTION_DOWN_FACTOR))

    x_low = F.interpolate(x, size=(h2, w2), mode="bilinear", align_corners=False)
    x_low = F.interpolate(x_low, size=(h, w), mode="bilinear", align_corners=False)

    return x * (1.0 - alpha) + x_low * alpha


_FUNC = {
    "GaussianNoise": gaussian_noise,
    "StripeNoise": stripe_noise,
    "ShotNoise": shot_noise,
    "OpticalNoise": optical_noise,
    "Blur": blur,
    "Contrast": contrast,
    "Resolution": resolution,
}