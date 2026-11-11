# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math

# # ============================================================
# # Feature Extractor（保持你原来的结构）
# # ============================================================

# def conv_downsample(in_filters, out_filters, normalization=False):
#     layers = [
#         nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1),
#         nn.LeakyReLU(0.2)
#     ]
#     if normalization:
#         layers.append(nn.InstanceNorm2d(out_filters, affine=True))
#     return layers


# class FeatureExtractor(nn.Module):
#     def __init__(self, type_num):
#         super().__init__()

#         self.encoder_type = nn.Sequential(
#             nn.Upsample(size=(256, 256), mode="bilinear", align_corners=False),
#             nn.Conv2d(3, 16, 3, stride=2, padding=1),
#             nn.LeakyReLU(0.2),
#             nn.InstanceNorm2d(16, affine=True),
#             *conv_downsample(16, 32, normalization=True),
#             *conv_downsample(32, 64, normalization=True),
#         )

#         self.decoder_type = nn.Sequential(
#             *conv_downsample(64, 128, normalization=True),
#             *conv_downsample(128, 128),
#             nn.Dropout(p=0.5),
#             nn.Conv2d(128, type_num * 5, 8),
#         )

#     def forward(self, x):
#         x = self.encoder_type(x)
#         x = self.decoder_type(x)
#         return x


# # ============================================================
# # 基础工具
# # ============================================================

# def _value_range(x):
#     return 255.0 if x.max() > 1.5 else 1.0


# def _gaussian_kernel(k, sigma, device):
#     ax = torch.arange(k, device=device) - k // 2
#     kernel = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
#     kernel = kernel / kernel.sum()
#     return kernel


# # ============================================================
# # 退化函数（纯 Tensor）
# # ============================================================

# def ng_lc(x: torch.Tensor, *args):
#     scale = _value_range(x)

#     brightness = torch.empty(1, device=x.device).uniform_(0.05, 0.15) * scale
#     contrast = torch.empty(1, device=x.device).uniform_(-0.3, 0.1)

#     mean = x.mean(dim=(1, 2), keepdim=True)
#     x = (x - mean) * (1 + contrast) + mean + brightness
#     return x.clamp(0, scale)


# def ng_blur(x: torch.Tensor, *args):
#     scale = _value_range(x)
#     C, H, W = x.shape

#     k = int(torch.randint(7, 24, (1,)).item()) | 1
#     sigma = torch.empty(1).uniform_(1.0, 3.0).item()

#     kernel_1d = _gaussian_kernel(k, sigma, x.device)
#     kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
#     kernel_2d = kernel_2d.expand(C, 1, k, k)

#     x = x.unsqueeze(0)
#     x = F.conv2d(x, kernel_2d, padding=k // 2, groups=C)
#     return x.squeeze(0).clamp(0, scale)


# def ng_down_up(x: torch.Tensor, *args):
#     C, H, W = x.shape
#     s = int(torch.randint(2, 5, (1,)).item())
#     y = F.avg_pool2d(x.unsqueeze(0), s, s)
#     y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
#     return y.squeeze(0)


# def ng_stripe_noise(x: torch.Tensor, *args):
#     scale = _value_range(x)
#     C, H, W = x.shape

#     if torch.rand(1) < 0.5:
#         g = torch.randn(1, W, device=x.device)
#         b = torch.randn(1, W, device=x.device)
#         g = g.unsqueeze(0)
#         b = b.unsqueeze(0)
#     else:
#         g = torch.randn(H, 1, device=x.device)
#         b = torch.randn(H, 1, device=x.device)
#         g = g.unsqueeze(0)
#         b = b.unsqueeze(0)

#     g = g * torch.rand(1, device=x.device).mul(0.07).add(0.03)
#     b = b * torch.rand(1, device=x.device).mul(1.5)

#     return (x + x * g + b).clamp(0, scale)


# def ng_nonuniform(x: torch.Tensor, *args):
#     scale = _value_range(x)
#     C, H, W = x.shape

#     yy, xx = torch.meshgrid(
#         torch.arange(H, device=x.device),
#         torch.arange(W, device=x.device),
#         indexing="ij",
#     )

#     ch = torch.randint(0, H, (1,), device=x.device)
#     cw = torch.randint(0, W, (1,), device=x.device)

#     dist = ((yy - ch) ** 2 + (xx - cw) ** 2).float().sqrt()
#     dist = dist / dist.max()

#     mask = torch.cos(dist * math.pi / 2) ** 4
#     delta = torch.randint(5, 25, (1,), device=x.device).float()

#     if torch.rand(1) < 0.5:
#         noise = mask * delta
#     else:
#         noise = (1 - mask) * delta

#     return (x + noise.unsqueeze(0)).clamp(0, scale)


# def ng_gaussian_noise(x: torch.Tensor, *args):
#     scale = _value_range(x)
#     std = torch.empty(1, device=x.device).uniform_(1.0, 6.0)
#     return (x + torch.randn_like(x) * std).clamp(0, scale)


# def ng_noise(x: torch.Tensor, *args):
#     x = ng_nonuniform(x)
#     x = ng_stripe_noise(x)
#     x = ng_gaussian_noise(x)
#     return x


# # ============================================================
# # 映射表（接口名保持）
# # ============================================================

# _NAME_TO_NG_FUNC = {
#     "LC": ng_lc,
#     "Blur": ng_blur,
#     "Noise": ng_noise,
#     "StripeNoise": ng_stripe_noise,
#     "NonuniformOptical": ng_nonuniform,
#     "DownUp": ng_down_up,
# }


# # ============================================================
# # NG 主体（接口完全保持）
# # ============================================================

# class NG(nn.Module):
#     def __init__(self, ng_types: list):
#         super().__init__()

#         self.type_num = len(ng_types)
#         self.feature_extractor = FeatureExtractor(self.type_num)
#         self.ng_types = ng_types

#         for name in self.ng_types:
#             assert name in _NAME_TO_NG_FUNC, f"Unknown ng_type: {name}"

#     def forward(self, img_input: torch.Tensor):
#         B = img_input.size(0)

#         weights = self.feature_extractor(img_input)
#         weights = weights.view(B, 5, self.type_num)

#         outputs = []

#         for i in range(B):
#             x = img_input[i]
#             w = torch.softmax(weights[i], dim=1)

#             for w_t in w:
#                 degraded = torch.zeros_like(x)
#                 for idx, ng_type in enumerate(self.ng_types):
#                     degraded = degraded + w_t[idx] * _NAME_TO_NG_FUNC[ng_type](x)
#                 x = degraded

#             outputs.append(x.unsqueeze(0))

#         return torch.cat(outputs, dim=0)








##这个能跑通非学习降质（随机降质）

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def conv_downsample(in_filters, out_filters, normalization=False):
    layers = [
        nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1),
        nn.LeakyReLU(0.2)
    ]
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
    return layers

class FeatureExtractor(nn.Module):
    def __init__(self, type_num):
        super().__init__()

        self.encoder_type = nn.Sequential(
            nn.Upsample(size=(256, 256), mode="bilinear", align_corners=False),
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(16, affine=True),
            *conv_downsample(16, 32, normalization=True),
            *conv_downsample(32, 64, normalization=True),
        )

        self.decoder_type = nn.Sequential(
            *conv_downsample(64, 128, normalization=True),
            *conv_downsample(128, 128),
            nn.Dropout(p=0.5),
            nn.Conv2d(128, type_num * 5, 8),
        )

    def forward(self, x):
        x = self.encoder_type(x)
        x = self.decoder_type(x)
        return x


def _value_range(x):
    return 255.0 if x.max() > 1.5 else 1.0


def _gaussian_kernel(k, sigma, device):
    ax = torch.arange(k, device=device) - k // 2
    kernel = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel


def ng_lc(x: torch.Tensor, *args):
    """
    极轻度亮度 / 对比度扰动
    - 支持 normalized 输入
    - 防止全黑 / 方差塌缩
    """
    brightness = torch.empty(1, device=x.device).uniform_(0.003, 0.01)
    contrast = torch.empty(1, device=x.device).uniform_(-0.03, 0.03)

    # per-channel statistics（关键）
    mean = x.mean(dim=(1, 2), keepdim=True)
    std  = x.std(dim=(1, 2), keepdim=True) + 1e-6

    # 轻度扰动
    x = (x - mean) * (1 + contrast) + mean + brightness

    # ===== 稳定下限：基于原始分布 =====
    min_val = mean - 2.5 * std
    max_val = mean + 2.5 * std

    return torch.clamp(x, min=min_val, max=max_val)




def ng_blur(x: torch.Tensor, *args):
    """轻度高斯模糊"""
    scale = _value_range(x)
    C, H, W = x.shape

    k = int(torch.randint(3, 7, (1,), device=x.device).item()) | 1
    sigma = torch.empty(1, device=x.device).uniform_(0.3, 0.8).item()

    kernel_1d = _gaussian_kernel(k, sigma, x.device)
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel_2d = kernel_2d.expand(C, 1, k, k)

    x = x.unsqueeze(0)
    x = F.conv2d(x, kernel_2d, padding=k // 2, groups=C)
    return x.squeeze(0).clamp(0, scale)


def ng_down_up(x: torch.Tensor, *args):
    """轻度分辨率退化"""
    C, H, W = x.shape
    s = int(torch.randint(2, 3, (1,), device=x.device).item())
    y = F.avg_pool2d(x.unsqueeze(0), s, s)
    y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
    return y.squeeze(0)


def ng_gaussian_noise(x: torch.Tensor, *args):
    """极轻度高斯噪声（接近红外热噪声）"""
    scale = _value_range(x)
    std = torch.empty(1, device=x.device).uniform_(0.001, 0.005) * scale
    return (x + torch.randn_like(x) * std).clamp(0, scale)

def ng_stripe(x: torch.Tensor, stage=0):
    """
    红外条纹噪声（方向性 + 低频）
    """
    C, H, W = x.shape
    device = x.device

    if stage == 0:
        amp = 0.002
    elif stage == 1:
        amp = 0.005
    else:
        amp = 0.01

    if torch.rand(1) < 0.5:
        # horizontal stripe
        stripe = torch.randn(1, H, 1, device=device)
        stripe = F.avg_pool1d(
            stripe.unsqueeze(0),
            kernel_size=9,
            stride=1,
            padding=4
        ).squeeze(0)
        stripe = stripe.expand(C, H, W)
    else:
        # vertical stripe
        stripe = torch.randn(1, 1, W, device=device)
        stripe = F.avg_pool1d(
            stripe,
            kernel_size=9,
            stride=1,
            padding=4
        )
        stripe = stripe.expand(C, H, W)

    x = x + amp * stripe
    return x.clamp(0, 1)

# ============================================================
# 映射表（接口名保持）
# ============================================================

_NAME_TO_NG_FUNC = {
    "LC": ng_lc,
    "Blur": ng_blur,
    "Noise": ng_gaussian_noise,
    "DownUp": ng_down_up,
    "Stripe": ng_stripe,
}


# ============================================================
# NG 主体（接口完全保持）
# ============================================================

class NG(nn.Module):
    """
    NG performs **single, low-intensity degradation per image**.
    A hard selection strategy is adopted to avoid excessive compound degradations.
    """

    def __init__(self, ng_types: list):
        super().__init__()

        self.type_num = len(ng_types)
        self.feature_extractor = FeatureExtractor(self.type_num)
        self.ng_types = ng_types

        for name in self.ng_types:
            assert name in _NAME_TO_NG_FUNC, f"Unknown ng_type: {name}"

    def forward(self, img_input: torch.Tensor):
        """
        Args:
            img_input: (B, C, H, W)
        Returns:
            degraded images with exactly ONE degradation applied per sample
        """
        B = img_input.size(0)

        weights = self.feature_extractor(img_input)
        weights = weights.view(B, 5, self.type_num)

        outputs = []

        for i in range(B):
            x = img_input[i]
            w = torch.softmax(weights[i], dim=1)

            w0 = w[0]

            idx = torch.argmax(w0).item()
            ng_type = self.ng_types[idx]

            x = _NAME_TO_NG_FUNC[ng_type](x)
            outputs.append(x.unsqueeze(0))

        return torch.cat(outputs, dim=0)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# from typing import Dict, Any


# # ============================================================
# # Feature Extractor（保持你原结构）
# # ============================================================

# def conv_downsample(in_filters, out_filters, normalization=False):
#     layers = [
#         nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1),
#         nn.LeakyReLU(0.2),
#     ]
#     if normalization:
#         layers.append(nn.InstanceNorm2d(out_filters, affine=True))
#     return layers


# class FeatureExtractor(nn.Module):
#     def __init__(self, type_num):
#         super().__init__()

#         self.encoder_type = nn.Sequential(
#             nn.Upsample(size=(256, 256), mode="bilinear", align_corners=False),
#             nn.Conv2d(3, 16, 3, stride=2, padding=1),
#             nn.LeakyReLU(0.2),
#             nn.InstanceNorm2d(16, affine=True),
#             *conv_downsample(16, 32, normalization=True),
#             *conv_downsample(32, 64, normalization=True),
#         )

#         self.decoder_type = nn.Sequential(
#             *conv_downsample(64, 128, normalization=True),
#             *conv_downsample(128, 128),
#             nn.Dropout(p=0.5),
#             nn.Conv2d(128, type_num * 5, 8),
#         )

#     def forward(self, x):
#         x = self.encoder_type(x)
#         x = self.decoder_type(x)
#         return x


# # ============================================================
# # 工具函数
# # ============================================================

# def _value_range(x):
#     return 255.0 if x.max() > 1.5 else 1.0


# def _gaussian_kernel(k, sigma, device):
#     ax = torch.arange(k, device=device) - k // 2
#     kernel = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
#     kernel = kernel / kernel.sum()
#     return kernel


# def _rand_uniform(cfg, device):
#     return torch.empty(1, device=device).uniform_(cfg["min"], cfg["max"])


# # ============================================================
# # 退化函数（严格 config 驱动）
# # ============================================================

# def ng_lc(x: torch.Tensor, cfg: Dict[str, Any]):
#     scale = _value_range(x)

#     alpha = _rand_uniform(cfg["alpha"], x.device)
#     beta = _rand_uniform(cfg["beta"], x.device) * scale

#     mean = x.mean(dim=(1, 2), keepdim=True)
#     x = (x - mean) * alpha + mean + beta
#     return x.clamp(0, scale)


# def ng_blur(x: torch.Tensor, cfg: Dict[str, Any]):
#     scale = _value_range(x)
#     C, H, W = x.shape

#     k = int(cfg["kernel_size"][0])
#     sigma = _rand_uniform(cfg["sigma"], x.device)

#     kernel_1d = _gaussian_kernel(k, sigma.item(), x.device)
#     kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
#     kernel_2d = kernel_2d.expand(C, 1, k, k)

#     x = F.conv2d(
#         x.unsqueeze(0),
#         kernel_2d,
#         padding=k // 2,
#         groups=C,
#     )

#     return x.squeeze(0).clamp(0, scale)


# def ng_noise(x: torch.Tensor, cfg: Dict[str, Any]):
#     scale = _value_range(x)

#     sigma = _rand_uniform(cfg["sigma"], x.device) * scale
#     noise = torch.randn_like(x) * sigma

#     return (x + noise).clamp(0, scale)


# # ============================================================
# # 映射表
# # ============================================================

# _NAME_TO_NG_FUNC = {
#     "LC": ng_lc,
#     "Blur": ng_blur,
#     "Noise": ng_noise,
# }


# # ============================================================
# # NG 主体（yaml 驱动）
# # ============================================================

# class NG(nn.Module):
#     def __init__(self, ng_config):
#         super().__init__()

#         self.ng_config = ng_config

#         # ===== 策略部分 =====
#         self.strategy_cfg = ng_config["degradation"]
#         self.ng_types = self.strategy_cfg["ng_types"]
#         self.type_num = len(self.ng_types)

#         self.apply_prob = self.strategy_cfg.get("apply_prob", 1.0)
#         self.random_order = self.strategy_cfg.get("random_order", False)

#         # ===== 算子参数（平级）=====
#         self.op_cfg = ng_config   # LC / Blur / Noise 在这里

#         self.feature_extractor = FeatureExtractor(self.type_num)

#         for name in self.ng_types:
#             assert name in _NAME_TO_NG_FUNC, f"Unknown ng_type: {name}"
#             assert name in self.op_cfg, f"Missing config for {name}"



#     def forward(self, img_input: torch.Tensor):
#         B = img_input.size(0)

#         weights = self.feature_extractor(img_input)
#         weights = weights.view(B, 5, self.type_num)

#         outputs = []

#         for i in range(B):
#             x = img_input[i]
#             w = torch.softmax(weights[i], dim=1)

#             for w_t in w:
#                 degraded = torch.zeros_like(x)

#                 for idx, ng_type in enumerate(self.ng_types):
#                     cfg = self.op_cfg[ng_type]

#                     # 每种退化自己的 prob
#                     if torch.rand(1, device=x.device) > cfg.get("prob", 1.0):
#                         continue

#                     degraded = degraded + w_t[idx] * _NAME_TO_NG_FUNC[ng_type](x, cfg)

#                 x = degraded

#             outputs.append(x.unsqueeze(0))

#         return torch.cat(outputs, dim=0)
