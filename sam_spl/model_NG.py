import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from albumentations import GaussNoise, Lambda, GaussianBlur, RandomBrightnessContrast, Compose
import os
import sys

# ------------------------------
# 这里假设你已经有 FeatureExtractor & conv_downsample 定义
# 直接复用你第二段代码里的 FeatureExtractor 和 conv_downsample
# ------------------------------

class FeatureExtractor(nn.Module):
    """
    把整张图片压缩为一个向量, 长度为 type_num * 5
    """
    def __init__(self, type_num):
        super().__init__()

        self.encoder_type = nn.Sequential(
            nn.Upsample(size=(256, 256), mode='bilinear'),  # [3, H, W] -> [3, 256, 256]
            nn.Conv2d(3, 16, 3, stride=2, padding=1),       # [3, 256, 256] -> [16, 128, 128]
            nn.LeakyReLU(0.2), nn.InstanceNorm2d(16, affine=True),
            *conv_downsample(16, 32, normalization=True),   # [16, 128, 128] -> [32, 64, 64]
            *conv_downsample(32, 64, normalization=True)    # [32, 64, 64] -> [64, 32, 32]
        )

        self.decoder_type = nn.Sequential(
            *conv_downsample(64, 128, normalization=True),  # [64, 32, 32] -> [128, 16, 16]
            *conv_downsample(128, 128), nn.Dropout(p=0.5),  # [128, 16, 16] -> [128, 8, 8]
            nn.Conv2d(128, type_num * 5, 8, padding=0),     # [128, 8, 8] -> [type_num*5, 1, 1]
        )

    def forward(self, img_input):
        fea = self.encoder_type(img_input)
        weights_type = self.decoder_type(fea)  # [B, type_num*5, 1, 1]
        return weights_type


def conv_downsample(in_filters, out_filters, normalization=False):
    layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1),
              nn.LeakyReLU(0.2)]
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
    return layers


# ----------------------------------------
# 辅助函数：Torch Tensor <-> Numpy uint8
# ----------------------------------------

def _tensor_to_uint8(image_tensor: torch.Tensor):
    """
    image_tensor: [C,H,W], float or uint8
    return: np.uint8 [H,W,C], 以及一些元信息用于还原
    """
    assert image_tensor.dim() == 3, f"expect [C,H,W], got {image_tensor.shape}"

    device = image_tensor.device
    dtype = image_tensor.dtype

    x = image_tensor.detach().cpu().clone()  # [C,H,W]
    # 判断是否在 [0,1] 区间
    x_min, x_max = float(x.min()), float(x.max())
    was_01 = x_max <= 1.0 + 1e-6

    if was_01:
        x = x * 255.0

    x = x.clamp(0, 255).to(torch.uint8)         # [C,H,W]
    x = x.permute(1, 2, 0).numpy()              # [H,W,C], uint8
    return x, device, dtype, was_01


def _uint8_to_tensor(img_np: np.ndarray, device, dtype, was_01: bool):
    """
    img_np: [H,W,C], uint8
    """
    x = torch.from_numpy(img_np).permute(2, 0, 1).to(device)  # [C,H,W]
    if was_01:
        x = x.float() / 255.0
        if dtype.is_floating_point:
            x = x.to(dtype)
    else:
        x = x.to(dtype)
    return x


def _apply_albu_transform(image_tensor: torch.Tensor, albu_transform, seed=None) -> torch.Tensor:
    """
    通用封装：把 [C,H,W] 的 torch tensor 丢给 albumentations 的 transform 处理
    保持 dtype / device 不变
    """
    img_np, device, dtype, was_01 = _tensor_to_uint8(image_tensor)
    # albumentations 接口：transform(image=...) -> dict
    aug = albu_transform(image=img_np)["image"]
    out_tensor = _uint8_to_tensor(aug, device, dtype, was_01)
    return out_tensor


# ----------------------------------------
# 使用第一段代码中的退化函数，封装成 NG 用的接口
# ----------------------------------------

# 假设你已经有：
# def stripe_noise(image, **params): ...
# def nonuniformity_optical(image, **params): ...
# def Noise(p=1): ...
# def LC(p=1): ...
# def Blur(p=1): ...

def stripe_noise(image, **params):
    if random.random() < 0.5:
        g = np.random.randn(1, image.shape[1]) * (np.random.rand() * 0.07 + 0.03)
        b = np.random.randn(1, image.shape[1]) * (np.random.rand() * 5)
    else:
        g = np.random.randn(image.shape[0], 1) * (np.random.rand() * 0.07 + 0.03)
        b = np.random.randn(image.shape[0], 1) * (np.random.rand() * 5)
    if len(image.shape) == 3:
        g = np.expand_dims(g, -1)
        b = np.expand_dims(b, -1)
    noise = image * g + b
    image = np.clip(image.astype("float32") + noise.astype("float32"), 0, 255).astype("uint8")
    return image

# def nonuniformity_optical(image, **params):
#     h, w = image.shape
#     noise = np.ones((h, w)).astype("float32")
#     idx_h = np.expand_dims(np.arange(1, h + 1), 1)
#     idx_w = np.expand_dims(np.arange(1, w + 1), 0)
#     delta = np.random.randint(15, 75 + 1)
#     ch = np.random.randint(h)
#     cw = np.random.randint(w)

#     p = (np.abs(idx_h - ch) ** 2 + np.abs(idx_w - cw) ** 2) ** 0.5
#     p /= np.max(p)
#     noise *= p
#     noise = np.cos(noise * np.pi / 2) ** 4
#     if len(image.shape) == 3:
#         noise = np.expand_dims(noise, -1)
#     if random.random() < 0.5:
#         image = np.clip(image.astype("float32") + noise.astype("float32") * delta, 0, 255).astype("uint8")
#     else:
#         image = np.clip(image.astype("float32") + (1 - noise.astype("float32")) * delta, 0, 255).astype("uint8")
#     return image

def Noise(p=1):
    return Compose([
        Lambda(image=nonuniformity_optical, p=1),
        Lambda(image=stripe_noise, p=1),
        GaussNoise(var_limit=(5.0 ** 2,  20.0 ** 2), p=1),
    ], p=p)


def LC(p=1):
    return RandomBrightnessContrast(brightness_limit=(0.2, 0.4), contrast_limit=(-0.8, -0.2), p=p)


def Blur(p=1):
    return Compose([
        GaussianBlur(blur_limit=(7, 23), sigma_limit=(1, 3), p=1)
    ], p=p)


def create_corrupt():
    dataset_dir = opt.dataset_dir + '/' + opt.dataset_name
    with open(dataset_dir + '/test.txt', 'r') as f:
        list = f.read().splitlines()

    tbar = tqdm(list)
    for img_name in tbar:
        img = Image.open((opt.dataset_dir + img_name).replace('//', '/')).convert('L')
        img_list = {'image': np.array(img, dtype=np.uint8)}
        img_list = LC()(**img_list)
        img_list = Blur()(**img_list)
        img_list = Noise()(**img_list)
        cor_img = Image.fromarray(img_list['image']).convert('L')

        if not os.path.exists(opt.save_dir + '/' + opt.dataset_name + '/'):
            os.makedirs(opt.save_dir + '/' + opt.dataset_name + '/')
        cor_img.save(opt.save_dir + '/' + opt.dataset_name + '/' + img_name)

def nonuniformity_optical(image, **params):
    """
    支持 2D (H, W) 和 3D (H, W, C) 图像的非均匀光学噪声
    - image: np.ndarray, uint8 or float32
    """
    # 1) 根据维度安全地取 H, W
    if image.ndim == 2:
        h, w = image.shape
        num_channels = None
    elif image.ndim == 3:
        h, w, num_channels = image.shape
    else:
        raise ValueError(f"nonuniformity_optical expects 2D or 3D array, got shape {image.shape}")

    # 2) 生成 [H, W] 的光照噪声 mask
    noise = np.ones((h, w), dtype="float32")
    idx_h = np.expand_dims(np.arange(1, h + 1), 1)
    idx_w = np.expand_dims(np.arange(1, w + 1), 0)
    delta = np.random.randint(15, 75 + 1)
    ch = np.random.randint(h)
    cw = np.random.randint(w)

    p = (np.abs(idx_h - ch) ** 2 + np.abs(idx_w - cw) ** 2) ** 0.5
    p /= np.max(p)
    noise *= p
    noise = np.cos(noise * np.pi / 2) ** 4  # [H, W]

    # 3) 若是彩色图，把 noise 扩展到 channel 维度 [H, W, 1]
    if image.ndim == 3:
        noise = np.expand_dims(noise, axis=-1)  # [H, W, 1]

    # 4) additive / inverse 模式
    img_f = image.astype("float32")
    if np.random.rand() < 0.5:
        img_corrupt = np.clip(img_f + noise * delta, 0, 255)
    else:
        img_corrupt = np.clip(img_f + (1.0 - noise) * delta, 0, 255)

    return img_corrupt.astype("uint8")

from albumentations import Lambda, Compose

def ng_lc(image_tensor: torch.Tensor, *args) -> torch.Tensor:
    """
    使用第一段代码中的 LC() 退化 (RandomBrightnessContrast)
    """
    albu_t = LC(p=1.0)
    return _apply_albu_transform(image_tensor, albu_t)

def ng_blur(image_tensor: torch.Tensor, *
            
            args) -> torch.Tensor:
    """
    使用第一段代码中的 Blur() 退化 (GaussianBlur)
    """
    albu_t = Blur(p=1.0)
    return _apply_albu_transform(image_tensor, albu_t)

def ng_noise(image_tensor: torch.Tensor, *args) -> torch.Tensor:
    """
    使用第一段代码中的 Noise() 退化
    内部包含 nonuniformity_optical + stripe_noise + GaussNoise
    """
    albu_t = Noise(p=1.0)
    return _apply_albu_transform(image_tensor, albu_t)

def ng_stripe_only(image_tensor: torch.Tensor, *args) -> torch.Tensor:
    """
    单独使用 stripe_noise 作为退化
    """
    albu_t = Compose([
        Lambda(image=stripe_noise, p=1.0)
    ], p=1.0)
    return _apply_albu_transform(image_tensor, albu_t)

def ng_nonuniform_only(image_tensor: torch.Tensor, *args) -> torch.Tensor:
    """
    单独使用 nonuniformity_optical 作为退化
    """
    albu_t = Compose([
        Lambda(image=nonuniformity_optical, p=1.0)
    ], p=1.0)
    return _apply_albu_transform(image_tensor, albu_t)

def ng_lc_blur_noise(image_tensor: torch.Tensor, *args) -> torch.Tensor:
    """
    按照你 create_corrupt 里的顺序:
    LC -> Blur -> Noise
    整体封装成一个组合退化
    """
    albu_t = Compose([
        Lambda(image=nonuniformity_optical, p=0),  # 占位，不用也行，只是示例
    ], p=1.0)  # 这里可以直接串 LC / Blur / Noise，
               # 但用 _apply_albu_transform 多次也可以，这里简单一点：

    # 为了和 create_corrupt 行为更接近，这里直接三次调用：
    x = ng_lc(image_tensor)
    x = ng_blur(x)
    x = ng_noise(x)
    return x


# 名字到退化函数的映射字典（供 NG 使用）
_NAME_TO_NG_FUNC_CORRUPT = {
    "LC": ng_lc,
    "Blur": ng_blur,
    "Noise": ng_noise,
    "StripeNoise": ng_stripe_only,
    "NonuniformOptical": ng_nonuniform_only,
    "LC_Blur_Noise": ng_lc_blur_noise,
}


# ----------------------------------------
# 新版 NG 网络：结构和原来的 NG 一样，只是退化函数换成第一段代码里的
# ----------------------------------------

class NG(nn.Module):
    """
    使用第一段 create_corrupt 代码中的退化函数的 NG 网络

    - ng_types: 例如 ["LC", "Blur", "Noise", "StripeNoise", "NonuniformOptical"]
      必须出现在 _NAME_TO_NG_FUNC_CORRUPT 字典里
    """
    def __init__(self, ng_types: list) -> None:
        super().__init__()

        self.type_num = len(ng_types)
        self.feature_extractor = FeatureExtractor(self.type_num)
        self.ng_types = ng_types

        # 做一个检查，防止名字写错
        for name in self.ng_types:
            assert name in _NAME_TO_NG_FUNC_CORRUPT, f"Unknown ng_type: {name}"

    def forward(self, img_input: torch.Tensor):
        """
        img_input: [B, 3, H, W]，建议为 float32，范围 [0,1] 或 [0,255] 都可以
        """
        batch_size = img_input.size(0)

        # [B, type_num*5, 1, 1]
        weights_type = self.feature_extractor(img_input)
        # -> [B, 5, type_num]
        weights_type = weights_type.view(batch_size, 5, self.type_num)

        img_batch = []
        for i, weight_type in enumerate(weights_type):
            # weight_type: [5, type_num]
            single_img = img_input[i]  # [3,H,W]

            # 对每一行（每一轮）做 softmax，得到 type_num 维度上的概率分布
            weight_type = torch.softmax(weight_type, dim=1)  # [5, type_num]

            seed = random.random()  # 目前没有显式用到，只是保持接口一致

            # 共 5 轮退化
            for w_t in weight_type:  # w_t: [type_num]
                # 对每种退化函数做一次，然后加权求和
                degraded = 0
                for ind, ng_type in enumerate(self.ng_types):
                    func = _NAME_TO_NG_FUNC_CORRUPT[ng_type]
                    degraded_img = func(single_img, seed)  # [3,H,W]
                    degraded = degraded + w_t[ind] * degraded_img

                single_img = degraded  # 作为下一轮的输入

            img_batch.append(single_img.unsqueeze(0))

        img_out = torch.cat(img_batch, dim=0)  # [B, 3, H, W]
        return img_out
