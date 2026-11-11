import torch
import torch.nn as nn
import torch.nn.functional as F
from guided_filter_pytorch.guided_filter import ConvGuidedFilter

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        # 输入 2 个通道（avg 和 max），输出 1 个通道的空间注意力图
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

        # 全局平均池化：每个通道一个数
        gap = F.adaptive_avg_pool2d(x, 1).view(B, C)   # [B, C]

        # MLP 生成通道权重
        w = self.mlp(gap)                              # [B, C]
        w = torch.sigmoid(w).view(B, C, 1, 1)          # [B, C, 1, 1]

        return w
class EfficientSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        res = torch.cat([avg_out, max_out], dim=1)
        res = self.conv(res)
        return self.sigmoid(res)
    
class HighLowPyramid(nn.Module):
    """
    基于 ConvGuidedFilter 的高低频分解 + 可学习 alpha 融合：
        low  = ConvGuidedFilter(x, x, x)
        high = x - low
        S    = alpha * high + (1 - alpha) * low
    """
    def __init__(self, channels=3, out_channels=32, radius=2, sizes=((64,64), (32,32), (16,16), (8,8))):
        super().__init__()
        self.guided_filter = ConvGuidedFilter(radius=radius, norm=nn.BatchNorm2d)
        # self.spatial_attn = SpatialAttention()
        self.spatial_attn=EfficientSpatialAttention()
        self.channel_attn = ChannelAttention(channels=out_channels, reduction=8)
        self.sizes = sizes

        self.pre_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=True
        )
        self.pre_act = nn.ReLU(inplace=True)

        self.proj = nn.Conv2d(
            in_channels=channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=True
        )


    def forward(self, x):
        
        low = self.guided_filter(x, x, x)  
        high = x - low

        x = self.pre_act(self.pre_conv(x))

        e = self.spatial_attn(x)
        
        w =  self.channel_attn(x)

        low  = self.proj(low)                  
        high = self.proj(high) 

        high = e * high
        low = (1.0 - e) * low * w


        low_list = []
        high_list = []
        for x,y in self.sizes:
            low_list.append(F.interpolate(low, size=(x, y), mode='area'))
            high_list.append(F.interpolate(high, size=(x, y), mode='area'))

        return low_list, high_list
