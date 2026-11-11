import torch
import torch.nn as nn
import torch.nn.functional as F

class EfficientCrossAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, sr_ratio=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.sr_ratio = sr_ratio
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        if sr_ratio > 1:
            # 修改点：改用卷积 + GroupNorm。
            # GroupNorm(1, hidden_dim) 对图像特征来说比 LayerNorm 更方便，效果一致。
            self.sr = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=sr_ratio, stride=sr_ratio),
                nn.GroupNorm(1, hidden_dim), # 1个组等价于对整个通道做LayerNorm
                nn.GELU()
            )
        else:
            self.sr = nn.Identity()

    def forward(self, query_feat, key_feat, value_feat):
        B, N, C = query_feat.shape
        H = W = int(N**0.5)

        # 1. Projections
        q = self.q_proj(query_feat).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        k_img = key_feat.transpose(1, 2).reshape(B, C, H, W)
        v_img = value_feat.transpose(1, 2).reshape(B, C, H, W)

        if self.sr_ratio > 1:
            k_img = self.sr(k_img)
            v_img = self.sr(v_img)
        
        k = self.k_proj(k_img.flatten(2).transpose(1, 2)).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(v_img.flatten(2).transpose(1, 2)).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
            x = F.scaled_dot_product_attention(q, k, v, scale=self.scale)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.out_proj(x)
        
        return x, None
    
    
    
class MultiScaleAdapter(nn.Module):
    def __init__(self, in_channels, hl_channels=32, hidden_dim=None, num_heads_t=4, num_heads_s=4):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_norm = nn.LayerNorm(in_channels) 
        
        self.wd = nn.Conv2d(in_channels, hidden_dim, 1)
        self.proj_high = nn.Conv2d(hl_channels, hidden_dim, 1)
        self.proj_low = nn.Conv2d(hl_channels, hidden_dim, 1)

        self.temporal_attn = nn.MultiheadAttention(hidden_dim, num_heads_t, batch_first=True)
        #self.spatial_attn = nn.MultiheadAttention(hidden_dim, num_heads_s, batch_first=True)
        self.spatial_attn = EfficientCrossAttention(hidden_dim,num_heads=num_heads_s)
        fused_channels = hidden_dim * 3
        self.conv3 = nn.Conv2d(fused_channels, hidden_dim, 3, padding=1)
        self.conv5 = nn.Conv2d(fused_channels, hidden_dim, 5, padding=2)
        self.conv7 = nn.Conv2d(fused_channels, hidden_dim, 7, padding=3)
        self.act = nn.GELU()
        self.wp = nn.Conv2d(hidden_dim, in_channels, 1)
        self.fuse_dh = nn.Sequential(
            nn.Conv2d(2 * hidden_dim, hidden_dim, kernel_size=3, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, bias=True)
        )
        self.spatial_fuse = nn.Sequential(
            nn.Conv2d(2 * hidden_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1)
        )
        # self.gate_offset = nn.Parameter(torch.tensor([-0.05])) #relu门控
        # nn.init.constant_(self.wp.weight, 0)
        # nn.init.constant_(self.wp.bias, 0.01)
    def forward(self, f, f_high, f_low):
        B, C, H, W = f.shape
        f_d = self.wd(f)
        fh = self.proj_high(f_high)
        fl = self.proj_low(f_low)
        fd_seq = f_d.view(B, self.hidden_dim, -1).permute(0, 2, 1)
        fh_seq = fh.view(B, self.hidden_dim, -1).permute(0, 2, 1)
        pair_t = torch.stack([fd_seq, fh_seq], dim=2).view(B * H * W, 2, self.hidden_dim)
        out_t, _ = self.temporal_attn(pair_t, pair_t, pair_t)  # [B*H*W, 2, C']
        out_t = out_t.mean(dim=1)                              # [B*H*W, 2, C'] -> [B*H*W, C']
        out_t = out_t.view(B, H * W, self.hidden_dim).permute(0, 2, 1).view(B, self.hidden_dim, H, W)
        f_h_prime = out_t
        fd_sp = f_d.view(B, self.hidden_dim, -1).permute(0, 2, 1)
        fl_sp = fl.view(B, self.hidden_dim, -1).permute(0, 2, 1)
        out_s, _ = self.spatial_attn(fd_sp, fl_sp, fl_sp)
        f_l_prime = out_s.permute(0, 2, 1).view(B, self.hidden_dim, H, W)
        fused = torch.cat([f_d, f_h_prime, f_l_prime], dim=1)
        o3 = self.conv3(fused)
        o5 = self.conv5(fused)
        o7 = self.conv7(fused)
        f_agg = (o3 + o5 + o7 + f_d) / 4.0
        f_agg = self.act(f_agg)
        out = self.wp(f_agg)              # [B,C,H,W]
        out = out.permute(0,2,3,1).contiguous()
        out = self.out_norm(out)
        out = out.permute(0,3,1,2).contiguous()
        # out = F.relu(out)
        return out + f

    # def forward(self, f, f_high, f_low):
    #     B, C, H, W = f.shape
    #     # print(f"[rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}] f.shape = {f.shape}")
    #     # 1. 对输入特征, 高低频特征降维
    #     f_d = self.wd(f)
    #     # print(f"[rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}] f_d.shape = {f_d.shape}")

    #     fh = self.proj_high(f_high) 
    #     fl = self.proj_low(f_low) 
    #     # fh = fh + F.avg_pool2d(fh, kernel_size=3, stride=1, padding=1)
    #     # fl = fl + F.avg_pool2d(fl, kernel_size=3, stride=1, padding=1)
    #     # print(f"[rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}] fh.shape = {fh.shape}")
    #     # print(f"[rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}] fl.shape = {fl.shape}")
    #     ###-----下采样对齐-----###
    #     # _, _, Hh, Wh = fh.shape
    #     # if H != Hh or W != Wh:
    #     #     f_d = F.interpolate(
    #     #         f_d,
    #     #         size=(Hh, Wh),
    #     #         mode="area"
    #     #     )
    #     #     H, W = Hh, Wh  # 更新空间尺度
    #     ###-------------------###
    #     ###-----上采样对齐-----###
    #     # _, _, Hh, Wh = fh.shape
    #     # if H != Hh or W != Wh:
    #     #     fh = F.interpolate(
    #     #         fh,
    #     #         size=(H, W),
    #     #         mode="bilinear",
    #     #         align_corners=False
    #     #     )
    #     ###-------------------###
    #     # print(f"[rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}] test:out.shape = {f_d.shape}")
    #     # print(f"[rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}] test:out.shape = {fh.shape}")
    #     ##----------------kkkk---#
    #     #2. fh与f_d通过时间注意力融合
    #     fd_seq = f_d.view(B, self.hidden_dim, -1).permute(0, 2, 1)
    #     # print(f"[rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}] test:out.shape = {fd_seq.shape}")
    #     fh_seq = fh.view(B, self.hidden_dim, -1).permute(0, 2, 1)
    #     # print(f"[rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}] test:out.shape = {fh_seq.shape}")
    #     # pair_t: [B*H*W, 2, C']
    #     pair_t = torch.stack([fd_seq, fh_seq], dim=2).view(B * H * W, 2, self.hidden_dim)
    #     out_t, _ = self.temporal_attn(pair_t, pair_t, pair_t)  # [B*H*W, 2, C']
    #     out_t = out_t.mean(dim=1)                              # [B*H*W, 2, C'] -> [B*H*W, C']
    #     # reshape回feature map
    #     out_t = out_t.view(B, H * W, self.hidden_dim).permute(0, 2, 1).view(B, self.hidden_dim, H, W)   
    #     f_h_prime = out_t
    #     ##----------------kkkk---#
    #     # pair = torch.cat([f_d, fh], dim=1)   # [B, 2C', H, W]
    #     # f_h_prime = self.fuse_dh(pair)       # [B, C', H, W]
    #     ##----------------下面是原版--##
    #     # # 3. fl与f_d通过空间注意力融合
    #     fd_sp = f_d.view(B, self.hidden_dim, -1).permute(0, 2, 1)
    #     fl_sp = fl.view(B, self.hidden_dim, -1).permute(0, 2, 1)
    #     out_s, _ = self.spatial_attn(fd_sp, fl_sp, fl_sp)
    #     f_l_prime = out_s.permute(0, 2, 1).view(B, self.hidden_dim, H, W)

    #     ###-上面是原版——————————————————#
    #     # # [B, C, H, W] -> [B, C]
    #     # fd_c = f_d.mean(dim=(2, 3))   # 全局描述
    #     # fl_c = fl.mean(dim=(2, 3))

    #     # # [B, C] -> [B, 1, C]
    #     # fd_c = fd_c.unsqueeze(1)
    #     # fl_c = fl_c.unsqueeze(1)

    #     # # Cross-attention on channel tokens
    #     # out_c, _ = self.spatial_attn(fd_c, fl_c, fl_c)  # 注意：attention 仍然是 cross 的

    #     # # [B, 1, C] -> [B, C, 1, 1]
    #     # f_l_prime = out_c.squeeze(1).unsqueeze(-1).unsqueeze(-1)

    #     # # broadcast 回空间
    #     # f_l_prime = f_l_prime.expand(-1, -1, H, W)
    #     ###上面是尝试####
    #     fused = torch.cat([f_d, f_h_prime, f_l_prime], dim=1)

    #     o3 = self.conv3(fused)
    #     o5 = self.conv5(fused)
    #     o7 = self.conv7(fused)

    #     f_agg = (o3 + o5 + o7 + f_d) / 4.0
    #     f_agg = self.act(f_agg)
    #     out = self.wp(f_agg)              # [B,C,H,W]
    #     out = out.permute(0,2,3,1).contiguous()
    #     out = self.out_norm(out)
    #     out = out.permute(0,3,1,2).contiguous()
    #     # print(f"gamma:",self.gamma)
    #     # print("after",f"min={out.min().item():.6f}, "f"max={out.max().item():.6f}")
    #     # print(f"[rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}] out.shape = {out.shape}")
    #     # print(f"[rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}] f.shape = {f.shape}")
    #     ###------上采样对齐-----###
    #     # _, _, H, W = f.shape
    #     # _, _, h, w = out.shape

    #     # if h != H or w != W:
    #     #     out = F.interpolate(
    #     #         out,
    #     #         size=(H, W),
    #     #         mode="area",
    #     #     )
    #     ###-------------------###
    #     # print(f"[rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}] after out.shape = {out.shape}")
    #     # print(f"[rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}] after f.shape = {f.shape}")
    #     out = F.relu(out)
    #     # return f
    #     return  f + out


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class MultiScaleAdapter(nn.Module):
#     def __init__(self, in_channels, hl_channels=32, hidden_dim=None, num_heads_t=4, num_heads_s=4):
#         super().__init__()
#         self.in_channels = in_channels
#         self.hidden_dim = hidden_dim if hidden_dim is not None else in_channels // 2
        
#         self.out_norm = nn.LayerNorm(in_channels)
        
#         self.wd = nn.Conv2d(in_channels, self.hidden_dim, 1)
#         self.proj_high = nn.Conv2d(hl_channels, self.hidden_dim, 1)
#         self.proj_low = nn.Conv2d(hl_channels, self.hidden_dim, 1)
#         self.act=nn.GELU()

#         self.high_freq_gate = nn.Sequential(
#             nn.Conv2d(self.hidden_dim * 2, self.hidden_dim, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(self.hidden_dim),
#             nn.ReLU(),
#             nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1), 
#             nn.Sigmoid() 
#         )

#         self.low_freq_gate = nn.Sequential(
#             nn.Conv2d(self.hidden_dim * 2, self.hidden_dim, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(self.hidden_dim),
#             nn.ReLU(),
#             nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1),
#             nn.Sigmoid()
#         )

#         fused_channels = self.hidden_dim * 3
#         self.conv3 = nn.Conv2d(fused_channels, self.hidden_dim, 3, padding=1)
#         self.conv5 = nn.Conv2d(fused_channels, self.hidden_dim, 5, padding=2)

#         self.wp = nn.Conv2d(self.hidden_dim, in_channels, 1)
#         nn.init.constant_(self.wp.weight, 0)
#         nn.init.constant_(self.wp.bias, 0.01)

#     def forward(self, f, f_high, f_low):
#         B, C, H, W = f.shape
        

#         f_d = self.wd(f)             # Main Feature
#         fh = self.proj_high(f_high)  # High Freq (含噪声)
#         fl = self.proj_low(f_low)    # Low Freq
#         combined_h = torch.cat([f_d, fh], dim=1) 
#         gate_h = self.high_freq_gate(combined_h)
#         f_h_prime = fh * gate_h 
#         combined_l = torch.cat([f_d, fl], dim=1)
#         gate_l = self.low_freq_gate(combined_l)
#         f_l_prime = fl * gate_l
#         fused = torch.cat([f_d, f_h_prime, f_l_prime], dim=1)
#         o3 = self.conv3(fused)
#         o5 = self.conv5(fused)
#         f_agg = (o3 + o5 + f_d) / 3.0
#         f_agg = self.act(f_agg)
#         out = self.wp(f_agg)
#         out = out.permute(0, 2, 3, 1).contiguous()
#         out = self.out_norm(out)
#         out = out.permute(0, 3, 1, 2).contiguous()
#         out = F.relu(out)
        
#         return out + f
