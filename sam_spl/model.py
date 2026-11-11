import torch
import torch.nn as nn
from base_model import *
from model_NG import *

class NgWrapperSamAdaptor(nn.Module):
    def __init__(self, detector: nn.Module, ng: nn.Module):
        super().__init__()
        self.detector = detector
        self.ng = ng

    def forward(self, x, return_ng_info=False):
        """
        Args:
            x: input images (B, C, H, W)
            return_ng_info: whether to return NG auxiliary outputs

        Returns:
            pred_logits
            (optional) ng_info
        """
        x_deg, ng_info = self.ng(x)

        pred_logits = self.detector(x_deg)

        if return_ng_info:
            return pred_logits, ng_info
        else:
            return pred_logits
    
def build_model(cfg):
    detector = make_adaptor(
        backbone_channel_list=cfg.backbone_channel_list,
        dense_low_channels=cfg.dense_low_channels,
        stages=cfg.stages,
        global_att_blocks=cfg.global_att_blocks,
        window_pos_embed_bkg_spatial_size=cfg.window_pos_embed_bkg_spatial_size,
        window_spec=cfg.window_spec,
        block=cfg.block,
        embed_dim=cfg.embed_dim,
        use_sam_decoder=cfg.use_sam_decoder,
        pe_inch=cfg.pe_inch,
        sam_ckpt_path=cfg.sam_ckpt_path,
    )

    ng = NG(cfg.ng)   
    model = NgWrapperSamAdaptor(detector, ng)
    return model