import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.vst import video_swin_t, video_swin_s, video_swin_b, video_swin_l, get_device
from utils.ST import SwinTransformer
from utils.mlp import TabularMLP
from configs.config import Config

class DeepSTEMI(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.cine_net = video_swin_b(pretrained=False,
                num_classes=Config.num_classes,
                device=Config.device
        )
        
        self.static_swin = SwinTransformer(
            img_size=Config.t2_shape[0],
            patch_size=4,
            in_chans=1,
            embed_dim=Config.swin_embed_dim,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7
        )
        
        self.tabular_mlp = TabularMLP(
            input_dim=4,  
            hidden_dim=Config.mlp_hidden_dim,
            output_dim=Config.mlp_output_dim
        )
        
        self.classifier = nn.Linear(
            Config.video_swin_embed_dim * 2 ** (len(self.static_swin.depths) - 1),
            Config.num_classes
        )
    
    def forward(self, cine, t2, lge, tabular):
        cine_feat = self.cine_net(cine)  
        t2_feat = self.static_swin.forward_features(t2)  
        lge_feat = self.static_swin.forward_features(lge)  
        tabular_feat = self.tabular_mlp(tabular) 
        fused_feat = cine_feat + t2_feat + lge_feat + tabular_feat
        logits = self.classifier(fused_feat)
        
        return logits