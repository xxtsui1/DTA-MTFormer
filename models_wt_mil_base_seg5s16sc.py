import torch
import torch.nn as nn
from functools import partial
from vt_weaktr_msk import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
from decoder import MaskTransformer
from segmenter import Segmenter
from utils_seg import padding,unpadding
from resnet_skip import ResNetV2, DecoderCup,SegmentationHead,DecoderAttCup_q,DecoderCupp
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
import math
import numpy as np
import scipy.stats as st

from AAF import *

__all__ = ['deit_small_WeakTr_patch16_224', 'deit_small_WeakT_AAF_AttnFeat_patch16_224', ]
def get_kernel(kernlen=3, nsig=6):    
    interval = (2*nsig+1.)/kernlen  
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)                                 
    kern1d = np.diff(st.norm.cdf(x))    
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))   
    kernel = kernel_raw/kernel_raw.sum()          
    return kernel

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'
    
class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1)
        return x

class WeakTr(VisionTransformer):
    def __init__(self, depth=12, num_heads=6, reduction=4, pool="avg", 
                 embed_dim=384, attn_feat=False, AdaptiveAttentionFusion=None, 
                 feat_reduction=None, *args, **kwargs):
        # timm>=1.0 VisionTransformer.__init__ no longer accepts these
        kwargs.pop('pretrained_cfg', None)
        kwargs.pop('pretrained_cfg_overlay', None)
        kwargs.pop('cache_dir', None)
        super().__init__(embed_dim=embed_dim, depth=depth, num_heads=num_heads, *args, **kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))

        self.head_conv = nn.Conv2d(self.embed_dim, self.num_classes, kernel_size=3, stride=1, padding=1)
        self.head_conv_fine = nn.Conv2d(self.embed_dim, self.num_classes, kernel_size=3, stride=1, padding=1)
        #self.head.apply(self._init_weights)
        self.head = nn.Linear(self.embed_dim, self.num_classes)

        self.flat = nn.Flatten(start_dim=2)

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        print(self.training)

        aaf_params = dict(channel=depth*num_heads, reduction=reduction, pool=pool)
        if feat_reduction is not None:
            aaf_params["feat_reduction"] = feat_reduction      
            aaf_params["feats_channel"] = embed_dim//num_heads       
            
        
        #instance learning for weight==========================
        self.L = 384
        self.D = 192
        self.K = 1
        self.abmil_attention4 = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K),
            Transpose(1, 2),
        )

        self.attn_feat = attn_feat
        self.unflat = nn.Unflatten(2, torch.Size([14, 14]))
        self.sm = nn.Sigmoid()
        self.up = nn.UpsamplingBilinear2d(scale_factor=16)
        kernel = get_kernel(kernlen=3,nsig=6) 
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)    
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        #resnetv2===================================
        self.mlp_seg = nn.Linear(196, 196*16) 
        self.a_postprocess = nn.Sequential(
        Transpose(1,2))
        self.act_postprocess = nn.Sequential(
        nn.Unflatten(2, torch.Size([14, 14])),
        nn.Conv2d(
            in_channels=self.embed_dim,
            out_channels=self.embed_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.ConvTranspose2d(
            in_channels=self.embed_dim,
            out_channels=self.embed_dim,
            kernel_size=4,
            stride=4,
            padding=0,
            bias=True,
            dilation=1,
            groups=1,
        ),
    )        
        self.fusion1 = nn.Sequential(
        nn.Conv2d(
            in_channels=self.embed_dim,
            out_channels=self.embed_dim//2,
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.ConvTranspose2d(
            in_channels=self.embed_dim//2,
            out_channels=self.embed_dim//4,
            kernel_size=4,
            stride=4,
            padding=0,
            bias=True,
            dilation=1,
            groups=1,
        ),
    )        

        self.segmentation_head = SegmentationHead(
            in_channels=self.embed_dim//4,
            out_channels=1,
            kernel_size=1,
        )
    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0:1]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]

        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[0]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)
    
    def forward_features(self, x, n=12):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        x = self.pos_drop(x)
        mask_all = []
        attn_feats = []

        for i, blk in enumerate(self.blocks):
            x, mask = blk(x,i)
            if i >=9:
                attn_feats.append(x[:, 1:])
                mask_all.append(mask)

        return x[:, 0:1], x[:, 1:], mask_all, attn_feats
    
    def data_normal(self, orign_data):
        d_min = orign_data.min()
        if d_min<0:
            orign_data += torch.abs(d_min)
            d_min = orign_data.min()
        d_max = orign_data.max()
        dst = d_max-d_min
        norm_data = (orign_data - d_min).true_divide(dst)
        return norm_data

    def forward(self, x, return_att=False, attention_type='fused'):
        batch = x.size(0)
        x_cls, x_patch, mask_all, attn_feats = self.forward_features(x)
        n, p, c = x_patch.shape

        #===========================================================
        mask_all = torch.stack(mask_all)
        mask_all = mask_all[:,:,:,:,1:] 
        mask_all = torch.mean(mask_all, dim=2) 
        mask_all_stack = mask_all
        mask_all = torch.mean(mask_all, dim=0) 
        mask_all = mask_all.reshape(batch,1,14,14)

        #===========================================
        x_patch = torch.reshape(x_patch , [n, int(p**0.5), int(p**0.5), c])  
        x_patch = x_patch.permute([0, 3, 1, 2])   
        x_patch = x_patch.contiguous()
        x_patch_mask=x_patch*mask_all
        x_patch = self.head_conv(x_patch)
        x_logits = self.avgpool(x_patch).squeeze(3).squeeze(2) 
        mask_avg = mask_all.clone() 
        mask_avg = F.conv2d(mask_avg, self.weight, padding=1)
        unc_loss=((1-mask_avg)*mask_avg).view(batch,-1).mean(-1)
        #================================================reshape
        cls_token_pred = self.head(x_cls.squeeze())#x_cls.mean(-1)

        #===========================================
        x_patch_seg_mask =self.mlp_seg(self.flat(x_patch_mask))
        x_patch_seg = torch.reshape(x_patch_seg_mask ,[n,self.embed_dim,14*4,14*4])
        x_fusion1 = self.fusion1(x_patch_seg)  
        masks = self.segmentation_head(x_fusion1) 
        #return x_logits
        return cls_token_pred,x_logits,unc_loss.mean(0),masks,mask_all#,mask_all_stack
 
@register_model
def deit_small_WeakTr_patch16_224_mil_base_seg5(pretrained=False, **kwargs):
    model = WeakTr(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), AdaptiveAttentionFusion=AAF, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def deit_small_WeakTr_AAF_AttnFeat_patch16_224(pretrained=False, **kwargs):
    model = WeakTr(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), AdaptiveAttentionFusion=AAF_AttnFeat,
        attn_feat=True, **kwargs)
    model.default_cfg = _cfg()
    return model