import math

from os.path import join as pjoin
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import densenet121 


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

class ClassificationHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.in_channels//4, self.num_classes)
        self.conv1 = Conv2dReLU(
            in_channels,
            in_channels//2,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        self.conv2 = Conv2dReLU(
            in_channels//2,
            in_channels//4,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        ) 

    def forward(self, x):
        x=self.conv1(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        x=self.conv2(x)
        x = self.avgpool(x)
        x=x.squeeze(3)
        x=x.squeeze(2)
        x = self.fc(x)
        return x
    

    
class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y

    def load_from(self, weights, n_block, n_unit):
        conv1_weight = np2th(weights[pjoin(n_block, n_unit, "conv1/kernel")], conv=True)
        conv2_weight = np2th(weights[pjoin(n_block, n_unit, "conv2/kernel")], conv=True)
        conv3_weight = np2th(weights[pjoin(n_block, n_unit, "conv3/kernel")], conv=True)

        gn1_weight = np2th(weights[pjoin(n_block, n_unit, "gn1/scale")])
        gn1_bias = np2th(weights[pjoin(n_block, n_unit, "gn1/bias")])

        gn2_weight = np2th(weights[pjoin(n_block, n_unit, "gn2/scale")])
        gn2_bias = np2th(weights[pjoin(n_block, n_unit, "gn2/bias")])

        gn3_weight = np2th(weights[pjoin(n_block, n_unit, "gn3/scale")])
        gn3_bias = np2th(weights[pjoin(n_block, n_unit, "gn3/bias")])

        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)

        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))

        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))

        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))

        if hasattr(self, 'downsample'):
            proj_conv_weight = np2th(weights[pjoin(n_block, n_unit, "conv_proj/kernel")], conv=True)
            proj_gn_weight = np2th(weights[pjoin(n_block, n_unit, "gn_proj/scale")])
            proj_gn_bias = np2th(weights[pjoin(n_block, n_unit, "gn_proj/bias")])

            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))

class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width)) for i in range(2, block_units[0] + 1)],
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2)) for i in range(2, block_units[1] + 1)],
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) for i in range(2, block_units[2] + 1)],
                ))),
        ]))

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body)-1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i+1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1]
    


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)

class DecoderCup(nn.Module):
    def __init__(self, head_channels,hidden_size,decoder_channels,n_skip,skip_channels):
        super().__init__()
        head_channels = 512
        self.n_skip=n_skip
        self.conv_more = Conv2dReLU(
            hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if n_skip != 0:
            skip_channels =skip_channels
            for i in range(4-n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, features=None):
        #B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        #h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        #x = hidden_states.permute(0, 2, 1)
        #x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x
    

class DecoderAttCup1(nn.Module):
    def __init__(self, head_channels,hidden_size,decoder_channels,n_skip,skip_channels):
        super().__init__()
        head_channels = 512
        self.n_skip=n_skip
        self.conv_more = Conv2dReLU(
            hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        self.patch_embeddings = nn.ModuleList([nn.Conv2d(in_channels=384,out_channels=head_channels,kernel_size=1),
                                 nn.Conv2d(in_channels=head_channels,out_channels=head_channels//2,kernel_size=1),
                                 nn.Conv2d(in_channels=head_channels//2,out_channels=head_channels//4,kernel_size=1),
                                 nn.Conv2d(in_channels=head_channels//4,out_channels=head_channels//8,kernel_size=1)])
        decoder_channels = decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if n_skip != 0:
            skip_channels =skip_channels
            for i in range(4-n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.multihead_attn = nn.ModuleList([nn.MultiheadAttention(embed_dim=head_channels, num_heads=1),nn.MultiheadAttention(embed_dim=head_channels//2, num_heads=1),
                                             nn.MultiheadAttention(embed_dim=head_channels//4, num_heads=1),nn.MultiheadAttention(embed_dim=head_channels//8, num_heads=1)])
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, q_fea, features=None):
        #B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        #h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        #x = hidden_states.permute(0, 2, 1)
        #x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        print(len(self.blocks))
        for i, decoder_block in enumerate(self.blocks): 
            bs, c, h, w = x.shape           
            x = x.flatten(2).permute(2, 0, 1)
            q_fea = self.patch_embeddings[i](q_fea)
            q_fea =q_fea.flatten(2).permute(2, 0, 1)
            if features is not None:
                skip = features[i] if (i < self.n_skip) else None
            else:
                skip = None
            q_fea,att_weight = self.multihead_attn[i](query=q_fea, key=x,value=x)
            x = torch.bmm(att_weight, x.permute(1, 0, 2))
            x = x.view(bs, c, h, w)
            q_fea = q_fea.permute(1, 2, 0).view(bs, c, h, w)
            q_fea = self.up(q_fea)
            x = decoder_block(x, skip=skip)
            print(q_fea.shape)
        return x


class DecoderAttCup(nn.Module):
    def __init__(self, head_channels,hidden_size,decoder_channels,n_skip,skip_channels):
        super().__init__()
        head_channels = 512
        self.n_skip=n_skip
        self.conv_more = nn.ModuleList([Conv2dReLU(384,head_channels,kernel_size=3, padding=1,use_batchnorm=True,),
                                        Conv2dReLU(384,head_channels//2,kernel_size=3, padding=1,use_batchnorm=True,),
                                        Conv2dReLU(384,head_channels//4,kernel_size=3, padding=1,use_batchnorm=True,),
                                        Conv2dReLU(384,head_channels//8,kernel_size=3, padding=1,use_batchnorm=True,)])
        self.patch_embeddings = nn.ModuleList([nn.Conv2d(hidden_size,out_channels=384,kernel_size=1),
                                 nn.Conv2d(in_channels=head_channels//2,out_channels=384,kernel_size=1),
                                 nn.Conv2d(in_channels=head_channels//4,out_channels=384,kernel_size=1),
                                 nn.Conv2d(in_channels=head_channels//8,out_channels=384,kernel_size=1)])
        decoder_channels = decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if n_skip != 0:
            skip_channels =skip_channels
            for i in range(4-n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.multihead_attn = nn.ModuleList([nn.MultiheadAttention(embed_dim=384, num_heads=1),nn.MultiheadAttention(embed_dim=384, num_heads=1),
                                             nn.MultiheadAttention(embed_dim=384, num_heads=1),nn.MultiheadAttention(embed_dim=384, num_heads=1)])

    def forward(self, x, q_fea, features=None):
        #B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        #h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        #x = hidden_states.permute(0, 2, 1)
        #x = x.contiguous().view(B, hidden, h, w)
        #x = self.conv_more(x)
        q_fea =q_fea.flatten(2).permute(2, 0, 1)
        for i, decoder_block in enumerate(self.blocks):
            x = self.patch_embeddings[i](x) 
            #x_=x
            if i >0:
                x = nn.UpsamplingBilinear2d(scale_factor=(0.5**i))(x)  
            bs, c, h, w = x.shape        
            x = x.flatten(2).permute(2, 0, 1)            
            q_fea,att_weight = self.multihead_attn[i](query=q_fea, key=x,value=x)
            x = torch.bmm(att_weight, x.permute(1, 0, 2))
            x = x.view(bs, c, h, w)
            if i>0:
                x = nn.UpsamplingBilinear2d(scale_factor=(2**i))(x)
            #x=x+x_
            #q_fea = q_fea.permute(1, 2, 0).view(bs, c, h, w)
            if features is not None:
                skip = features[i] if (i < self.n_skip) else None
            else:
                skip = None
            x = self.conv_more[i](x)            
            x = decoder_block(x, skip=skip)            
        return x
    

class DecoderAttCup_qq(nn.Module):#return q
    def __init__(self, head_channels,hidden_size,decoder_channels,n_skip,skip_channels):
        super().__init__()
        head_channels = 512
        self.n_skip=n_skip
        self.conv_more = nn.ModuleList([Conv2dReLU(384,head_channels,kernel_size=3, padding=1,use_batchnorm=True,),
                                        Conv2dReLU(384,head_channels//2,kernel_size=3, padding=1,use_batchnorm=True,),
                                        Conv2dReLU(384,head_channels//4,kernel_size=3, padding=1,use_batchnorm=True,),
                                        Conv2dReLU(384,head_channels//8,kernel_size=3, padding=1,use_batchnorm=True,)])
        self.patch_embeddings = nn.ModuleList([nn.Conv2d(hidden_size,out_channels=384,kernel_size=1),
                                 nn.Conv2d(in_channels=head_channels//2,out_channels=384,kernel_size=1),
                                 nn.Conv2d(in_channels=head_channels//4,out_channels=384,kernel_size=1),
                                 nn.Conv2d(in_channels=head_channels//8,out_channels=384,kernel_size=1)])
        decoder_channels = decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if n_skip != 0:
            skip_channels =skip_channels
            for i in range(4-n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.multihead_attn = nn.ModuleList([nn.MultiheadAttention(embed_dim=384, num_heads=1),nn.MultiheadAttention(embed_dim=384, num_heads=1),
                                             nn.MultiheadAttention(embed_dim=384, num_heads=1),nn.MultiheadAttention(embed_dim=384, num_heads=1)])

    def forward(self, x, q_fea, features=None):
        #B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        #h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        #x = hidden_states.permute(0, 2, 1)
        #x = x.contiguous().view(B, hidden, h, w)
        #x = self.conv_more(x)
        q_fea =q_fea.flatten(2).permute(2, 0, 1)
        for i, decoder_block in enumerate(self.blocks):
            x = self.patch_embeddings[i](x) 
            #x_=x
            if i >0:
                x = nn.UpsamplingBilinear2d(scale_factor=(0.5**i))(x)  
            bs, c, h, w = x.shape        
            x = x.flatten(2).permute(2, 0, 1)            
            q_fea,att_weight = self.multihead_attn[i](query=q_fea, key=x,value=x)
            x = torch.bmm(att_weight, x.permute(1, 0, 2))
            x = x.view(bs, c, h, w)
            if i>0:
                x = nn.UpsamplingBilinear2d(scale_factor=(2**i))(x)
            #x=x+x_
            #q_fea = q_fea.permute(1, 2, 0).view(bs, c, h, w)
            if features is not None:
                skip = features[i] if (i < self.n_skip) else None
            else:
                skip = None
            x = self.conv_more[i](x)            
            x = decoder_block(x, skip=skip)            
        return x,q_fea.permute(1, 2, 0).view(bs, c, h, w)

class DecoderAttCup_x(nn.Module):#add the attented x
    def __init__(self, head_channels,hidden_size,decoder_channels,n_skip,skip_channels):
        super().__init__()
        head_channels = 512
        self.n_skip=n_skip
        self.conv_more = nn.ModuleList([Conv2dReLU(384,head_channels,kernel_size=3, padding=1,use_batchnorm=True,),
                                        Conv2dReLU(384,head_channels//2,kernel_size=3, padding=1,use_batchnorm=True,),
                                        Conv2dReLU(384,head_channels//4,kernel_size=3, padding=1,use_batchnorm=True,),
                                        Conv2dReLU(384,head_channels//8,kernel_size=3, padding=1,use_batchnorm=True,)])
        self.patch_embeddings = nn.ModuleList([nn.Conv2d(hidden_size,out_channels=384,kernel_size=1),
                                 nn.Conv2d(in_channels=head_channels//2,out_channels=384,kernel_size=1),
                                 nn.Conv2d(in_channels=head_channels//4,out_channels=384,kernel_size=1),
                                 nn.Conv2d(in_channels=head_channels//8,out_channels=384,kernel_size=1)])
        decoder_channels = decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if n_skip != 0:
            skip_channels =skip_channels
            for i in range(4-n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.multihead_attn = nn.ModuleList([nn.MultiheadAttention(embed_dim=384, num_heads=1),nn.MultiheadAttention(embed_dim=384, num_heads=1),
                                             nn.MultiheadAttention(embed_dim=384, num_heads=1),nn.MultiheadAttention(embed_dim=384, num_heads=1)])

    def forward(self, x, q_fea, features=None):
        #B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        #h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        #x = hidden_states.permute(0, 2, 1)
        #x = x.contiguous().view(B, hidden, h, w)
        #x = self.conv_more(x)
        q_fea =q_fea.flatten(2).permute(2, 0, 1)
        for i, decoder_block in enumerate(self.blocks):
            x = self.patch_embeddings[i](x) 
            x_=x
            if i >0:
                x = nn.UpsamplingBilinear2d(scale_factor=(0.5**i))(x)  
            bs, c, h, w = x.shape        
            x = x.flatten(2).permute(2, 0, 1)            
            q_fea,att_weight = self.multihead_attn[i](query=q_fea, key=x,value=x)
            x = torch.bmm(att_weight, x.permute(1, 0, 2))
            x = x.view(bs, c, h, w)
            if i>0:
                x = nn.UpsamplingBilinear2d(scale_factor=(2**i))(x)
            x=x+x_
            #q_fea = q_fea.permute(1, 2, 0).view(bs, c, h, w)
            if features is not None:
                skip = features[i] if (i < self.n_skip) else None
            else:
                skip = None
            x = self.conv_more[i](x)            
            x = decoder_block(x, skip=skip)            
        return x


class DecoderAttCup_q(nn.Module):#return q
    def __init__(self, head_channels,hidden_size,decoder_channels,n_skip,skip_channels):
        super().__init__()
        head_channels = 512
        self.n_skip=n_skip
        self.conv_more = Conv2dReLU(384,head_channels,kernel_size=3, padding=1,use_batchnorm=True,)
        self.patch_embeddings = nn.Conv2d(hidden_size,out_channels=384,kernel_size=1)
        decoder_channels = decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if n_skip != 0:
            skip_channels =skip_channels
            for i in range(4-n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=384, num_heads=1)

    def forward(self, x, q_fea, features=None):
        #B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        #h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        #x = hidden_states.permute(0, 2, 1)
        #x = x.contiguous().view(B, hidden, h, w)
        #x = self.conv_more(x)
        q_fea =q_fea.flatten(2).permute(2, 0, 1)
        x = self.patch_embeddings(x) 
        bs, c, h, w = x.shape        
        x = x.flatten(2).permute(2, 0, 1)            
        q_fea,att_weight = self.multihead_attn(query=q_fea, key=x,value=x)
        x = torch.bmm(att_weight, x.permute(1, 0, 2))
        x = x.view(bs, c, h, w)
        x = self.conv_more(x)      
        for i, decoder_block in enumerate(self.blocks):            
            #x=x+x_
            #q_fea = q_fea.permute(1, 2, 0).view(bs, c, h, w)
            if features is not None:
                skip = features[i] if (i < self.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)            
        return x,q_fea.permute(1, 2, 0).view(bs, c, h, w)
    

class DecoderAttCup_q5(nn.Module):#return q
    def __init__(self, head_channels,hidden_size,decoder_channels,n_skip,skip_channels):
        super().__init__()
        head_channels = 512
        self.n_skip=n_skip
        self.conv_more = Conv2dReLU(384,head_channels,kernel_size=3, padding=1,use_batchnorm=True,)
        self.patch_embeddings = nn.Conv2d(hidden_size,out_channels=384,kernel_size=1)
        decoder_channels = decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if n_skip != 0:
            skip_channels =skip_channels
            for i in range(4-n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=384, num_heads=1)

    def forward(self, x, q_fea, features=None):
        #B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        #h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        #x = hidden_states.permute(0, 2, 1)
        #x = x.contiguous().view(B, hidden, h, w)
        #x = self.conv_more(x)
        q_fea =q_fea.flatten(2).permute(2, 0, 1)
        x = self.patch_embeddings(x) 
        bs, c, h, w = x.shape        
        x = x.flatten(2).permute(2, 0, 1)            
        q_fea,att_weight = self.multihead_attn(query=q_fea, key=x,value=x)
        x = torch.bmm(att_weight, x.permute(1, 0, 2))
        x = x.view(bs, c, h, w)
        x = self.conv_more(x)      
        for i, decoder_block in enumerate(self.blocks):            
            #x=x+x_
            #q_fea = q_fea.permute(1, 2, 0).view(bs, c, h, w)
            if features is not None:
                skip = features[i] if (i < self.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)            
        return x,q_fea.permute(1, 2, 0).view(bs, c, h, w)

    
class DecoderAttCup_q4(nn.Module):#return q
    def __init__(self, head_channels,hidden_size,decoder_channels,n_skip,skip_channels):
        super().__init__()
        head_channels = 512
        self.n_skip=n_skip
        self.conv_more = Conv2dReLU(1024,head_channels,kernel_size=3, padding=1,use_batchnorm=True,)
        self.patch_embeddings = nn.Conv2d(hidden_size,out_channels=384,kernel_size=1)
        decoder_channels = decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if n_skip != 0:
            skip_channels =skip_channels
            for i in range(4-n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=384, num_heads=1)
        self.multihead_attn2 = nn.MultiheadAttention(embed_dim=384, num_heads=1)

    def forward(self, x, q_fea, features=None):
        #B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        #h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        #x = hidden_states.permute(0, 2, 1)
        #x = x.contiguous().view(B, hidden, h, w)
        #x = self.conv_more(x)
        x1=x
        q_fea =q_fea.flatten(2).permute(2, 0, 1)
        x = self.patch_embeddings(x) 
        bs, c, h, w = x.shape        
        x = x.flatten(2).permute(2, 0, 1)            
        q_fea,att_weight = self.multihead_attn(query=q_fea, key=x,value=x)
        x = self.conv_more(x1)      
        for i, decoder_block in enumerate(self.blocks):            
            #x=x+x_
            #q_fea = q_fea.permute(1, 2, 0).view(bs, c, h, w)
            if features is not None:
                skip = features[i] if (i < self.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)            
        return x,q_fea.permute(1, 2, 0).view(bs, c, h, w)
    

class DecoderAttCup_q3(nn.Module):#return q
    def __init__(self, head_channels,hidden_size,decoder_channels,n_skip,skip_channels):
        super().__init__()
        head_channels = 512
        self.n_skip=n_skip
        self.conv_more = Conv2dReLU(384,head_channels,kernel_size=3, padding=1,use_batchnorm=True,)
        self.patch_embeddings = nn.Conv2d(hidden_size,out_channels=384,kernel_size=1)
        decoder_channels = decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if n_skip != 0:
            skip_channels =skip_channels
            for i in range(4-n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=384, num_heads=1)

    def forward(self, x, q_fea, features=None):
        #B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        #h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        #x = hidden_states.permute(0, 2, 1)
        #x = x.contiguous().view(B, hidden, h, w)
        #x = self.conv_more(x)
        q_fea =q_fea.flatten(2).permute(2, 0, 1)
        x = self.patch_embeddings(x) 
        bs, c, h, w = x.shape        
        x = x.flatten(2).permute(2, 0, 1)  
        q_fea1=q_fea          
        q_fea,att_weight = self.multihead_attn(query=q_fea, key=x,value=x)
        q_fea=q_fea+q_fea1
        x = torch.bmm(att_weight, x.permute(1, 0, 2))
        x = x.view(bs, c, h, w)
        x = self.conv_more(x)      
        for i, decoder_block in enumerate(self.blocks):            
            #x=x+x_
            #q_fea = q_fea.permute(1, 2, 0).view(bs, c, h, w)
            if features is not None:
                skip = features[i] if (i < self.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)            
        return x,q_fea.permute(1, 2, 0).view(bs, c, h, w)
    
class DecoderAttCup_q1(nn.Module):#return q
    def __init__(self, head_channels,hidden_size,decoder_channels,n_skip,skip_channels):
        super().__init__()
        head_channels = 512
        self.n_skip=n_skip
        self.conv_more = Conv2dReLU(384,head_channels,kernel_size=3, padding=1,use_batchnorm=True,)
        self.patch_embeddings = nn.Conv2d(hidden_size,out_channels=384,kernel_size=1)
        decoder_channels = decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if n_skip != 0:
            skip_channels =skip_channels
            for i in range(4-n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=384, num_heads=1)

    def forward(self, x, q_fea, features=None):
        #B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        #h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        #x = hidden_states.permute(0, 2, 1)
        #x = x.contiguous().view(B, hidden, h, w)
        #x = self.conv_more(x)
        q_fea =q_fea.flatten(2).permute(2, 0, 1)
        x = self.patch_embeddings(x) 
        bs, c, h, w = x.shape        
        x = x.flatten(2).permute(2, 0, 1)            
        x,att_weight = self.multihead_attn(query=x, key=q_fea,value=q_fea)
        q_fea = torch.bmm(att_weight, q_fea.permute(1, 0, 2))
        x = x.permute(1,2, 0).view(bs, c, h, w)
        x = self.conv_more(x)      
        for i, decoder_block in enumerate(self.blocks):            
            #x=x+x_
            #q_fea = q_fea.permute(1, 2, 0).view(bs, c, h, w)
            if features is not None:
                skip = features[i] if (i < self.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)            
        return x,q_fea.permute(0,2,1).view(bs, c, h, w)


class DecoderAttCup_q2(nn.Module):#return q
    def __init__(self, head_channels,hidden_size,decoder_channels,n_skip,skip_channels):
        super().__init__()
        head_channels = 512
        self.n_skip=n_skip
        self.conv_more = Conv2dReLU(384,head_channels,kernel_size=3, padding=1,use_batchnorm=True,)
        self.patch_embeddings = nn.Conv2d(hidden_size,out_channels=384,kernel_size=1)
        decoder_channels = decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if n_skip != 0:
            skip_channels =skip_channels
            for i in range(4-n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=384, num_heads=1)

    def forward(self, x, q_fea, features=None):
        #B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        #h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        #x = hidden_states.permute(0, 2, 1)
        #x = x.contiguous().view(B, hidden, h, w)
        #x = self.conv_more(x)
        q_fea =q_fea.flatten(2).permute(2, 0, 1)
        x = self.patch_embeddings(x) 
        bs, c, h, w = x.shape        
        x = x.flatten(2).permute(2, 0, 1)            
        x,att_weight = self.multihead_attn(query=x, key=q_fea,value=q_fea)
        q_fea1 = q_fea.permute(1,0,2)   
        q_fea = torch.bmm(att_weight, q_fea.permute(1, 0, 2))+q_fea1
        x = x.permute(1,2, 0).view(bs, c, h, w)
        x = self.conv_more(x)      
        for i, decoder_block in enumerate(self.blocks):            
            #x=x+x_
            #q_fea = q_fea.permute(1, 2, 0).view(bs, c, h, w)
            if features is not None:
                skip = features[i] if (i < self.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)            
        return x,q_fea.permute(0,2,1).view(bs, c, h, w)






class Cos_Attn_no(nn.Module):
    def __init__(self, inplanes):
        super(Cos_Attn_no, self).__init__()

        self.sub_sample = False

        self.in_channels = inplanes
        self.inter_channels = None

        if self.inter_channels is None:
            self.inter_channels = self.in_channels // 4
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d
        bn = nn.BatchNorm2d


        self.g1 = conv_nd(in_channels=384, out_channels=self.inter_channels,
                         kernel_size=1)
        self.g2 = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1)

        self.W = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1),
            bn(self.in_channels)
        )

        self.K = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels//2,
                    kernel_size=1),
            bn(self.in_channels//2)
        )

        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

        self.Q = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.inter_channels,
                    kernel_size=1),
            bn(self.inter_channels)
        )
        nn.init.constant_(self.Q[1].weight, 0)
        nn.init.constant_(self.Q[1].bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1)
        self.phi = conv_nd(in_channels=384, out_channels=self.inter_channels,
                           kernel_size=1)

        self.concat_project = nn.Sequential(
            nn.Conv2d(self.inter_channels * 2, 1, 1, 1, 0, bias=False),
            nn.ReLU()
        )

        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)

    def forward(self, detect, aim):

        batch_size, _, height_a, width_a = aim.shape
        batch_size, _, height_d, width_d = detect.shape

        #####################################find aim image similar object ####################################################
        detect_t=self.g1(detect)
        d_x = detect_t.view(batch_size, self.inter_channels, -1)
        d_x = d_x.permute(0, 2, 1).contiguous()
        a_x = self.g2(aim).view(batch_size, self.inter_channels, -1)
        a_x = a_x.permute(0, 2, 1).contiguous()

        theta_x = self.theta(aim).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(detect).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = F.softmax(f,-1)#f / N
        non_aim = torch.matmul(f_div_C, d_x)
        non_aim = non_aim.permute(0, 2, 1).contiguous()
        non_aim = non_aim.view(batch_size, self.inter_channels, height_a, width_a)
        #non_aim = self.W(non_aim)
        #non_aim = non_aim + aim
        non_aim = self.K(non_aim)
        ##################################### Response in chaneel weight ####################################################    
        return non_aim#non_det, act_det, act_aim, c_weight



class DecoderAttCup_q6(nn.Module):#return q
    def __init__(self, head_channels,hidden_size,decoder_channels,n_skip,skip_channels):
        super().__init__()
        head_channels = 512
        self.n_skip=n_skip
        decoder_channels = decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if n_skip != 0:
            skip_channels =skip_channels
            for i in range(4-n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.multihead_attn = Cos_Attn_no(hidden_size)

    def forward(self, x, q_fea, features=None):
        x = self.multihead_attn(q_fea,x)
        for i, decoder_block in enumerate(self.blocks):            
            #x=x+x_
            #q_fea = q_fea.permute(1, 2, 0).view(bs, c, h, w)
            if features is not None:
                skip = features[i] if (i < self.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)            
        return x,q_fea

class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)

class DecoderCupps(nn.Module):
    def __init__(self, head_channels,hidden_size,decoder_channels,n_skip,skip_channels):
        super().__init__()
        head_channels = 512
        self.conv_more = Conv2dReLU(
            hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if n_skip != 0:
            skip_channels = self.skip_channels
            for i in range(4-self.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(hidden)), int(np.sqrt(hidden))
        x = hidden_states
        x = x.contiguous().view(B, n_patch, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x



class DecoderCupp(nn.Module):
    def __init__(self, head_channels,hidden_size,decoder_channels,n_skip,skip_channels):
        super().__init__()
        head_channels = 512
        self.conv_more = Conv2dReLU(
            hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if n_skip != 0:
            skip_channels = self.skip_channels
            for i in range(4-self.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(hidden)), int(np.sqrt(hidden))
        x = hidden_states
        x = x.contiguous().view(B, n_patch, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x
    
class Bottleneck(nn.Module):
    #每个stage维度中扩展的倍数
    extention=4
    def __init__(self,inplanes,planes,stride,downsample=None):
        '''

        :param inplanes: 输入block的之前的通道数
        :param planes: 在block中间处理的时候的通道数
                planes*self.extention:输出的维度
        :param stride:
        :param downsample:
        '''
        super(Bottleneck, self).__init__()

        self.conv1=nn.Conv2d(inplanes,planes,kernel_size=1,stride=stride,bias=False)
        self.bn1=nn.BatchNorm2d(planes)

        self.conv2=nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(planes)

        self.conv3=nn.Conv2d(planes,planes*self.extention,kernel_size=1,stride=1,bias=False)
        self.bn3=nn.BatchNorm2d(planes*self.extention)

        self.relu=nn.ReLU(inplace=True)

        #判断残差有没有卷积
        self.downsample=downsample
        self.stride=stride

    def forward(self,x):
        #参差数据
        residual=x

        #卷积操作
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)

        out=self.conv2(out)
        out=self.bn2(out)
        out=self.relu(out)

        out=self.conv3(out)
        out=self.bn3(out)
        out=self.relu(out)

        #是否直连（如果Indentity blobk就是直连；如果Conv2 Block就需要对残差边就行卷积，改变通道数和size
        if self.downsample is not None:
            residual=self.downsample(x)

        #将残差部分和卷积部分相加
        out=out+residual
        out=self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self,block,layers,num_class):
        #inplane=当前的fm的通道数
        self.inplane=64
        super(ResNet, self).__init__()

        #参数
        self.block=block
        self.layers=layers

        #stem的网络层
        self.conv1=nn.Conv2d(3,self.inplane,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(self.inplane)
        self.relu=nn.ReLU()
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        #64,128,256,512指的是扩大4倍之前的维度，即Identity Block中间的维度
        self.stage1=self.make_layer(self.block,64,layers[0],stride=1)
        self.stage2=self.make_layer(self.block,128,layers[1],stride=2)
        self.stage3=self.make_layer(self.block,256,layers[2],stride=2)
        self.stage4=self.make_layer(self.block,512,layers[3],stride=2)

        #后续的网络
        self.avgpool=nn.AvgPool2d(7)
        self.fc=nn.Linear(512*block.extention,num_class)

    def forward(self,x):

        #stem部分：conv+bn+maxpool
        features=[]
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        features.append(out)
        out=self.maxpool(out)

        #block部分
        out=self.stage1(out)
        features.append(out)
        out=self.stage2(out)
        features.append(out)
        out=self.stage3(out)
        seg=out
        out=self.stage4(out)

        #分类
        out=self.avgpool(out)
        out=torch.flatten(out.clone(),1)
        out=self.fc(out)

        return seg,features[::-1],out

    def make_layer(self,block,plane,block_num,stride=1):
        '''
        :param block: block模板
        :param plane: 每个模块中间运算的维度，一般等于输出维度/4
        :param block_num: 重复次数
        :param stride: 步长
        :return:
        '''
        block_list=[]
        #先计算要不要加downsample
        downsample=None
        if(stride!=1 or self.inplane!=plane*block.extention):
            downsample=nn.Sequential(
                nn.Conv2d(self.inplane,plane*block.extention,stride=stride,kernel_size=1,bias=False),
                nn.BatchNorm2d(plane*block.extention)
            )

        # Conv Block输入和输出的维度（通道数和size）是不一样的，所以不能连续串联，他的作用是改变网络的维度
        # Identity Block 输入维度和输出（通道数和size）相同，可以直接串联，用于加深网络
        #Conv_block
        conv_block=block(self.inplane,plane,stride=stride,downsample=downsample)
        block_list.append(conv_block)
        self.inplane=plane*block.extention

        #Identity Block
        for i in range(1,block_num):
            block_list.append(block(self.inplane,plane,stride=1))

        return nn.Sequential(*block_list)