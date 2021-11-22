#DLASR
import os
import sys
import torch
import math
import numpy as np
import logging
import cv2
import os
import shutil
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import functools
import copy
from functools import partial, reduce
import itertools
from collections import OrderedDict
from . import block as B
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=True)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=True)
def pixelUnshuffle(x, r=1):
    b, c, h, w = x.size()
    out_chl = c * (r ** 2)
    out_h = h // r
    out_w = w // r
    x = x.view(b, c, out_h, r, out_w, r)
    out = x.permute(0, 1, 3, 5, 2, 4).contiguous().view(b, out_chl, out_h, out_w)
    return out
def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class ResBlock(nn.Module):
    def __init__(self, nf, stride=1, downsample=None, res_scale=1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = conv3x3(nf, nf, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(nf,nf)
    def forward(self, x):
        x1 = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out * self.res_scale + x1
        return out
class IFE(nn.Module):
    def __init__(self, n_feats):
        super(IFE, self).__init__()
        self.conv_head = conv3x3(3, n_feats)    
        self.RBs = nn.ModuleList()   
        self.RBs.append(ResBlock(nf=n_feats))
    def forward(self, x):
        x = F.relu(self.conv_head(x))
        x1 = x     
        x = self.RBs[0](x)
        x = x + x1
        return x
        

 
class FE(nn.Module):
    def __init__(self, in_chl, nf, n_blks=[4, 4, 4], act='relu'):
        super(FE, self).__init__()
        block = functools.partial(ResBlock, nf=nf)
        self.conv_L1 = nn.Conv2d(in_chl, nf, 3, 1, 1, bias=True)
        self.blk_L1 = make_layer(block, n_layers=n_blks[0])
        self.conv_L2 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.blk_L2 = make_layer(block, n_layers=n_blks[1])

        self.conv_L3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.blk_L3 = make_layer(block, n_layers=n_blks[2])

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        fea_L1 = self.blk_L1(self.act(self.conv_L1(x)))
        fea_L2 = self.blk_L2(self.act(self.conv_L2(fea_L1)))
        fea_L3 = self.blk_L3(self.act(self.conv_L3(fea_L2)))

        return [fea_L1, fea_L2, fea_L3]


class Union(nn.Module):
    def __init__(self, n_feats):
        super(Union, self).__init__()
        self.conv13 = conv1x1(n_feats, n_feats)
        self.conv23 = conv1x1(n_feats, n_feats)
        self.conv_merge = conv3x3(n_feats * 4, n_feats)
        self.conv_tail1 = conv3x3(n_feats, n_feats)

    def forward(self, x, x1, x2, x3):


        x = F.interpolate(x, scale_factor=4, mode='bicubic')
        x13 = F.interpolate(x1, scale_factor=4, mode='bicubic')
        x13 = F.relu(self.conv13(x13))
        x23 = F.interpolate(x2, scale_factor=2, mode='bicubic')
        x23 = F.relu(self.conv23(x23))

        x = F.relu(self.conv_merge(torch.cat((x, x3, x13, x23), dim=1)))
        x = self.conv_tail1(x)

        return x
class gradient(nn.Module):
    def __init__(self):
        super(gradient, self).__init__()

   
    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        #self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
        #self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False) #cpu测试
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=2)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=2)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=2)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=2)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=2)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=2)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x

class gradient_nopadding(nn.Module):
    def __init__(self):
        super(gradient_nopadding, self).__init__()
        # kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        # kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        # self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).cuda()
        # self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False).cuda()


    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()
        #self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        #self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding = 1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding = 1)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding = 1)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding = 1)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding = 1)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding = 1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x
class APFM1(nn.Module):
    def __init__(self,n_feats):
        super(APFM1, self).__init__()
        self.n_feats=n_feats
        self.conv11_head = conv3x3(64 + n_feats, n_feats)
        self.F_f = nn.Sequential(
            nn.Linear(self.n_feats, 4 * self.n_feats),
            nn.ReLU(),
            nn.Linear(4 * self.n_feats, self.n_feats),
            nn.Sigmoid()
        )
        self.RB11 = nn.ModuleList()
        for i in range(12):
            self.RB11.append(ResBlock(nf=n_feats))
        # out channel: 160
        self.F_p = nn.Sequential(
            conv1x1(self.n_feats, 4 * self.n_feats),
            conv1x1(4 *  self.n_feats, self.n_feats)
        )

     # condense layer
        self.condense = conv3x3(self.n_feats, self.n_feats)

    def forward(self, x, x1,S):
        
        f_ref = x
        cor = torch.cat([f_ref, x1], dim=1)
        cor = self.conv11_head(cor)
        cor = cor * S
        cor = f_ref + cor

        w = F.adaptive_avg_pool2d(cor, (1, 1)).squeeze()  # (n,c) : (4, 160)
        if len(w.shape) == 1:
            w = w.unsqueeze(dim=0)
        w = self.F_f(w)
        w = w.reshape(*w.shape, 1, 1)

        for i in range(12):
            cor = self.RB11[i](cor)

        cor = self.F_p(cor)
        cor1 = self.condense(w * cor)

        return cor1
class APFM2(nn.Module):
    def __init__(self,n_feats):
        super(APFM2, self).__init__()
        self.n_feats=n_feats
        self.conv22_head = conv3x3(64 + n_feats, n_feats)
        self.F_f = nn.Sequential(
            nn.Linear(self.n_feats, 4 *  self.n_feats),
            nn.ReLU(),
            nn.Linear(4 * self.n_feats,  self.n_feats),
            nn.Sigmoid()
        )

        self.RB22 = nn.ModuleList()
        for i in range(12):
            self.RB22.append(ResBlock(nf=n_feats))

        # out channel: 160
        self.F_p = nn.Sequential(
            conv1x1(self.n_feats, 4 * self.n_feats),
            conv1x1(4 * self.n_feats, self.n_feats)
        )
        # condense layer
        self.condense = conv3x3(self.n_feats,  self.n_feats)

    def forward(self, x, x2,S):
        _,_,H,W = x2.size()
        x = F.interpolate(x, scale_factor=2, mode='bicubic')#
        S = F.interpolate(S, size=(H,W), mode='bicubic')

        f_ref = x
        cor = torch.cat([f_ref, x2], dim=1)
        cor = self.conv22_head(cor)

        cor = cor * S
        cor = f_ref + cor

        w = F.adaptive_avg_pool2d(cor, (1, 1)).squeeze()  # (n,c) : (4, 160)
        if len(w.shape) == 1:
            w = w.unsqueeze(dim=0)
        w = self.F_f(w)
        w = w.reshape(*w.shape, 1, 1)

        for i in range(12):
            cor = self.RB22[i](cor)

        cor = self.F_p(cor)
        cor2 = self.condense(w * cor)

        return cor2

class APFM3(nn.Module):
    def __init__(self,n_feats):
        super(APFM3, self).__init__()
        self.n_feats=n_feats
        self.conv33_head = conv3x3(64 + n_feats, n_feats)
        self.F_f = nn.Sequential(
            nn.Linear(self.n_feats, 4 * self.n_feats),
            nn.ReLU(),
            nn.Linear(4 * self.n_feats, self.n_feats),
            nn.Sigmoid()
        )

        self.RB33 = nn.ModuleList()
        for i in range(12):
            self.RB33.append(ResBlock(nf=n_feats))

        # out channel: 160
        self.F_p = nn.Sequential(
            conv1x1(self.n_feats, 4 * self.n_feats),
            conv1x1(4 *  self.n_feats, self.n_feats)
        )
        # condense layer
        self.condense = conv3x3(self.n_feats, self.n_feats)

    def forward(self, x, x3,S):
        #
        _,_,H,W = x3.size()
        x = F.interpolate(x, scale_factor=2, mode='bicubic')#
        S = F.interpolate(S, size=(H,W), mode='bicubic')

        f_ref = x

        cor = torch.cat([f_ref, x3], dim=1)
        cor = self.conv33_head(cor)

        cor = cor * S
        cor = f_ref + cor

        w = F.adaptive_avg_pool2d(cor, (1, 1)).squeeze()  # (n,c) : (4, 160)
        if len(w.shape) == 1:
            w = w.unsqueeze(dim=0)
        w = self.F_f(w)
        w = w.reshape(*w.shape, 1, 1)

        for i in range(12):
            cor = self.RB33[i](cor)

        cor = self.F_p(cor)
        cor3 = self.condense(w * cor)

        return cor3

class MFFM(nn.Module):
    def __init__(self, num_res_blocks=[12,16,8,4], n_feats=64, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, norm_type=None, \
                 act_type='leakyrelu', mode='CNA', upsample_mode='upconv', upscale=4):
        super(MFFM, self).__init__()
        self.n_feats = n_feats
        self.IFE = IFE(n_feats)
        self.union = Union(n_feats)
        self.g_nopadding = gradient_nopadding()
        self.apfm1 = APFM1(n_feats)
        self.apfm2 = APFM2(n_feats)
        self.apfm3 = APFM3(n_feats)
        n_upscale = int(math.log(upscale, 2))
        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        if upscale == 3:
            b_upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            b_upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
        self.b_fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)

        self.b_concat_1 = B.conv_block(2 * nf, nf, kernel_size=3, norm_type=None, act_type=None)
        self.b_block_1 = B.RRDB(nf * 2, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
                                norm_type=norm_type, act_type=act_type, mode='CNA')

        self.b_concat_2 = B.conv_block(2 * nf, nf, kernel_size=3, norm_type=None, act_type=None)
        self.b_block_2 = B.RRDB(nf * 2, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
                                norm_type=norm_type, act_type=act_type, mode='CNA')

        self.b_concat_3 = B.conv_block(2 * nf, nf, kernel_size=3, norm_type=None, act_type=None)
        self.b_block_3 = B.RRDB(nf * 2, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
                                norm_type=norm_type, act_type=act_type, mode='CNA')
    
        self.conv_w = B.conv_block(nf, out_nc, kernel_size=1, norm_type=None, act_type=None)
        self.f_HR_conv1 = B.conv_block(nf*2, out_nc, kernel_size=3, norm_type=None, act_type=None)

    def forward(self, x, S=None, T_lv3=None, T_lv2=None, T_lv1=None):
        ### shallow feature extraction
        x_grad = self.g_nopadding(x)

        x = self.IFE(x)

        T_lv1 = self.apfm1(x, T_lv1, S)
        x_fea1 = T_lv1
       

        T_lv2 = self.apfm2(T_lv1, T_lv2, S) #上个T_lv1作为下个的输入
        x_fea2 = T_lv2

        T_lv3 = self.apfm3(T_lv2, T_lv3, S)
        x_fea3 = T_lv3
        x = self.union(x, T_lv1, T_lv2, T_lv3)

        ##grad
        x_b_fea = self.b_fea_conv(x_grad)
        
        x_cat_1 = torch.cat([x_b_fea, x_fea1], dim=1)  #低纹理cat低梯度

        x_cat_1 = self.b_block_1(x_cat_1)
        x_cat_1 = self.b_concat_1(x_cat_1)
        x_cat_1 = F.interpolate(x_cat_1, scale_factor=2, mode='bicubic')  # 放大特征，而不是缩小特征
        x_cat_2 = torch.cat([x_cat_1, x_fea2], dim=1) # 

        x_cat_2 = self.b_block_2(x_cat_2)
        x_cat_2 = self.b_concat_2(x_cat_2)
        x_cat_2 = F.interpolate(x_cat_2, scale_factor=2, mode='bicubic')  #
        x_cat_3 = torch.cat([x_cat_2, x_fea3], dim=1)#

        x_cat_3 = self.b_block_3(x_cat_3)
        x_cat_3 = self.b_concat_3(x_cat_3)
   
        x_out_branch = self.conv_w(x_cat_3)

        ########
        x_branch_d = x_cat_3

        x_f_cat = torch.cat([x_branch_d, x], dim=1)
        x_out = self.f_HR_conv1(x_f_cat)

        ########
        return x_out_branch, x_out



class DLASR(nn.Module):
    def __init__(self, args):
        super(DLASR, self).__init__()
        in_chl = args.input_nc
        nf = args.nf
        n_blks = [4, 4, 4]
  

        self.scale = args.sr_scale
        self.num_nbr = args.num_nbr
        self.psize = 3
        self.lr_block_size_ = 9
        self.ref_down_block_size = 1.5
        self.dilations =  [2]

        self.fe = FE(in_chl=in_chl, nf=nf, n_blks=n_blks)

        self.mffm = MFFM(num_res_blocks=[12,16,8,4], n_feats=64, in_nc=3,\
         out_nc=3, nf=64, nb=23, gc=32)#nf值得就是通道数目


        self.criterion = nn.L1Loss(reduction='mean')

        self.weight_init(scale=0.1)

    def weight_init(self, scale=0.1):
        for name, m in self.named_modules():
            classname = m.__class__.__name__
            if classname == 'DCN':
                continue
            elif classname == 'Conv2d' or classname == 'ConvTranspose2d':
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.5 * math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('BatchNorm') != -1:
                if m.weight is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            elif classname.find('Linear') != -1:
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data = torch.ones(m.bias.data.size())

        for name, m in self.named_modules():
            classname = m.__class__.__name__
            if classname == 'ResBlock':
                m.conv1.weight.data *= scale
                m.conv2.weight.data *= scale

    def bis(self, input, dim, index):
        # batch index select
        # input: [N, C*k*k, H*W]
        # dim: scalar > 0
        # index: [N, Hi, Wi]
        views = [input.size(0)] + [1 if i != dim else -1 for i in range(1, len(input.size()))]  # views = [N, 1, -1]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1  # expanse = [-1, C*k*k, -1]
        index = index.clone().view(views).expand(expanse)  # [N, Hi, Wi] -> [N, 1, Hi*Wi] - > [N, C*k*k, Hi*Wi]
        return torch.gather(input, dim, index)  # [N, C*k*k, Hi*Wi]

    def search_org(self, lr, reflr, ks=3, pd=1, stride=1):
        # lr: [N, C, H, W]
        # reflr: [N, C, Hr, Wr]

        batch, c, H, W = lr.size()
        _, _, Hr, Wr = reflr.size()

        reflr_unfold = F.unfold(reflr, kernel_size=(ks, ks), padding=0, stride=stride)  # [N, C*k*k, Hr*Wr]
        lr_unfold = F.unfold(lr, kernel_size=(ks, ks), padding=0, stride=stride)
        lr_unfold = lr_unfold.permute(0, 2, 1)  # [N, H*W, C*k*k]

        lr_unfold = F.normalize(lr_unfold, dim=2)
        reflr_unfold = F.normalize(reflr_unfold, dim=1)

        corr = torch.bmm(lr_unfold, reflr_unfold)  # [N, H*W, Hr*Wr]
        corr = corr.view(batch, H-2, W-2, (Hr-2)*(Wr-2))
        sorted_corr, ind_l = torch.topk(corr, self.num_nbr, dim=-1, largest=True, sorted=True)  # [N, H, W, num_nbr]
        return sorted_corr, ind_l

    def search(self, lr, reflr, ks=3, pd=1, stride=1, dilations=[2]):
        # lr: [N, p*p, C, k_y, k_x]
        # reflr: [N, C, Hr, Wr]

        N, C, Hr, Wr = reflr.size()
        _, _, _, k_y, k_x = lr.size()
        x, y = k_x // 2, k_y // 2
        corr_sum = 0
        for i, dilation in enumerate(dilations):
            reflr_patches = F.unfold(reflr, kernel_size=(ks, ks), padding=dilation, stride=stride, dilation=dilation)  # [N, C*ks*ks, Hr*Wr]
            lr_patches = lr[:, :, :, y - dilation: y + dilation + 1: dilation,
                                     x - dilation: x + dilation + 1: dilation]  # [N, p*p, C, ks, ks]
            lr_patches = lr_patches.contiguous().view(N, -1, C * ks * ks)  # [N, p*p, C*ks*ks]

            lr_patches = F.normalize(lr_patches, dim=2)
            reflr_patches = F.normalize(reflr_patches, dim=1)
            corr = torch.bmm(lr_patches, reflr_patches)  # [N, p*p, Hr*Wr]
            corr_sum = corr_sum + corr

        sorted_corr, ind_l = torch.topk(corr_sum, self.num_nbr, dim=-1, largest=True, sorted=True)  # [N, p*p, num_nbr]

        return sorted_corr, ind_l

    def transfer(self, fea, index, soft_att, ks=3, pd=1, stride=1):
        # fea: [N, C, H, W]
        # index: [N, Hi, Wi]
        # soft_att: [N, 1, Hi, Wi]
        scale = stride

        fea_unfold = F.unfold(fea, kernel_size=(ks, ks), padding=0, stride=stride)  # [N, C*k*k, H*W]
        out_unfold = self.bis(fea_unfold, 2, index)  # [N, C*k*k, Hi*Wi]
        divisor = torch.ones_like(out_unfold)

        _, Hi, Wi = index.size()
        out_fold = F.fold(out_unfold, output_size=(Hi*scale, Wi*scale), kernel_size=(ks, ks), padding=pd, stride=stride)
        divisor = F.fold(divisor, output_size=(Hi*scale, Wi*scale), kernel_size=(ks, ks), padding=pd, stride=stride)
        soft_att_resize = F.interpolate(soft_att, size=(Hi*scale, Wi*scale), mode='bilinear')
        out_fold = out_fold / divisor * soft_att_resize
        return out_fold

    def make_grid(self, idx_x1, idx_y1, diameter_x, diameter_y, s):
        idx_x1 = idx_x1 * s
        idx_y1 = idx_y1 * s
        idx_x1 = idx_x1.view(-1, 1).repeat(1, diameter_x * s)
        idx_y1 = idx_y1.view(-1, 1).repeat(1, diameter_y * s)
        idx_x1 = idx_x1 + torch.arange(0, diameter_x * s, dtype=torch.long, device=idx_x1.device).view(1, -1)
        idx_y1 = idx_y1 + torch.arange(0, diameter_y * s, dtype=torch.long, device=idx_y1.device).view(1, -1)

        ind_y_l = []
        ind_x_l = []
        for i in range(idx_x1.size(0)):
            grid_y, grid_x = torch.meshgrid(idx_y1[i], idx_x1[i])
            ind_y_l.append(grid_y.contiguous().view(-1))
            ind_x_l.append(grid_x.contiguous().view(-1))
        ind_y = torch.cat(ind_y_l)
        ind_x = torch.cat(ind_x_l)

        return ind_y, ind_x

    def forward(self, lr, ref, ref_down, gt=None):
        size = 8
        _, _, h, w = lr.size()
        
        px = w // size#宽度方向的块数
        py = h // size
        k_x = w // px #每块宽度
        k_y = h // py
        _, _, h, w = ref_down.size()
        diameter_x = 2 * int(w // (2 * px) * self.ref_down_block_size) + 1   
        diameter_y = 2 * int(h // (2 * py) * self.ref_down_block_size) + 1
        lrsr = F.interpolate(lr, scale_factor=self.scale, mode='bicubic')
        fea_lr_l = self.fe(lr)
        fea_reflr_l = self.fe(ref_down)
        fea_ref_l = self.fe(ref)

        N, C, H, W = fea_lr_l[0].size()#[fea_L1, fea_L2, fea_L3]
        _, _, Hr, Wr = fea_reflr_l[0].size()

        lr_patches = F.pad(fea_lr_l[0], pad=(1, 1, 1, 1), mode='replicate')
        lr_patches = F.unfold(lr_patches, kernel_size=(k_y + 2, k_x + 2), padding=(0, 0),
                              stride=(k_y, k_x))  # [N, C*(k_y+2)*(k_x+2), py*px],py*px:LR 块数
        lr_patches = lr_patches.view(N, C, k_y + 2, k_x + 2, py * px).permute(0, 4, 1, 2, 3)  # [N, py*px, C, k_y+2, k_x+2]
 
        ## find the corresponding ref patch for each lr patch
        sorted_corr, ind_l = self.search(lr_patches, fea_reflr_l[0],
                                         ks=3, pd=1, stride=1, dilations=self.dilations)

        ## crop corresponding ref patches
        index = ind_l[:, :, 0]  # [N, py*px]
        idx_x = index % Wr
        idx_y = index // Wr
        idx_x1 = idx_x - diameter_x//2 - 1
        idx_x2 = idx_x + diameter_x//2 + 1
        idx_y1 = idx_y - diameter_y//2 - 1
        idx_y2 = idx_y + diameter_y//2 + 1

        mask = (idx_x1 < 0).long()
        idx_x1 = idx_x1 * (1 - mask)
        idx_x2 = idx_x2 * (1 - mask) + (diameter_x + 1) * mask

        mask = (idx_x2 > Wr - 1).long()
        idx_x2 = idx_x2 * (1 - mask) + (Wr - 1) * mask
        idx_x1 = idx_x1 * (1 - mask) + (idx_x2 - (diameter_x + 1)) * mask

        mask = (idx_y1 < 0).long()
        idx_y1 = idx_y1 * (1 - mask)
        idx_y2 = idx_y2 * (1 - mask) + (diameter_y + 1) * mask

        mask = (idx_y2 > Hr - 1).long()
        idx_y2 = idx_y2 * (1 - mask) + (Hr - 1) * mask
        idx_y1 = idx_y1 * (1 - mask) + (idx_y2 - (diameter_y + 1)) * mask

        ind_y_x1, ind_x_x1 = self.make_grid(idx_x1, idx_y1, diameter_x+2, diameter_y+2, 1)
        ind_y_x2, ind_x_x2 = self.make_grid(idx_x1, idx_y1, diameter_x+2, diameter_y+2, 2)
        ind_y_x4, ind_x_x4 = self.make_grid(idx_x1, idx_y1, diameter_x+2, diameter_y+2, 4)

        ind_b = torch.repeat_interleave(torch.arange(0, N, dtype=torch.long, device=idx_x1.device), py * px * (diameter_y+2) * (diameter_x+2))
        ind_b_x2 = torch.repeat_interleave(torch.arange(0, N, dtype=torch.long, device=idx_x1.device), py * px * ((diameter_y+2)*2) * ((diameter_x+2)*2))
        ind_b_x4 = torch.repeat_interleave(torch.arange(0, N, dtype=torch.long, device=idx_x1.device), py * px * ((diameter_y+2)*4) * ((diameter_x+2)*4))

        reflr_patches = fea_reflr_l[0][ind_b, :, ind_y_x1, ind_x_x1].view(N*py*px, diameter_y+2, diameter_x+2, C).permute(0, 3, 1, 2).contiguous()  # [N*py*px, C, (radius_y+1)*2, (radius_x+1)*2]
        ref_patches_x1 = fea_ref_l[2][ind_b, :, ind_y_x1, ind_x_x1].view(N*py*px, diameter_y+2, diameter_x+2, C).permute(0, 3, 1, 2).contiguous()
        ref_patches_x2 = fea_ref_l[1][ind_b_x2, :, ind_y_x2, ind_x_x2].view(N*py*px, (diameter_y+2)*2, (diameter_x+2)*2, C).permute(0, 3, 1, 2).contiguous()
        ref_patches_x4 = fea_ref_l[0][ind_b_x4, :, ind_y_x4, ind_x_x4].view(N*py*px, (diameter_y+2)*4, (diameter_x+2)*4, C).permute(0, 3, 1, 2).contiguous()

        ## calculate correlation between lr patches and their corresponding ref patches
        lr_patches = lr_patches.contiguous().view(N*py*px, C, k_y+2, k_x+2)
        corr_all_l, index_all_l = self.search_org(lr_patches, reflr_patches,
                                              ks=self.psize, pd=self.psize // 2, stride=1)
        index_all = index_all_l[:, :, :, 0]  # ID
        soft_att_all = corr_all_l[:, :, :, 0:1].permute(0, 3, 1, 2)  # weight

        warp_ref_patches_x1 = self.transfer(ref_patches_x1, index_all, soft_att_all,
                                            ks=self.psize, pd=self.psize // 2, stride=1)  # [N*py*px, C, k_y, k_x]
        warp_ref_patches_x2 = self.transfer(ref_patches_x2, index_all, soft_att_all,
                                            ks=self.psize * 2, pd=self.psize // 2 * 2, stride=2)  # [N*py*px, C, k_y*2, k_x*2]
        warp_ref_patches_x4 = self.transfer(ref_patches_x4, index_all, soft_att_all,
                                            ks=self.psize * 4, pd=self.psize // 2 * 4, stride=4)  # [N*py*px, C, k_y*4, k_x*4]

        warp_ref_patches_x1 = warp_ref_patches_x1.view(N, py, px, C, H//py, W//px).permute(0, 3, 1, 4, 2, 5).contiguous()  # [N, C, py, H//py, px, W//px]
        warp_ref_patches_x1 = warp_ref_patches_x1.view(N, C, H, W)
        warp_ref_patches_x2 = warp_ref_patches_x2.view(N, py, px, C, H//py*2, W//px*2).permute(0, 3, 1, 4, 2, 5).contiguous()  # [N, C, py, H//py*2, px, W//px*2]
        warp_ref_patches_x2 = warp_ref_patches_x2.view(N, C, H*2, W*2)
        warp_ref_patches_x4 = warp_ref_patches_x4.view(N, py, px, C, H//py*4, W//px*4).permute(0, 3, 1, 4, 2, 5).contiguous()  # [N, C, py, H//py*4, px, W//px*4]
        warp_ref_patches_x4 = warp_ref_patches_x4.view(N, C, H*4, W*4)

        warp_ref_l = [warp_ref_patches_x4, warp_ref_patches_x2, warp_ref_patches_x1]
        soft_att_all = soft_att_all.contiguous().view(N,1,k_y*py,k_x*px)
        _,out = self.mffm(lr,soft_att_all,warp_ref_patches_x4, warp_ref_patches_x2,warp_ref_patches_x1)
        out = out + lrsr

        if gt is not None:
            L1_loss = self.criterion(out, gt)
            loss_dict = OrderedDict(L1=L1_loss)
            return loss_dict
        else:
            return out




if __name__ == "__main__":
    pass