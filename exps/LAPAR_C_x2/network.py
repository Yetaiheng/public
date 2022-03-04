import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from utils.modules.lightWeightNet import WeightNet
from ecb import ECB,SeqConv3x3


class ComponentDecConv(nn.Module):
    def __init__(self, k_path, k_size):
        super(ComponentDecConv, self).__init__()

        kernel = pickle.load(open(k_path, 'rb'))
        kernel = torch.from_numpy(kernel).float().view(-1, 1, k_size, k_size)
        self.register_buffer('weight', kernel)

    def forward(self, x):
        out = F.conv2d(x, weight=self.weight, bias=None, stride=1, padding=0, groups=1)

        return out

class eSR_TM(nn.Module):
    def __init__(self, config):
        super(eSR_TM,self).__init__()
        
        self.channels  = config.ESRLAPAR.CC
        self.kernel_size = config.ESRLAPAR.KK
        self.stride = config.ESRLAPAR.SS
        self.k_size = config.MODEL.KERNEL_SIZE
        self.decom_conv = ComponentDecConv(config.MODEL.KERNEL_PATH, self.k_size)
        self.criterion = nn.L1Loss(reduction='mean')
        
        self.pixel_shuffle = nn.PixelShuffle(self.stride)
        self.softmax = nn.Softmax(dim=1)
        self.filter = nn.Conv2d(
            in_channels=3,
            out_channels=2*3*self.stride*self.stride*self.channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=(
                (self.kernel_size-1)//2,
                (self.kernel_size-1)//2
            ),
            groups=1,
            bias=False,
            dilation=1
        )
        nn.init.xavier_normal_(self.filter.weight, gain=1.)
        self.filter.weight.data[:, 0, self.kernel_size//2, self.kernel_size//2] = 1.
        
    def forward(self, x, gt=None):
        filtered = self.pixel_shuffle(self.filter(x))

        value_r, key_r, value_g, key_g, value_b, key_b = torch.split(filtered, [self.channels, self.channels, self.channels, self.channels, self.channels, self.channels], dim=1)
        out_r  = torch.sum( value_r *  self.softmax(key_r ),dim=1, keepdim=True )
        out_g = torch.sum( value_g * self.softmax(key_g ),dim=1, keepdim=True )
        out_b = torch.sum( value_b * self.softmax(key_b),dim=1, keepdim=True )
        out = torch.cat((out_r, out_g, out_b), 1)
        if gt is not None:
            loss_dict = dict(L1=self.criterion(out, gt))
            return loss_dict
        else:
            return out
        
class eSR_ECB(nn.Module):
    def __init__(self, config):
        super(eSR_ECB,self).__init__()
        
        self.channels  = config.ESRLAPAR.CC
        self.kernel_size = config.ESRLAPAR.KK
        self.stride = config.ESRLAPAR.SS
        self.k_size = config.MODEL.KERNEL_SIZE
        self.decom_conv = ComponentDecConv(config.MODEL.KERNEL_PATH, self.k_size)
        self.criterion = nn.L1Loss(reduction='mean')
        
        self.pixel_shuffle = nn.PixelShuffle(self.stride)
        self.softmax = nn.Softmax(dim=1)
        self.filter = ECB(3, 2*3*self.stride*self.stride*self.channels,depth_multiplier=2.0 ,act_type='prelu', with_idt=0)
        
    def forward(self, x, gt=None):
        filtered = self.pixel_shuffle(self.filter(x))

        value_r, key_r, value_g, key_g, value_b, key_b = torch.split(filtered, [self.channels, self.channels, self.channels, self.channels, self.channels, self.channels], dim=1)
        out_r  = torch.sum( value_r *  self.softmax(key_r ),dim=1, keepdim=True )
        out_g = torch.sum( value_g * self.softmax(key_g ),dim=1, keepdim=True )
        out_b = torch.sum( value_b * self.softmax(key_b),dim=1, keepdim=True )
        out = torch.cat((out_r, out_g, out_b), 1)
        if gt is not None:
            loss_dict = dict(L1=self.criterion(out, gt))
            return loss_dict
        else:
            return out
        
class eSR_LAPAR(nn.Module):
    def __init__(self, config):
        super(eSR_LAPAR,self).__init__()
        
        self.channels  = config.ESRLAPAR.CC
        self.kernel_size = config.ESRLAPAR.KK
        self.stride = config.ESRLAPAR.SS
        self.k_size = config.MODEL.KERNEL_SIZE
        self.decom_conv = ComponentDecConv(config.MODEL.KERNEL_PATH, self.k_size)
        self.criterion = nn.L1Loss(reduction='mean')
        bias = torch.randn(1) * 1e-3+1.05
        self.bias = nn.Parameter(torch.FloatTensor(bias))
        
        self.pixel_shuffle = nn.PixelShuffle(self.stride)
        self.softmax = nn.Softmax(dim=4)
        self.filter = nn.Conv2d(
            in_channels=3,
            out_channels=self.stride*self.stride*self.channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=(
                (self.kernel_size-1)//2,
                (self.kernel_size-1)//2
            ),
            groups=1,
            bias=False,
            dilation=1
        )
        nn.init.xavier_normal_(self.filter.weight, gain=1.)
        #self.filter.weight.data[:, 0, self.kernel_size//2, self.kernel_size//2] = 1.
        
    def forward(self, x, gt=None):
        B, C, H, W = x.size()
        bic = F.interpolate(x, scale_factor=self.stride, mode='bicubic', align_corners=False)##双三次差值放大两倍
        pad = self.k_size // 2
        x_pad = F.pad(bic, pad=(pad, pad, pad, pad), mode='reflect')
        pad_H, pad_W = x_pad.size()[2:]
        x_pad = x_pad.view(B * 3, 1, pad_H, pad_W)
        x_com = self.decom_conv(x_pad).view(B, 3, -1, self.stride* H, self.stride * W)  # B, 3, N_K, Hs, Ws
        
        weight  = self.pixel_shuffle(self.filter(x))
        weight = weight.view(B, 1, -1, self.stride * H, self.stride * W)  # B, 1, N_K, Hs, Ws
        weight = weight.permute( (0, 1, 3,4,2))
        weight = self.softmax(weight)
        weight = weight.permute((0, 1, 4, 2, 3))*self.bias
        #print(self.bias)
        out = torch.sum(weight * x_com, dim=2)
        
        if gt is not None:
            loss_dict = dict(L1=self.criterion(out, gt))
            return loss_dict
        else:
            return out

class eSR_LAPAR_TR(nn.Module):
    def __init__(self, config):
        super(eSR_LAPAR_TR,self).__init__()
        
        self.channels  = config.ESRLAPAR.CC
        self.kernel_size = config.ESRLAPAR.KK
        self.stride = config.ESRLAPAR.SS
        self.k_size = config.MODEL.KERNEL_SIZE
        self.decom_conv = ComponentDecConv(config.MODEL.KERNEL_PATH, self.k_size)
        self.criterion = nn.L1Loss(reduction='mean')
        
        self.pixel_shuffle = nn.PixelShuffle(self.stride)
        self.softmax = nn.Softmax(dim=4)
        self.filter = nn.Conv2d(
            in_channels=3,
            out_channels=2*self.stride*self.stride*self.channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=(
                (self.kernel_size-1)//2,
                (self.kernel_size-1)//2
            ),
            groups=1,
            bias=False,
            dilation=1
        )
        nn.init.xavier_normal_(self.filter.weight, gain=1.)
        #self.filter.weight.data[:, 0, self.kernel_size//2, self.kernel_size//2] = 1.
        
    def forward(self, x, gt=None):
        B, C, H, W = x.size()
        bic = F.interpolate(x, scale_factor=self.stride, mode='bicubic', align_corners=False)##双三次差值放大两倍
        pad = self.k_size // 2
        x_pad = F.pad(bic, pad=(pad, pad, pad, pad), mode='reflect')
        pad_H, pad_W = x_pad.size()[2:]
        x_pad = x_pad.view(B * 3, 1, pad_H, pad_W)
        x_com = self.decom_conv(x_pad).view(B, 3, -1, self.stride* H, self.stride * W)  # B, 3, N_K, Hs, Ws
        
        
        weighted  = self.pixel_shuffle(self.filter(x))# B, 2*N_K, Hs, Ws
        value, key = torch.split(weighted, [self.channels, self.channels], dim=1)# B, N_K, Hs, Ws
        weight = value*key
        weight = weight.view(B, 1, -1, self.stride * H, self.stride * W)  # B, 1, N_K, Hs, Ws
        weight = weight.permute( (0, 1, 3,4,2))
        weight = self.softmax(weight)
        weight = weight.permute((0, 1, 4, 2, 3))
        out = torch.sum(weight * x_com, dim=2)
        
        if gt is not None:
            loss_dict = dict(L1=self.criterion(out, gt))
            return loss_dict
        else:
            return out
        
class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()

        self.k_size = config.MODEL.KERNEL_SIZE
        self.s = config.MODEL.SCALE

        self.w_conv = WeightNet(config.MODEL)
        self.decom_conv = ComponentDecConv(config.MODEL.KERNEL_PATH, self.k_size)

        self.criterion = nn.L1Loss(reduction='mean')


    def forward(self, x, gt=None):
        B, C, H, W = x.size()

        bic = F.interpolate(x, scale_factor=self.s, mode='bicubic', align_corners=False)##双三次差值放大两倍
        pad = self.k_size // 2
        x_pad = F.pad(bic, pad=(pad, pad, pad, pad), mode='reflect')
        pad_H, pad_W = x_pad.size()[2:]
        x_pad = x_pad.view(B * 3, 1, pad_H, pad_W)
        x_com = self.decom_conv(x_pad).view(B, 3, -1, self.s * H, self.s * W)  # B, 3, N_K, Hs, Ws

        weight = self.w_conv(x)
        weight = weight.view(B, 1, -1, self.s * H, self.s * W)  # B, 1, N_K, Hs, Ws

        out = torch.sum(weight * x_com, dim=2)

        if gt is not None:
            loss_dict = dict(L1=self.criterion(out, gt))
            return loss_dict
        else:
            return out

