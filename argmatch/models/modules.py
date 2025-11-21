from re import T
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import kornia.utils as KU
import math
from argmatch.utils import *

default_length = (608*800)/16/16
def kde(x, std = 0.1):
    # use a gaussian kernel to estimate density
    x = x.half() # Do it in half precision TODO: remove hardcoding
    scores = (-torch.cdist(x,x)**2/(2*std**2)).exp()
    density = scores.sum(dim=-1)
    return density

class SoftplusParameterization(nn.Module):
    def forward(self, x):
        return F.softplus(x)
    
class BasicLayer(nn.Module):
    """
        Basic Convolutional Layer: Conv2d -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, relu = True, relu6=False, obn=False):
        super().__init__()
        act = nn.ReLU6 if relu6 else nn.ReLU
        self.layer = nn.Sequential(
                                    nn.Conv2d( in_channels, out_channels, kernel_size, padding = padding, stride=stride, dilation=dilation, bias = not (relu|obn)),
                                    nn.BatchNorm2d(out_channels, affine=True, momentum=0.01) if relu|obn else nn.Identity(),
                                    act() if relu else nn.Identity()
                                    )
    def forward(self, x):
        return self.layer(x)

class  ResBasicLayer(nn.Module):
    """
        Basic Convolutional Layer: Conv2d -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, obn=False, relu6=False,drop=False):
        super().__init__()
        self.layer = nn.Sequential(
                                    nn.Conv2d( in_channels, out_channels, kernel_size, padding = padding, stride=stride, dilation=dilation, bias = not (obn|relu6)),
                                    nn.BatchNorm2d(out_channels, affine=True, momentum=0.01) if obn|relu6 else nn.Identity(),
                                    nn.LeakyReLU(negative_slope=0.05) if not (relu6|obn) else nn.ReLU6(),
                                    nn.Conv2d(out_channels,out_channels,1),
                                    )
        if drop:
            self.dropout = nn.Dropout(0.1)
        else:
            self.dropout = None
        self.skip = nn.Conv2d(in_channels,out_channels,1,bias=False)
    def forward(self, x):
        if self.dropout is not None:
            t = self.skip(x)
            for i in range(len(self.layer)):
                x = self.layer[i](x)
                if i == len(self.layer)-2:
                    x = self.dropout(x)
            return x + t
        return self.layer(x)+self.skip(x)
    
class BasicMulLayer(nn.Module):
    """
    Basic Convolutional Layer: Conv2d -> BatchNorm -> ReLU
    """

    def __init__(
        self,
        in_channels,
        out_channels=None,
        relu=True,
        wo_norm = False,
        relu_1=False
    ):
        super().__init__()
        self.layer1 = nn.Conv2d(in_channels,out_channels,1)
        self.act1 = nn.Tanh() if not relu_1 else nn.ReLU()
        if not wo_norm:
            self.layer2 = nn.Sequential(nn.Conv2d(in_channels,out_channels,1,bias=False),
                                        nn.BatchNorm2d(out_channels,momentum=0.01),
                                        nn.Sigmoid()
                                        )
        else:
            self.layer2 = nn.Sequential(nn.Conv2d(in_channels,out_channels,1,bias=True),
                                        nn.Sigmoid()
                                        )
        self.merge = nn.Conv2d(in_channels+out_channels,out_channels,1,bias=not relu)
        if relu:
            self.act2 = nn.Sequential(nn.BatchNorm2d(out_channels,momentum=0.01),nn.ReLU())
        else:
            self.act2 = nn.Identity()
        # with torch.no_grad():
        #     self.layer1.weight.mul_(0.1)
        #     if self.layer1.bias is not None:
        #         self.layer1.bias.mul_(0)
    def forward(self, x):
        x1 = self.act1(self.layer1(x))*self.layer2(x)
        return self.act2(self.merge(torch.cat([x1,x],dim=1)))
    

class BasicDWLayer(nn.Module):
    """
        Basic Convolutional Layer: Conv2d -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels=None, kernel_size=5, padding=2, dilation=1, relu=True, skip=True, wo_norm=False, relu6=False):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        if wo_norm:
            self.layer = nn.Sequential(
                                    nn.Conv2d( in_channels, out_channels, 1, bias=False),
                                    nn.Conv2d( out_channels, out_channels, kernel_size, padding = padding, bias = True, groups=out_channels, dilation=dilation),
                                    nn.LeakyReLU(negative_slope=0.05),
                                    nn.Conv2d( out_channels, out_channels, 1))
        else:            
            self.layer = nn.Sequential(
                                    nn.Conv2d( in_channels, out_channels, 1, bias=False),
                                    nn.Conv2d( out_channels, out_channels, kernel_size, padding = padding, bias = False, groups=out_channels, dilation=dilation),
                                    nn.BatchNorm2d(out_channels, affine=True, momentum=0.01),
                                    nn.ReLU(inplace = True),
                                    nn.Conv2d( out_channels, out_channels, 1))
        self.skip = skip
        
        if self.skip:
            self.skip_layer = nn.Conv2d(in_channels,out_channels,1,bias=False)

            
        if relu:
            self.activation = nn.Sequential(nn.BatchNorm2d(out_channels,momentum=0.01),nn.ReLU() if not relu6 else nn.ReLU6())
        else:
            self.activation = nn.Identity()
            
    def forward(self, x):
        if self.skip:
            x = self.layer(x)+self.skip_layer(x)
        else:
            x = self.layer(x)
            
        return self.activation(x)


class compressor(nn.Module):
    def __init__(self,ic,oc,hidden=None,dropout=False,obn=True):
        super().__init__()
        if hidden is None:
            hidden = int(1.5*oc)
            
        if hidden>ic:
            hidden = ic

        if dropout:
            self.do = nn.Dropout2d(0.1)
        else:
            self.do = None
        if obn:
            self.layers = nn.Sequential(
            BasicLayer(ic,hidden),
            BasicDWLayer(hidden,hidden),
            BasicLayer(hidden,oc,relu=False,obn=True))
            self.skip = nn.Conv2d(ic,oc,1,bias=False)
        else:
            self.layers = nn.Sequential(BasicLayer(ic,hidden),
            BasicDWLayer(hidden,hidden),
            BasicLayer(hidden,oc,relu=False,obn=False))
            self.skip = nn.Conv2d(ic,oc,1,bias=False)
        self.obn = obn
        # if obn:
        #     self.bn = nn.BatchNorm2d(oc,momentum=0.01)
    def forward(self,x):
        if self.do is not None:
            t = x
            for i in range(len(self.layers)):
                t = self.layers[i](t)
                if i==0:
                    t = self.do(t)
        else:
            t = self.layers(x)
        x = t+self.skip(x)
        # if self.obn:
        #     x = self.bn(x)
        return x

class compressor_light(nn.Module):
    def __init__(self,ic,oc,hidden=None):
        super().__init__()
        if hidden is None:
            hidden = int(1.5*oc)
            
        if hidden>ic:
            hidden = ic

        self.layers = nn.Sequential(
        BasicDWLayer(ic,hidden),
        BasicLayer(hidden,oc,relu=False,obn=False))
            
        self.skip = nn.Conv2d(ic,oc,1,bias=False)
    def forward(self,x):
        return self.layers(x)+self.skip(x)
        
import torch.nn.utils.parametrize as parametrize

class fuser(nn.Module):
    def __init__(self, ic1, ic2, oc, obn=True):
        super().__init__()
        ic = ic1 + ic2
        self.score_estimator = nn.Sequential(
            BasicLayer(ic, ic // 16),
            BasicLayer(ic//16, ic,relu=False, obn=True),
            nn.Sigmoid()
            
        )
        if obn:
            self.fus = nn.Sequential(BasicDWLayer(ic, ic), BasicLayer(ic,oc,relu=False, obn=True))
        else:
            self.fus = nn.Sequential(BasicDWLayer(ic, ic, relu=False), nn.LeakyReLU(negative_slope=0.05), BasicLayer(ic, oc, relu=False, obn=False))
        self.skip = nn.Conv2d(ic,oc,1,bias=False)
    

    def forward(self, x, x_new, type="221", outnorm=False):
        b,c,h1,w1 = x.shape
        b,c,h2,w2 = x_new.shape
        if h1!=h2:
            if type=="221":
                x_new = F.interpolate(x_new,[h1,w1],mode='bilinear',align_corners=False)
            else:
                x = F.interpolate(x,[h2,w2],mode='bilinear',align_corners=False)
        x_new = torch.cat([x, x_new], dim=1)
        score = self.score_estimator(x_new)
        return self.fus(x_new * score) + self.skip(x_new)

class fuser_1D(nn.Module):
    def __init__(self, ic1, ic2, oc, new_skip = True, obn=False):
        super().__init__()
        ic = ic1 + ic2
        self.score_estimator = nn.Sequential(
            BasicLayer(ic, ic // 16, kernel_size=1,padding=0),
            BasicLayer(ic//16, ic, kernel_size=1,padding=0,relu=False,obn=True),
            #nn.BatchNorm2d(ic, momentum=0.01),
            nn.Sigmoid(),
        )

        self.fus = nn.Sequential(BasicLayer(ic, oc, kernel_size=1, padding=0), BasicLayer(oc, oc, kernel_size=1, padding=0, relu=False, obn=True))
        self.skip = nn.Conv2d(ic,oc,1,bias=False)
        #self.conv = nn.Conv2d(ic,oc,1,bias=True)

    def forward(self, x, x_new, type="221", outnorm=False):
        x_new = torch.cat([x, x_new], dim=1)
        score = self.score_estimator(x_new)
        return self.fus(x_new * score) + self.skip(x_new)
    

    
import warnings
from typing import Callable, List, Optional, Tuple
if  hasattr(F, "scaled_dot_product_attention"):
    FLASH_AVAILABLE = True
else:
    FLASH_AVAILABLE = False

class Attention(nn.Module):
    def __init__(self, allow_flash=True) -> None:
        super().__init__()
        if allow_flash and not FLASH_AVAILABLE:
            warnings.warn(
                "FlashAttention is not available. For optimal speed, "
                "consider installing torch >= 2.0 or flash-attn.",
                stacklevel=2,
            )
        self.enable_flash = allow_flash and FLASH_AVAILABLE
        self.has_sdp = hasattr(F, "scaled_dot_product_attention")
        if self.has_sdp:
            torch.backends.cuda.enable_flash_sdp(allow_flash)

    def forward(self, q, k, v, mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        args = [x for x in [q, k, v]]
        c = q.shape[-1]
        n = k.shape[-2]
        #print(k.shape)
        scale = math.log(n)/(math.log(default_length)*c**0.5)
        v = F.scaled_dot_product_attention(*args, attn_mask=mask,scale=scale).to(q.dtype)
        return v if mask is None else v.nan_to_num()


class Wqk_l2norm_1d(nn.Module):
    def __init__(self, dim,head=2,oc=-1):
        super().__init__()
        if oc<0:
            oc = dim
        self.head = head
        self.conv1 = nn.Conv1d(dim,dim,1,bias=True)
        self.conv2 = nn.Sequential(nn.Conv1d(2*dim,2*oc,1,bias=False),nn.BatchNorm1d(2*oc,momentum=0.01))
    def forward(self,x):
        b,c,n = x.shape
        x_norm = F.normalize(self.conv1(x).reshape(b,self.head,-1,n),dim=2,eps=1e-3).reshape(b,-1,n)
        x = torch.cat([x,x_norm],dim=1)
        return self.conv2(x)

class Wqk_l2norm_2d(nn.Module):
    def __init__(self, dim, head=2,oc=-1):
        super().__init__()
        if oc<0:
            oc = dim
        self.head = head
        self.conv1 = nn.Conv2d(dim,dim,1,bias=True)
        self.conv2 = nn.Sequential(nn.Conv2d(2*dim,2*oc,1,bias=False),nn.BatchNorm2d(2*oc,momentum=0.01))
    def forward(self,x):
        b,c,h,w = x.shape
        x_norm = F.normalize(self.conv1(x).reshape(b,self.head,-1,h,w),dim=2,eps=1e-3).reshape(b,-1,h,w)
        x = torch.cat([x,x_norm],dim=1)
        return self.conv2(x)
    
class CrossTrans(nn.Module):
    def __init__(self, ic, dim: int, num_heads: int=2,
                 flash: bool = True, bias: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        dim_head = dim // num_heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.extra_token_v = nn.parameter.Parameter(torch.zeros(1,self.num_heads,1,self.dim_head))
        self.extra_token_k = nn.parameter.Parameter(torch.zeros(1,self.num_heads,1,self.dim_head))
        self.Wqk = Wqk_l2norm_1d(dim=dim,head=num_heads)
        self.Wv = nn.Conv1d(dim,dim,1,bias=False)
        self.Wv_norm = nn.Sequential(nn.InstanceNorm1d(dim,affine=False),nn.Conv1d(dim,dim,1))
        self.fuser = fuser(ic, dim, dim, False)
        self.dim3 = dim*3
        self.dim2 = dim*2
        self.dim = dim
        if flash and FLASH_AVAILABLE:
            self.flash = Attention(True)
        else:
            self.flash = None

    def forward_(self,x0, q, k, v):
        B,C,H,W = x0.shape
        k = torch.cat([k,self.extra_token_k.repeat(B,1,1,1)],dim=-2)
        v = torch.cat([v,self.extra_token_v.repeat(B,1,1,1)],dim=-2)
        q = self.flash(q, k, v)
        q = q.transpose(-1, -2).reshape(B,-1,H,W)
        x0 = self.fuser(x0,q)
        return x0

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> List[torch.Tensor]:
        B = x0.shape[0]
        x0_flatten,x1_flatten = x0.flatten(-2),  x1.flatten(-2)
        n, m = x0_flatten.shape[-1], x1_flatten.shape[-1]
        x = torch.cat([x0_flatten, x1_flatten], dim=-1)
        qk = self.Wqk(x)
        v = (self.Wv(x)+self.Wv_norm(x)).reshape(B, self.num_heads, self.dim_head,-1).transpose(-1,-2)
        qk = qk.reshape(B, 2, self.num_heads, self.dim_head,-1).transpose(-1,-2)
        q,k = qk[:,0], qk[:,1]
        x0 = self.forward_(x0,q[:,:,:n],k[:,:,n:],v[:,:,n:])
        x1 = self.forward_(x1,q[:,:,n:],k[:,:,:n],v[:,:,:n])
        return x0, x1


class SelfTrans(nn.Module):
    def __init__(self, dim: int, num_heads: int=4,
                 flash: bool = True, bias: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        dim_head = dim // num_heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.extra_token_v_ = nn.parameter.Parameter(torch.zeros(1,self.num_heads,1,self.dim_head))
        self.extra_token_k_ = nn.parameter.Parameter(torch.zeros(1,self.num_heads,1,self.dim_head))
        self.Wqk = Wqk_l2norm_2d(dim=dim,head=num_heads)
        self.Wv = nn.Conv2d(dim,dim,1,bias=True)

        self.fus = fuser(dim,dim,dim,False)
        self.dim3 = dim*3
        self.dim2 = dim*2
        if flash and FLASH_AVAILABLE:
            self.flash = Attention(True)
        else:
            self.flash = None
        self.running_step = 0
    def forward(self, x0: torch.Tensor) -> List[torch.Tensor]:
        B,C,H,W = x0.shape

        qk = self.Wqk(x0)
        qk = qk.reshape(B, 2, self.num_heads, self.dim_head,-1).transpose(-1,-2)
        v = self.Wv(x0).reshape(B, self.num_heads, self.dim_head,-1).transpose(-1,-2)
        q,k = qk[:,0], qk[:,1]
        k = torch.cat([k,self.extra_token_k_.repeat(B,1,1,1)],dim=-2)
        v = torch.cat([v,self.extra_token_v_.repeat(B,1,1,1)],dim=-2)
        q = self.flash(q, k, v)
        q = q.transpose(-1, -2).reshape(B,-1,H,W)
        x0 = self.fus(x0,q)
        return x0

import numpy as np
def gaussian_kernel(size=9, sigma=1.0):
    """Generates a Gaussian kernel."""
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * 
                     np.exp(-((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (size, size)
    )
    return kernel / np.sum(kernel)



from torch import nn, Tensor
from natten.functional import na2d,na2d_qk,na2d_av
class LocTrans(nn.Module):
    """
    Neighborhood Attention 2D Module
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 2,
        kernel_size: list = [5,5],
        dilation: int = 1,
        qk_scale: Optional[float] = None,
    ):
        super().__init__()
        if dim//num_heads < 32:
            num_heads = 1
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        self.kernel_size = kernel_size
        self.dim3 = dim*3
        self.dim2 = dim*2
        self.dilation = dilation

        self.Wqk = Wqk_l2norm_2d(dim=dim,head=num_heads)
        self.Wv = nn.Conv2d(dim,dim,1,bias=True)

        self.fus = fuser(dim,dim,dim,False)
        kernel = torch.from_numpy(gaussian_kernel(9,1.5)).reshape(1,9,9).repeat(2,1,1)
        self.rpb_ = nn.Parameter(
            kernel
        )

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        qk = self.Wqk(x)
        qk = qk.reshape(B,2,self.num_heads,self.head_dim,H,W).permute(0,1,2,4,5,3)
        q,k = qk[:,0], qk[:,1]
        rpb = self.rpb_
        attn = na2d_qk(
            q,
            k,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            rpb=rpb,
        )
        attn = attn.softmax(dim=-1)
        v = self.Wv(x).reshape(B,self.num_heads,self.head_dim,H,W).permute(0,1,3,4,2)
        q = na2d_av(
            attn,
            v,
            kernel_size=self.kernel_size,
            dilation=self.dilation
        )
        q = q.permute(0, 1, 4, 2, 3).reshape(B, C, H, W)
        x = self.fus(x,q)
        return x
    
class CBAM(nn.Module):
    def __init__(self,ic,oc,stride=2,drop=False) -> None:
        super().__init__()
        self.down_sample = BasicLayer(ic,oc,3,stride=stride)
        self.skip_down = nn.Conv2d(ic,oc,3,stride=stride,padding=1)
        self.mid_layer2 = BasicDWLayer(oc,oc)
        self.mid_layer1 = BasicDWLayer(oc,oc)
        self.channel_score = nn.Sequential(
                                            BasicLayer(oc*2,oc*2,kernel_size=1,padding=0),
                                            BasicLayer(oc*2,oc,kernel_size=1,padding=0,relu=False,obn=True),
                                            nn.Sigmoid())
        
        self.skip = nn.Conv2d(oc,oc,1)
        self.spatial_score = nn.Sequential(
                                           nn.Conv2d(oc,oc//4,3,padding=1, bias = False),
                                           nn.BatchNorm2d(oc//4, affine=True,momentum=0.01),
                                           nn.ReLU(),
                                           BasicDWLayer(oc//4,oc, relu=False),
                                           nn.BatchNorm2d(oc,momentum=0.01),
                                           nn.Sigmoid())

        self.output_layer = BasicLayer(oc,oc,relu=False,obn=True)
        
        self.maxpool1 = nn.AdaptiveMaxPool2d(1)
        self.avgpool1 = nn.AdaptiveAvgPool2d(1)

    
    def forward(self,x):
        skip1 = self.skip_down(x)
        x = self.down_sample(x)
        x_ = F.tanh(self.mid_layer1(x))
        channel_score = self.channel_score(torch.cat([self.maxpool1(x_),self.avgpool1(x_)],dim=1))
        x = x*channel_score
        spatial_score = self.spatial_score(x) 
        skip2 = self.skip(x)
        x = self.mid_layer2(x)
        x = x*spatial_score
        return self.output_layer(x)+skip1+skip2
        
class DownBLK(nn.Module):
    def __init__(self,ic,oc,stride=2,drop=False,out_norm=False) -> None:
        super().__init__()
        self.input_block = nn.Sequential(BasicLayer( ic, ic, dilation=2, padding=2),
                                        BasicLayer( ic, ic, dilation=4, padding=4),
                                        BasicLayer( ic, ic, dilation=8, padding=8),
                                        nn.Conv2d(ic, ic, 3,padding=1,bias=False),
                                        nn.BatchNorm2d(ic,momentum=0.01)
                                        )
        self.skip1 = nn.Conv2d(ic,ic,1)
        self.CBAM_down = CBAM(ic,oc,stride=stride,drop=drop)
        self.out_norm = out_norm
        
    def forward(self,x):
        x = self.input_block(x)+self.skip1(x)
        x = self.CBAM_down(x)
        return x
    

    
def kde(x, std = 0.1):
    # use a gaussian kernel to estimate density
    x = x.half() # Do it in half precision TODO: remove hardcoding
    scores = (-torch.cdist(x,x)**2/(2*std**2)).exp()
    density = scores.sum(dim=-1)
    return density

