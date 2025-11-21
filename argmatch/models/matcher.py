
import os
import math
from pickle import FALSE
from git import Tree
import numpy as np
from sympy import true
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.nn as nn
from argmatch.models.modules import *
from argmatch.utils import *
import torch.nn.utils.parametrize as parametrize
import random

class drop2d(nn.Module):
    def __init__(self, p=0.01) -> None:
        super().__init__()
        self.p = p
    def forward(self,x):
        if self.training:
            mask = torch.rand_like(x[:,0:1])>self.p
            x = mask*x
        return x
loc_context_dim = 16
class scale_estimator(nn.Module):
    def __init__(self, dim_embed = 128,) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.Conv2d(10+dim_embed,18,3,padding=1,bias=False),
                                    nn.BatchNorm2d(18,momentum=0.01))
        
        self.scale_decoder = nn.Sequential(BasicDWLayer(27+dim_embed+5,32,relu=False),
                                     BasicMulLayer(32,16,relu=False),
                                     BasicDWLayer(16,8,relu=False),
                                     BasicMulLayer(8,2,relu=False),nn.Sigmoid()
        )
    
    def forward(self, flow, certainty, context, image_size,stage):
        b,c,h,w = flow.shape
        flow = torch.cat([flow,certainty.detach()],dim=1)
        flow_unfold = F.unfold(flow, [3,3], padding=1)
        flow_unfold = flow_unfold.view(b, 3, 9, h, w)
        certainty_diff = flow_unfold[:,-1]
        basic_size = torch.FloatTensor([w,h]).reshape(1,2,1,1).to(flow.device)
        flow_diff = (flow_unfold[:,:2]-flow[:,:2].unsqueeze(2)).abs().detach() #2*9
        context = torch.cat([context,certainty_diff,certainty],dim=1)
        score = self.score(context).reshape(b,2,-1,h,w).softmax(dim=2)
        # score = score/(score.sum(dim=2,keepdim=True).clamp(min=1e-3))
        scale = score*flow_diff[:,:2]
        ratio = image_size/basic_size
        ratio = torch.cat([ratio,basic_size*stage],dim=1).repeat(b,1,h,w)
        scale = self.scale_decoder(torch.cat([context,scale.reshape(b,-1,h,w),ratio],dim=1)) #scale is non-dimensional
        return scale*3+0.5
    
class cv_decoder(nn.Module):
    def __init__(self,dim_embed, cv_len, r):
        super().__init__()
        self.r = r
        self.content_encoding = nn.Sequential(
                                    BasicMulLayer(dim_embed+2+36,dim_embed,relu=False),
                                    BasicLayer(dim_embed,dim_embed,kernel_size=1,padding=0,relu=False))
        
        self.cv_mapping = nn.Sequential(
                                    BasicMulLayer(cv_len+dim_embed,cv_len*2+loc_context_dim,relu=False),
                                    nn.LeakyReLU(negative_slope=0.05),
                                    BasicMulLayer(cv_len*2+loc_context_dim,cv_len*2+loc_context_dim,relu=False),
                                    nn.LeakyReLU(negative_slope=0.05),
                                    BasicMulLayer(cv_len*2+loc_context_dim,cv_len*2+loc_context_dim,relu=False))
        
        self.cv_skip_weight = nn.parameter.Parameter((torch.zeros([cv_len,2*cv_len])-2).reshape(1,cv_len,2*cv_len))
        
        self.offset_decoder = nn.Sequential(BasicMulLayer(dim_embed,cv_len*2,relu=False),
                                            BasicMulLayer(cv_len*2,cv_len*2,relu=False),
                                            nn.Conv2d(cv_len*2,cv_len*2,1))
        self.cv_len = cv_len
        self.cv_bn = nn.BatchNorm3d(2,momentum=0.01)
        # self.certainty_vec_bn = nn.BatchNorm2d(loc_context_dim)
    def forward(self, context, cv, scale, sv):
        b,c,h,w = cv.shape
        z = self.content_encoding(torch.cat([context, scale, sv],dim=1))
        cv_new = self.cv_mapping(torch.cat([z,cv],dim=1))
        cv_new, certainty_vec = cv_new[:,:-loc_context_dim], cv_new[:,-loc_context_dim:]
        cv_new = (cv_new+torch.einsum("bcm,bchw->bmhw",F.softplus(self.cv_skip_weight).repeat(z.shape[0],1,1),cv)).reshape(b,2,c,h,w)
        cv_new = F.softmax(self.cv_bn(cv_new),dim=2)
        offset = self.offset_decoder(z).reshape(b,2,c,h,w)
        disp = (offset*cv_new).sum(dim=2)
        return disp, certainty_vec
    
    
class ResOffsetEstimator(nn.Module):
    def __init__(self,dim_feat,dim_embed,dim_hidden, r, scale) -> None:
        super().__init__()
        cv_len = (2*r+1)**2
        extra_dim = 32
        self.scale_estimator = scale_estimator(dim_hidden//2)
        self.compress_org_feat01 = compressor(dim_feat, dim_hidden+extra_dim)
        self.compress_org_feat02 = compressor(dim_feat, dim_hidden+extra_dim)
        
        self.compress_org_feat1_cv = compressor_light(dim_feat,dim_hidden)
        self.compress_org_feat2_cv = compressor_light(dim_feat,dim_hidden+3)
        
        self.compress_cont_feat1 = compressor_light(dim_embed+dim_feat,dim_embed)
        
        
        self.fus_cont_feat = fuser_1D(dim_embed, 2*dim_hidden, 2*dim_hidden)
        self.fus_cont_old_new_ = fuser_1D(dim_embed+loc_context_dim, dim_hidden*2, dim_embed+loc_context_dim)

        self.compress_feat12 = nn.Sequential(BasicLayer(2*dim_hidden+extra_dim*2+1,2*dim_hidden,kernel_size=1, padding=0),
                                             BasicLayer(2*dim_hidden,2*dim_hidden,kernel_size=1,padding=0),
                                             BasicLayer(2*dim_hidden,2*dim_hidden,1, padding=0, relu=False, obn=True))
        self.scale_t = scale/512
        self.scale = scale/16
        
        self.r = r
        self.cv_decoder = cv_decoder(dim_hidden,cv_len,r)
        self.register_buffer('local_window',create_meshgrid(2*r+1,2*r+1,'cpu').reshape(1,-1,2).unsqueeze(1).unsqueeze(1),False)
        self.dim_hidden = dim_hidden 
        if r == 2:
            self.resample_idx = [12,0,2,4,10,14,20,22,24]
        if r == 5:
            self.resample_idx = [60,0,5,10,55,65,110,115,120]
            
    def local_corr(self,
        feature0,
        feature1,
        flow = None,
        scale = 1,
        image_size=1,
    ):
        feature1 = self.compress_org_feat2_cv(feature1)
        r = self.r
        B, C, h, w = feature0.size()
        flow = flow.permute(0,2,3,1)  #flow b*h*w*2
        scale = (r*2/image_size*scale).permute(0,2,3,1).contiguous()
        local_window = self.local_window*scale.unsqueeze(-2) 
        local_window = flow.unsqueeze(-2).detach()+local_window
        local_window = local_window.reshape(B,h,-1,2) #b*h*wrr*2
        mask = (local_window<-1) | (local_window>1)
        mask = mask[...,0] | mask[...,1]
        mask = mask.reshape(B,h,w,-1).permute(0,3,1,2)
        sampled_features = F.grid_sample(feature1, local_window, align_corners=False,mode='bilinear').reshape(B,feature1.shape[1],h,w,-1) #b*c*h*wrr
        resample_feat = sampled_features[:,:,:,:,self.resample_idx]
        SV = (resample_feat[...,0].unsqueeze(-1)*resample_feat).sum(dim=1,keepdim=True).detach()
        cv_scale = feature0.shape[1]*10
        SV = torch.cat([resample_feat[:,:3],(F.tanh(SV/(cv_scale)))],dim=1).permute(0,1,4,2,3).reshape(B,-1,h,w)
        cv = torch.einsum('bchw,bchwr->bhwr', feature0, sampled_features[:,3:])/cv_scale
        cv = cv.permute(0,3,1,2)
        cv = F.tanh(cv).masked_fill(mask,-1)*10
        return cv,SV
    
    def couple(self, offset, flow_old, alpha, flag):
        if self.training:
            thres = 0.1
            mask = (alpha>thres).float()
            alpha = 0.01*(alpha-alpha.detach())
            flow_new_sup = flow_old.detach()+offset
            flow_new = flow_old.detach()+offset if flag else 0.9*flow_old.detach()+0.1*flow_old+offset
            flow_new = flow_new*(mask+alpha)+flow_old*(1-mask-alpha)
        else:
            thres = .1
            mask = (alpha>thres)
            flow_new_sup = None
            flow_new = flow_old+offset
            flow_new = flow_new*(mask)+flow_old*(~mask)
        return flow_new_sup, flow_new
    
    
    def disp_estimate(self,context_new, feat1, feat2,flow,overlap, img_size):  
        feat1 = context_new[:,:self.dim_hidden]+self.compress_org_feat1_cv(feat1)
        scale = self.scale_estimator(flow, overlap, context_new[:,-self.dim_hidden//2:], img_size, self.scale_t)
        
        cv,SV = self.local_corr(feat1,feat2,flow,scale, img_size)
        disp, certainty = self.cv_decoder(context_new[:,-self.dim_hidden:], cv, scale, SV)
        return disp, certainty
    
    def update_context(self,feat1,feat2,context,overlap,flow):
        context = self.compress_cont_feat1(torch.cat([context,feat1],dim=1))
        feat2 = F.grid_sample(self.compress_org_feat01(feat2), flow.permute(0, 2, 3, 1), align_corners=False, mode = 'bilinear')
        feat2 = torch.cat([self.compress_org_feat02(feat1), feat2, overlap],dim=1)
        feat2 = self.compress_feat12(feat2)
        context = self.fus_cont_feat(context, feat2)
        return context, feat2
    
    def forward(self, flow, certainty, feat1, context, feat2, detach_flag):
        b,c,hy,wy = feat2.shape
        img_size = torch.FloatTensor([wy,hy])[None,:,None,None].to(feat1.device)
        overlap = F.tanh(certainty[:,1:]/10)
        context_new, feat_fused = self.update_context(feat1,feat2,context,overlap,flow.detach())
        disp, local_context = self.disp_estimate(context_new, feat1,
                                            feat2,flow.detach(), overlap,img_size)
        alpha = local_context[:,:1].sigmoid()
        local_context = F.tanh(local_context/10)*10
        context_new = self.fus_cont_old_new_(torch.cat([context,local_context],dim=1), feat_fused)
        flow_new_sup,flow_new = self.couple(disp/img_size,flow,alpha, detach_flag)
        return flow_new_sup, flow_new, alpha, certainty, context_new
    
class Rectifier(nn.Module):
    def __init__(self, dim_feat, dim_context, dim_hidden, scale) -> None:
        super().__init__()
        self.PE = PosGen(16,type=1)
        context_dim_ = dim_context + 16 
        self.compressor_ = compressor_light(dim_context+dim_feat+2+16+loc_context_dim, context_dim_)
        self.interp = nn.Sequential(
                                    BasicDWLayer(context_dim_,context_dim_,relu=False),
                                    LocTrans(context_dim_,dilation=2 if scale<8 else 1),
                                    BasicDWLayer(context_dim_,context_dim_,relu=False),
                                    LocTrans(context_dim_,dilation=2 if scale<16 else 1)
                                    )
        
        self.flow_decoder = nn.Sequential(BasicMulLayer(context_dim_+2,32,relu=False,relu_1=True),   
                                          ResBasicLayer(32,32,3,padding=1),
                                          ResBasicLayer(32,32,3,padding=1),         
                                          ResBasicLayer(32,2,3,padding=1),
                                        )
        
        self.alpha_estimator = nn.Sequential( ResBasicLayer(context_dim_+2+2,32,1,padding=0,obn=True),
                                        BasicMulLayer(32,32,relu=False),
                                        ResBasicLayer(32,2,1,padding=0,obn=True))
        
        self.fus_cont_old_new_ = fuser_1D(context_dim_, context_dim_, context_dim_)
        self.dim_context = dim_context
        self.cert_bn = nn.BatchNorm2d(2,momentum=0.01)

    
    def forward(self, flow, certainty, feat, context):
        context_new = self.compressor_(torch.cat([feat, context, F.tanh(certainty/10), self.PE(flow.detach())],dim=1))
        context_new = self.interp(context_new)
        certainty = certainty+self.cert_bn(context_new[:,0:2])
        flow_new = self.flow_decoder(torch.cat([context_new, flow.detach()],dim=1))
        alpha = self.alpha_estimator(torch.cat([context_new, flow.detach(),flow_new.detach()],dim=1))
        alpha1 = alpha[:,0:1]/4
        alpha2 = alpha[:,1:]/16-alpha1
        flow_new = (alpha1).sigmoid()*flow+(alpha2).sigmoid()*flow_new
        context_new = self.fus_cont_old_new_(context_new,context)
        return flow_new, certainty, context_new

from natten.functional import na2d_qk
class Upsampler(nn.Module):
    def __init__(self, dim_feat, dim_context, dim_hidden, out_context=True) -> None:
        super().__init__()
        self.PE = PosGen(16,type=1)
        self.compressor_ = compressor_light(dim_feat+dim_context+16+loc_context_dim+2, dim_hidden+loc_context_dim*2)
        self.Wqk_ = Wqk_l2norm_2d(dim_hidden+loc_context_dim,2,32)
        self.dim_hidden = dim_hidden
        self.zp = nn.ConstantPad2d(2,-1)
        self.score_estimator_ = nn.Sequential(BasicDWLayer(dim_hidden+loc_context_dim+25, 9*4*4),
                                             BasicMulLayer(9*4*4, 9*4*4),
                                             BasicLayer(9*4*4,9*4*3),
                                             BasicDWLayer(9*4*3,9*4*2),
                                             BasicMulLayer(9*4*2,9*4*2),
                                             BasicLayer(9*4*2,9*4*1+12,relu=False))
        self.score_bn = nn.BatchNorm3d(4,momentum=0.01)
        if out_context:   
            self.context_compress_ = BasicDWLayer(dim_hidden+loc_context_dim+25,dim_hidden,relu=False,skip=True)
            self.fus_cont_old_new = fuser_1D(dim_hidden, dim_context+loc_context_dim, dim_hidden)
        else:
            self.fus_cont_old_new = None
        self.cert_decoder = nn.Sequential(ResBasicLayer(2+2,16,3,padding=1,obn=True),
                                        BasicMulLayer(16,16,relu=False),
                                        ResBasicLayer(16,2,3,padding=1,obn=True),
                                        nn.BatchNorm2d(2,momentum=0.01))
        self.dim_hideen = dim_hidden
    def compute_cv(self,flow, certainty, feat, context):
        b,c,H,W = context.shape
        context_new = torch.cat([feat, context, F.tanh(certainty/10), self.PE(flow.detach())],dim=1)
        feat = self.compressor_(context_new)
        q = self.zp(self.Wqk_(feat[:,:self.dim_hidden+loc_context_dim])).permute(0,2,3,1).unsqueeze(1)
        q,k = q[...,:32],q[...,32:]
        cv = na2d_qk(q,k+q,kernel_size=5)
        cv = cv[:,0,2:-2,2:-2].reshape(b,H,W,-1).permute(0,3,1,2)
        return feat[:,-(self.dim_hidden+loc_context_dim):], cv


    def estimate_score(self, flow, certainty, feat, context):
        b,c,H,W = context.shape
        feat,cv = self.compute_cv(flow, certainty, feat, context)
        context_new = torch.cat([feat, cv],dim=1)
        score = self.score_estimator_(context_new)
        certainty = score[:,-12:]
        score = self.score_bn(score[:,:-12].view(b, 4, 9, H, W)).softmax(dim=2)
        if self.fus_cont_old_new is not None:
            context = self.fus_cont_old_new(self.context_compress_(context_new),context)
        return score, certainty, context
    
    
    def forward(self, flow, certainty, feat, context):
        # flow = flow
        """ Upsample flow field [H/2, W/2, 2] -> [H, W, 2] using convex combination """
        b, _, H, W = flow.shape
        score, certainty_new, context = self.estimate_score(flow.detach(), certainty, feat, context)
        score = score.reshape(b,1,2,2,9,H,W)
        
        flow_old = F.interpolate(flow, [2*H,2*W], mode='bicubic', align_corners=False)
        flow = F.unfold(flow, [3,3], padding=1)
        flow = flow.view(b, 2, 1, 1, 9, H, W)
        flow = torch.sum(score*flow, dim=4)
        flow = flow.permute(0, 1, 4, 2, 5, 3).reshape(b, 2, 2*H, 2*W)
        certainty_new = certainty_new.reshape(b, 3, 2, 2, H, W).permute(0,1,4,2,5,3).reshape(b, 3, 2*H, 2*W)
        alpha = certainty_new[:,0:1]/10
        flow = flow*(alpha).sigmoid()+flow_old*(-alpha).sigmoid()
        certainty_old = F.interpolate(certainty,[2*H,2*W], mode='bilinear', align_corners=False)
        certainty_new = self.cert_decoder(F.tanh(torch.cat([certainty_new[:,1:],certainty_old],dim=1)/3))+certainty_old
        return flow, certainty_new, context
        
class LocMatcher(nn.Module):
    def __init__(
        self,
        dim_feat = 64,
        dim_context = 128,
        dim_hidden = 64,
        local_corr_radius = None,
        out_embed_dim = 128,
        out_embed=True,
        scale = 4,
        sample_mode = "bilinear",

    ):
        super().__init__()
        self.scale = scale
        self.dim_feat = dim_feat
        self.delta_flow_estimator = ResOffsetEstimator(dim_feat, dim_context, dim_hidden, local_corr_radius, scale)
        self.flow_refiner = Rectifier(dim_feat, dim_context, dim_hidden, scale)
        self.flow_upsampler_x2 = Upsampler(dim_feat, dim_context, dim_hidden, out_embed)
        self.local_corr_radius = local_corr_radius
        self.sample_mode = sample_mode
        if out_embed:
            self.context_decoder = fuser(dim_hidden, dim_context, out_embed_dim) 
        else:
            self.context_decoder = None
    

    def forward(self, x, y, flow, certainty, context=None, detach_flag=False):
        b,c,hx,wx = x.shape
        b,c,hy,wy = y.shape

        if context.shape[2]!=hx or context.shape[3]!=wx:
            context = F.interpolate(context,scale_factor=2.0,mode='bilinear',align_corners=False)
        context_old = context
        
        flow_res_sup, flow_res, alpha, certainty, context = self.delta_flow_estimator(flow, certainty, x, context, y, detach_flag)

        flow_rec, certainty, context = self.flow_refiner(flow_res, certainty, x, context)
        
        flow_up, certainty, context = self.flow_upsampler_x2(flow_rec, certainty, x, context)

        if self.context_decoder is not None:
            context = self.context_decoder(context,context_old)
        else:
            context = None
            
        inter_flows = [flow_res_sup, flow_rec, flow_up]
        return flow_up, inter_flows, alpha, context, certainty 
    

class PosGen(nn.Module):
    def __init__(self,hidden_dim=32, type=0,feat_dim=-1,detach=True) -> None:
        super().__init__()
        if type==0:
            self.mapping = nn.Sequential(BasicMulLayer(2,16,relu=False,wo_norm=True,relu_1=True),
                                    BasicMulLayer(16,32,relu=False,wo_norm=True,relu_1=True),
                                    nn.Conv2d(32,hidden_dim,1))
            nn.init.zeros_(self.mapping[-1].weight)
        elif type==1:
            self.mapping = nn.Sequential(BasicMulLayer(4,16,relu=False,wo_norm=True,relu_1=True),
                                    BasicMulLayer(16,32,relu=False,wo_norm=True,relu_1=True),
                                    nn.Conv2d(32,hidden_dim-2,1))

        elif type==2:
            self.mapping = nn.Sequential(BasicMulLayer(feat_dim+2,feat_dim,relu=False,relu_1=True,wo_norm=True),
                                
                                    BasicMulLayer(feat_dim,hidden_dim-2,relu=False,relu_1=True,wo_norm=True))
        self.detach_flag = detach
        self.type=type
        
    def forward(self,feat):
        if self.detach_flag:
            feat = feat.detach()
        if self.type==0:
            b,c,h,w = feat.shape
            pos = create_meshgrid(h,w,device=feat.device).permute(0,3,1,2).contiguous()
            pos_hidden = self.mapping(pos+1)
            mask = torch.zeros_like(pos_hidden)
            mask[:,-16:] = 1
            pos_hidden = pos_hidden*mask
            return pos_hidden+feat
        if (self.type==1):
            b,c,h,w = feat.shape
            pos = create_meshgrid(h,w,device=feat.device).permute(0,3,1,2).contiguous().repeat(b,1,1,1)
            pos_hidden = self.mapping(torch.cat([feat.detach()+1,pos+1],dim=1))
            return torch.cat([pos_hidden,feat],dim=1)
        else:
            b,c,h,w = feat.shape
            pos = create_meshgrid(h,w,device=feat.device).permute(0,3,1,2).contiguous().repeat(b,1,1,1)
            pos_hidden = self.mapping(torch.cat([pos+1,feat],dim=1))
            return torch.cat([pos_hidden,pos],dim=1)
        
class GLAtten(nn.Module):
    def __init__(self,ic,oc) -> None:
        super().__init__()
        self.ca1 = CrossTrans(oc,oc)
        self.ca2 = CrossTrans(oc,oc)
        self.ca3 = CrossTrans(oc,oc)
        self.conv1 = compressor(ic, oc, obn=True, dropout=True)
        self.conv2 = compressor(oc, oc, obn=True)
        self.conv3 = compressor(oc, oc, obn=True)
        self.PE = PosGen(oc,0,detach=False)

        
    def forward(self,embed1,embed2):
        embed1, embed2 = self.conv1(embed1), self.conv1(embed2)
        embed1, embed2 = self.PE(embed1),self.PE(embed2)
        embed1, embed2 = self.ca1(embed1, embed2)
        embed1, embed2 = self.conv2(embed1), self.conv2(embed2)
        embed1, embed2 = self.ca2(embed1, embed2)
        embed1, embed2 = self.conv3(embed1), self.conv3(embed2)
        embed1, embed2 = self.ca3(embed1, embed2)
        
        return embed1, embed2
    

class Matching(nn.Module):
    def __init__(
        self,
        feat_dim=128,
        hidden_dim = 256,

    ):
        super().__init__()
        self.pos_emb = PosGen(32+2,2,feat_dim=feat_dim)
        self.flash_attn = Attention()
        self.dustbin_vec = nn.parameter.Parameter(torch.rand([1,1,1,256])/10)
        self.dustbin_pos = nn.parameter.Parameter(torch.zeros([1,1,1,34])-10)
        self.fuser = fuser_1D(feat_dim,hidden_dim,256,obn=False)
        self.conv_y = BasicDWLayer(256, 256, relu=False, wo_norm=True)
    def forward(self, x, y, feat1, feat2):
        b,c,hy,wy = y.shape
        b,c,hx,wx = x.shape
        pos = self.pos_emb(feat2.detach())
        c_pose = pos.shape[1]
        pos=pos.reshape(b,1,c_pose,-1).transpose(-1,-2)
        x = self.fuser(x,feat1)
        y = self.fuser(y,feat2)
        y = self.conv_y(y)
        c = y.shape[1]
        x = x.reshape(b,1,c,-1).transpose(-1,-2)
        y = y.reshape(b,1,c,-1).transpose(-1,-2)
        pos = torch.cat([pos, self.dustbin_pos.repeat(b,1,1,1)],dim=-2)
        y = torch.cat([y,self.dustbin_vec.repeat(b,1,1,1)],dim=-2)
        flow_embed = self.flash_attn(x,y,pos).transpose(-1,-2).reshape(b,-1,hx,wx).contiguous()
        return flow_embed
        
class GloMatcher(nn.Module):
    def __init__(
        self,
        feat_dim = 256,
        match_dim = 192,
        hidden_dim = 128,
    ):
        super().__init__()
        self.Compress = compressor_light(feat_dim+match_dim,hidden_dim-16)
        self.CrossAttn = GLAtten(feat_dim,  match_dim)
        
        self.SelfAttn = nn.Sequential(
            BasicDWLayer(hidden_dim+32,hidden_dim,relu=False),
            SelfTrans(hidden_dim),
            BasicDWLayer(hidden_dim,hidden_dim,relu=False),
            SelfTrans(hidden_dim),
            )
        self.cert_estimator = CertEst(match_dim,hidden_dim)


        self.flow_decoder = nn.Sequential(ResBasicLayer(hidden_dim+2,32,1,padding=0),          
                                          BasicMulLayer(32,32,relu=False,relu_1=True),
                                          ResBasicLayer(32,2,1,padding=0),
                                        )
        
        self.alpha_estimator_ = nn.Sequential( ResBasicLayer(hidden_dim+2+2,32,1,padding=0,obn=True),
                                        BasicMulLayer(32,32,relu=False),
                                        ResBasicLayer(32,4,1,padding=0,obn=True))
        
        self.MatchingLayer = Matching(feat_dim,match_dim)

        self.PE = PosGen(16,1)
    def forward(self,  feat0, feat1):

        embed0, embed1 = self.CrossAttn(feat0, feat1)
        coarse_match = self.MatchingLayer(embed0, embed1, feat0, feat1)
        flow_old = coarse_match[:,-2:]
        feat0 = torch.cat([embed0,feat0],dim=1)
        # feat0 = feat0*0.1+feat0.detach()*0.9
        feat0 = self.Compress(feat0)
        feat0 = torch.cat([feat0,coarse_match[:,:-2],self.PE(flow_old.detach())],dim=1)
        feat0 = self.SelfAttn(feat0)
        flow_new = self.flow_decoder(torch.cat([feat0,flow_old.detach()],dim=1))
        alpha = self.alpha_estimator_(torch.cat([feat0,flow_old.detach(),flow_new.detach()],dim=1))
        alpha1 = alpha[:,0:1]/4
        alpha2 = alpha[:,1:2]/16-alpha1
        flow_new = (alpha1).sigmoid()*flow_old+(alpha2).sigmoid()*flow_new
        certainty = self.cert_estimator(embed0,embed1,feat0,flow_new,alpha[:,2:])
        return flow_new,flow_old,certainty
    
    def forward_symetric(self,  feat0, feat1):
        embed0, embed1 = self.CrossAttn(feat0, feat1)
        embed0, embed1 = torch.cat([embed0,embed1],dim=0),torch.cat([embed1,embed0],dim=0)
        feat0,feat1 = torch.cat([feat0,feat1],dim=0),torch.cat([feat1,feat0],dim=0)
        coarse_match = self.MatchingLayer(embed0, embed1, feat0, feat1)
        flow_old = coarse_match[:,-2:]
        feat0 = torch.cat([embed0,feat0],dim=1)
        feat0 = self.Compress(feat0)
        feat0 = torch.cat([feat0,coarse_match[:,:-2],self.PE(flow_old.detach())],dim=1)
        feat0 = self.SelfAttn(feat0)
        flow_new = self.flow_decoder(torch.cat([feat0,flow_old.detach()],dim=1))
        alpha = self.alpha_estimator_(torch.cat([feat0,flow_old.detach(),flow_new.detach()],dim=1))
        alpha1 = alpha[:,0:1]/4
        alpha2 = alpha[:,1:2]/16-alpha1
        flow_new = (alpha1).sigmoid()*flow_old+(alpha2).sigmoid()*flow_new
        certainty = self.cert_estimator(embed0,embed1,feat0,flow_new,alpha[:,2:])
        return flow_new,flow_old,certainty
    
class CertEst(nn.Module):
    def __init__(
        self,
        feat_dim = 256,
        hidden_dim = 192,
    ):
        super().__init__()
        self.compress = BasicDWLayer(feat_dim,feat_dim//2,relu=False)
        self.cert_decoder = nn.Sequential( BasicDWLayer(hidden_dim+2+1+2,32),
                                        BasicMulLayer(32,32,relu=False),
                                        BasicDWLayer(32,2,relu=False))
    def forward(self,embed0,embed1,feat0,flow,alpha):
        flow = flow.detach()
        embed0 = self.compress(embed0)
        embed1 = self.compress(embed1)
        embed1 = F.grid_sample(embed1,flow.permute(0,2,3,1),mode='bilinear',align_corners=False)
        simi = (embed0*embed1).mean(dim=1,keepdim=True)
        mask = (flow.abs()>1).count_nonzero(dim=1)[:,None]
        simi = simi-10*mask
        cert= self.cert_decoder(torch.cat([F.tanh(simi), flow, feat0, alpha],dim=1))
        return cert

class ContextInit(nn.Module):
    def __init__(
        self,
        feat_dim = 256,
        output_dim = 192,
    ):
        super().__init__()
        self.compress_org_feat01 = compressor_light(feat_dim, output_dim)
        self.compress_org_feat02 = compressor_light(feat_dim, output_dim//2)
        self.fuser = fuser_1D(output_dim+1,output_dim//2,output_dim)
        self.compressor = compressor_light(output_dim+32+2,output_dim+2)
        self.PosGen = PosGen(32,type=1)
        self.mapping = nn.Sequential(nn.Conv2d(feat_dim,1,1),nn.Tanh())
    
    def forward(self,flow,certainty,feat1,feat2):
        feat2_resample = F.grid_sample(feat2,flow.permute(0,2,3,1),mode='bilinear',align_corners=False)
        simi = self.mapping(feat1*feat2_resample)
        mask = (flow.abs()>1).count_nonzero(dim=1)[:,None]
        simi = simi-10*mask
        feat1 = self.compress_org_feat01(feat1)
        feat2 = self.compress_org_feat02(feat2.detach())
        flow = flow.detach()
        feat2 = F.grid_sample(feat2,flow.permute(0,2,3,1),mode='bilinear',align_corners=False)
        feat1 = torch.cat([feat1,simi],dim=1)
        feat1 = self.fuser(feat1,feat2)
        flow = self.PosGen(flow)
        feat1= self.compressor(torch.cat([feat1,flow,F.tanh(certainty[:,:1:]/10),simi],dim=1))
        return feat1[:,2:],certainty+feat1[:,:2]




