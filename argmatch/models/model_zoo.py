

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from pathlib import Path
import numpy as np
from torch import nn
from PIL import Image
import torchvision.transforms.functional as TF
import argmatch
from argmatch.models.matcher import LocMatcher,GloMatcher,ContextInit
from argmatch.models.encoder import Encoder
from argmatch.models.roma_head import DenseHead
from argmatch.utils import create_meshgrid
from argmatch.models.modules import kde
import cv2
import numpy as np

class ArgMatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample=False
        self.symmetric=False
        
        self.h = 608
        self.w = 800
        
        self.encoder = Encoder()
        
        self.loc_refiner_4 = LocMatcher(128, 88, 80, 2, 0, False, 4)
        self.loc_refiner_8 = LocMatcher(160, 104, 96, 2, 88, True, 8)
        self.loc_refiner_16 = LocMatcher(192, 120, 112, 5, 104, True, 16)
        self.glo_matcher_16 = GloMatcher(224,216,128)
        self.context_init = ContextInit(224, 120)
        self.AP = nn.AvgPool2d(11,stride=1)
        self.sample_thresh = 0.05
        self.consistent_r = 5
        self.register_buffer('local_window',create_meshgrid(self.consistent_r,self.consistent_r,'cpu').reshape(1,-1,2).unsqueeze(1).unsqueeze(1),False)
        self.minimal_scale = 2
    
    @property
    def device(self):
        return self.encoder.block1[-1].layer[0].weight.device

    @torch.inference_mode()
    def match(self, im0, im1, batched = True, to_origin_reso=False, *args):
        device = self.device
        if isinstance(im0, (str, Path)):
            im0, im1 = self.load_from_path(im0, im1)
        elif isinstance(im0, Image.Image):
            im0 = TF.to_tensor((im0))[None].to(device)
            im1 = TF.to_tensor((im1))[None].to(device)
            
        B,C,H0,W0 = im0.shape 
        im0_ = TF.resize(im0,(self.h,self.w))
        im1_ = TF.resize(im1,(self.h,self.w))
        self.eval()
        corresps = self.forward({"im_A":im0_, "im_B":im1_})
            
        outscale = min(list(corresps.keys()))
        flow = corresps[outscale]["flow"].permute(0,2,3,1)
        certainty = corresps[outscale]["certainty"]
        if to_origin_reso:
            flow = F.interpolate(
                corresps[outscale]["flow"], 
                size = (H0, W0), 
                mode = "bilinear", align_corners = False).permute(0,2,3,1).reshape(-1,H0,W0,2)
            certainty = F.interpolate(corresps[outscale]["certainty"], size = (H0,W0), mode = "bilinear", align_corners = False)
        _,H_flow,W_flow,_ = flow.shape
        grid = create_meshgrid(H_flow,W_flow,device=flow.device).expand(B,H_flow,W_flow,2)
        if not self.symmetric:
            warp, cert = torch.cat((grid, flow), dim = -1), certainty[:,0]
        else:
            warp0 = torch.cat([grid[:B],flow[0:B]],dim=-1)
            warp1 = torch.cat([flow[B:],grid[:B]],dim=-1)
            warp = torch.cat([warp0,warp1],dim=2)
            cert = torch.cat([certainty[0:B],certainty[B:]],dim=3)[:,0]
        if batched:
            return warp, cert
        else:
            return warp[0], cert[0]

    def sym_consistent_check(self,certainty,matches,believe_thres = 0.1):
        certainty = certainty.unsqueeze(1)
        _,h,w,_ = matches.shape
        half_w = w//2
        thres =10/(h*half_w)**0.5 #5px
        matches12 = matches[:,:,:half_w]
        matches21 = matches[:,:,half_w:]
        
        image_size = torch.FloatTensor([half_w,h])[None,None,None].to(matches.device)
        scale = (2/image_size)
        
        local_window = self.local_window*scale.unsqueeze(-2) 
        local_window = matches12[:,:,:,2:].unsqueeze(-2).detach()+local_window
        local_window = local_window.reshape(1,h,-1,2) #b*h*wrr*2
        mask = (local_window<-1) | (local_window>1)
        mask = mask[...,0] | mask[...,1]
        mask = mask.reshape(1,h,half_w,-1)
        
        certainty_2_warp  = F.grid_sample(certainty[:,:,:,half_w:], matches12[...,-2:], mode='bilinear',padding_mode='zeros',align_corners=False) #b*c*h*wrr
        believe2 = (certainty_2_warp>believe_thres)
        
        warped_matches21 = F.grid_sample(matches21[...,:2].permute(0,3,1,2), local_window, mode='bilinear',padding_mode='zeros',align_corners=False).reshape(1,2,h,half_w,-1) #b*c*h*wrr
        dist = (warped_matches21-matches12[...,:2].permute(0,3,1,2).unsqueeze(-1)).norm(dim=1)
        consistency221 = (dist<thres).count_nonzero(dim=-1)
        inconsistent_mask1 = (consistency221<self.consistent_r)&believe2[:,0] #consistent score is lower than a threshold and the confidence of the corresponding region is high, then query matches might be wrong
        
        local_window = self.local_window*scale.unsqueeze(-2) 
        local_window = matches21[:,:,:,:2].unsqueeze(-2).detach()+local_window
        local_window = local_window.reshape(1,h,-1,2) #b*h*wrr*2
        mask = (local_window<-1) | (local_window>1)
        mask = mask[...,0] | mask[...,1]
        mask = mask.reshape(1,h,half_w,-1)
        
        certainty_1_warp  = F.grid_sample(certainty[:,:,:,:half_w], matches21[...,:2], mode='bilinear',padding_mode='zeros',align_corners=False)
        believe1 = (certainty_1_warp>believe_thres)
        
        warped_matches12 = F.grid_sample(matches12[...,-2:].permute(0,3,1,2), local_window, mode='bilinear',padding_mode='zeros',align_corners=False).reshape(1,2,h,half_w,-1) #b*c*h*wrr
        dist = (warped_matches12-matches21[:,:,:,2:].permute(0,3,1,2).unsqueeze(-1)).norm(dim=1)
        consistency122 = (dist<thres).count_nonzero(dim=-1)
        inconsistent_mask2 = (consistency122<self.consistent_r)&believe1[:,0]
        return torch.cat([inconsistent_mask1,inconsistent_mask2],dim=-1)

    @torch.inference_mode()
    def sample(
        self,
        matches,
        certainty,
        num=5000,
        expansion_factor = 4
    ):  
        num_samples_first_round = num*expansion_factor
        certainty_raw = certainty.clone()
        certainty = certainty.sigmoid()
        mask_th = certainty>self.sample_thresh
        
        if self.symmetric:
            consistent_mask=self.sym_consistent_check(certainty_raw.sigmoid(),matches)&mask_th
            
        certainty[mask_th] = 1
        if self.symmetric:
            certainty[consistent_mask] = self.sample_thresh*2
        mask_boundary = (matches>1) | (matches<-1)
        mask_boundary = mask_boundary.count_nonzero(dim=-1)>0
        certainty[mask_boundary] = 0

        mask = certainty
        b,h,w = certainty.shape
        if self.symmetric:
            nms_mask = (certainty_raw.sigmoid()).clamp(min=1e-3).reshape(1,h,2,w//2)
            nms_mask = nms_mask.permute(0,2,1,3).reshape(2,1,h,w//2)
            nms_mask_ap = F.pad(nms_mask,[5,5,5,5],value=1)
            nms_mask = (nms_mask/self.AP(nms_mask_ap)).permute(1,2,0,3).reshape(1,h,w)
            mask = nms_mask*mask
        else:
            nms_mask = (certainty_raw.sigmoid()).clamp(min=1e-3)
            nms_mask = nms_mask.unsqueeze(1)
            nms_mask_ap = F.pad(nms_mask,[5,5,5,5],value=1)
            nms_mask = (nms_mask/self.AP(nms_mask_ap)).reshape(1,h,w)
            mask = nms_mask*mask

        matches, mask, certainty = (
            matches.reshape(-1, 4),
            mask.reshape(-1),
            certainty.reshape(-1),
        )
        
        good_samples = torch.multinomial(mask.float(), 
                        num_samples = num_samples_first_round, 
                        replacement=False)
        good_matches, good_certainty = matches[good_samples], certainty[good_samples]
    
        density = kde(good_matches, std=0.1)
        p = (1 / (density+1))
        p[density <5] *= 0.01
        if torch.any(torch.isnan(p)):
            p = torch.zeros_like(p)+1e-7
        balanced_samples = torch.multinomial(p, 
                          num_samples = min(num,len(good_certainty)), 
                          replacement=False)
        good_matches = good_matches[balanced_samples]
        good_certainty = good_certainty[balanced_samples] 
        # self.draw_kpt(certainty_raw.sigmoid(),good_matches)
        return good_matches, good_certainty

    def draw_kpt(self,img,matches):
        c, h, w = img.shape
        img = img.permute(1,2,0).repeat(1,1,3)
        kpts = matches[:,:2].cpu().numpy()
        img = (img*255).cpu().numpy().astype(np.uint8)
        kpts = np.hstack([(kpts[:,0:1]+1)*w/4,(kpts[:,1:2]+1)*h/2])
        for i in range(kpts.shape[0]):
            img = cv2.circle(img,[int(kpts[i][0]),int(kpts[i][1])],radius=1, color=(0, 255, 0))
        Image.fromarray(img).save("kpt.png")
        
    def to_pixel_coordinates(self, coords, H_A, W_A, H_B = None, W_B = None):
        if coords.shape[-1] == 2:
            return self._to_pixel_coordinates(coords, H_A, W_A) 
        
        if isinstance(coords, (list, tuple)):
            kpts_A, kpts_B = coords[0], coords[1]
        else:
            kpts_A, kpts_B = coords[...,:2], coords[...,2:]
        return self._to_pixel_coordinates(kpts_A, H_A, W_A), self._to_pixel_coordinates(kpts_B, H_B, W_B)

    def _to_pixel_coordinates(self, coords, H, W):
        kpts = torch.stack((W/2 * (coords[...,0]+1), H/2 * (coords[...,1]+1)),axis=-1)
        return kpts

    @torch.inference_mode()
    def load_from_path(self, im0_path, im1_path):
        device = self.device
        im0 = TF.to_tensor(Image.open(im0_path))[None].to(device)
        im1 = TF.to_tensor(Image.open(im1_path))[None].to(device)
        return im0,im1

    def forward(self, batch):
        """
            input:
                x -> torch.Tensor(B, C, H, W) grayscale or rgb images
            return:

        """
        im0 = batch["im_A"]
        im1 = batch["im_B"]
        corresps = {}
        with torch.autocast(device_type="cuda",dtype=torch.float16):
            feat0_4, feat0_8, feat0_16 = self.encoder(im0)
            feat1_4, feat1_8, feat1_16 = self.encoder(im1)
        
        with torch.autocast(device_type="cuda",dtype=torch.float16):
            if self.symmetric:
                flow_16,flow_16_old, certainty_16 = self.glo_matcher_16.forward_symetric(feat0_16[:,:224],feat1_16[:,:224])
            else:
                flow_16, flow_16_old, certainty_16 = self.glo_matcher_16.forward(feat0_16[:,:224],feat1_16[:,:224])
            
            if self.symmetric:
                feat0_16, feat1_16 = torch.cat([feat0_16,feat1_16],dim=0), torch.cat([feat1_16,feat0_16],dim=0)
                feat0_8, feat1_8 = torch.cat([feat0_8,feat1_8],dim=0), torch.cat([feat1_8,feat0_8],dim=0)
                feat0_4, feat1_4 = torch.cat([feat0_4,feat1_4],dim=0), torch.cat([feat1_4,feat0_4],dim=0)

        
            context,certainty_16 = self.context_init(flow_16, certainty_16,
                                        feat0_16[:,:224],feat1_16[:,:224])

        flow_16 = flow_16.float()
        flow_16_old = flow_16_old.float()
        certainty_16 = certainty_16.float()
        with torch.autocast(device_type="cuda",dtype=torch.float32):
            flow_8, inter_flows_8, alphas_8, context, certainty_8 = self.loc_refiner_16(feat0_16[:,-192:],feat1_16[:,-192:], flow_16, certainty_16,  context)

            flow_4, inter_flows_4, alphas_4, context, certainty_4 = self.loc_refiner_8(feat0_8,feat1_8,flow_8, certainty_8, context)

            flow_2, inter_flows_2, alphas_2, context, certainty_2 = self.loc_refiner_4(feat0_4,feat1_4,flow_4, certainty_4, context)

        corresps[16] = {"flow":flow_16, "flow_old":flow_16_old, "certainty":certainty_16[:,:1], "overlap": certainty_16[:,1:] }

        corresps[8] = {"flow": flow_8, "certainty": certainty_8[:,:1], "overlap":certainty_8[:,1:], "inter_flows":inter_flows_8, "alphas":alphas_8}
        
        corresps[4] = {"flow": flow_4, "certainty": certainty_4[:,:1], "overlap":certainty_4[:,1:], "inter_flows":inter_flows_4, "alphas":alphas_4}

        corresps[2] = {"flow": flow_2, "certainty": certainty_2[:,:1], "overlap":certainty_2[:,1:], "inter_flows":inter_flows_2, "alphas":alphas_2}
        
        batch['flag'] = 0
        return corresps 
    
from collections import OrderedDict

class ArgMatch_plus(ArgMatch):
    def __init__(self):
        super().__init__()   
        self.FullResoHead = DenseHead()
        
    def load_state_dict(self, state_dict, strict: bool = True):

        if isinstance(state_dict, OrderedDict):
            filtered = OrderedDict()
        else:
            filtered = {}
        for k, v in state_dict.items():
            if not k.startswith("FullResoHead."):
                filtered[k] = v
        return super().load_state_dict(filtered, strict=strict)
    
    def forward(self, batch):
        corresps = super().forward(batch)
        corresps[1] = self.FullResoHead(corresps[2]['flow'])
        return corresps