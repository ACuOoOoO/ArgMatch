import torch
import torchvision.models as tvm
import torch.nn as nn
import torch.nn.functional as F

def get_autocast_params(device=None, enabled=False, dtype=None):
    if device is None:
        autocast_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        #strip :X from device
        autocast_device = str(device).split(":")[0]
    if 'cuda' in str(device):
        out_dtype = dtype
        enabled = True
    else:
        out_dtype = torch.bfloat16
        enabled = False
        # mps is not supported
        autocast_device = "cpu"
    return autocast_device, enabled, out_dtype

class VGG19(nn.Module):
    def __init__(self, pretrained=False, amp = False, amp_dtype = torch.float16) -> None:
        super().__init__()
        self.layers = nn.ModuleList(tvm.vgg19_bn(pretrained=pretrained).features[:14])
        self.amp = amp
        self.amp_dtype = amp_dtype

    def forward(self, x, **kwargs):
        autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(x.device, self.amp, torch.float16)
        with torch.autocast(autocast_device, enabled=autocast_enabled, dtype = autocast_dtype):
            feats = {}
            scale = 1
            for layer in self.layers:
                if isinstance(layer, nn.MaxPool2d):
                    feats[scale] = x
                    scale = scale*2
                x = layer(x)
        return feats


class encoder(nn.Module):
    def __init__(self, pretrained=False, amp = False, amp_dtype = torch.float16) -> None:
        super().__init__()
        self.cnn=VGG19(False,True)
    def forward(self, x):
        return self.cnn(x)
        
class ConvRefiner(nn.Module):
    def __init__(
        self,
        in_dim=6,
        hidden_dim=16,
        out_dim=2,
        dw=False,
        kernel_size=5,
        hidden_blocks=3,
        displacement_emb = None,
        displacement_emb_dim = None,
        local_corr_radius = None,
        corr_in_other = None,
        no_im_B_fm = False,
        amp = False,
        concat_logits = False,
        use_bias_block_1 = True,
        use_cosine_corr = False,
        disable_local_corr_grad = False,
        is_classifier = False,
        sample_mode = "bilinear",
        norm_type = nn.BatchNorm2d,
        bn_momentum = 0.1,
        amp_dtype = torch.float16,
    ):
        super().__init__()
        self.bn_momentum = bn_momentum
        self.block1 = self.create_block(
            in_dim, hidden_dim, dw=dw, kernel_size=kernel_size, bias = use_bias_block_1,
        )
        self.hidden_blocks = nn.Sequential(
            *[
                self.create_block(
                    hidden_dim,
                    hidden_dim,
                    dw=dw,
                    kernel_size=kernel_size,
                    norm_type=norm_type,
                )
                for hb in range(hidden_blocks)
            ]
        )
        self.hidden_blocks = self.hidden_blocks
        self.out_conv = nn.Conv2d(hidden_dim, out_dim, 1, 1, 0)
        if displacement_emb:
            self.has_displacement_emb = True
            self.disp_emb = nn.Conv2d(2,displacement_emb_dim,1,1,0)
        else:
            self.has_displacement_emb = False
        self.local_corr_radius = local_corr_radius
        self.corr_in_other = corr_in_other
        self.no_im_B_fm = no_im_B_fm
        self.amp = amp
        self.concat_logits = concat_logits
        self.use_cosine_corr = use_cosine_corr
        self.disable_local_corr_grad = disable_local_corr_grad
        self.is_classifier = is_classifier
        self.sample_mode = sample_mode
        self.amp_dtype = amp_dtype
        
    def create_block(
        self,
        in_dim,
        out_dim,
        dw=False,
        kernel_size=5,
        bias = True,
        norm_type = nn.BatchNorm2d,
    ):
        num_groups = 1 if not dw else in_dim
        if dw:
            assert (
                out_dim % in_dim == 0
            ), "outdim must be divisible by indim for depthwise"
        conv1 = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=num_groups,
            bias=bias,
        )
        norm = norm_type(out_dim, momentum = self.bn_momentum) if norm_type is nn.BatchNorm2d else norm_type(num_channels = out_dim)
        relu = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(out_dim, out_dim, 1, 1, 0)
        return nn.Sequential(conv1, norm, relu, conv2)
        
    def forward(self, x, y, flow, scale_factor = 1, logits = None):
        b,c,hs,ws = x.shape
        autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(x.device, enabled=self.amp, dtype=self.amp_dtype)
        with torch.autocast(autocast_device, enabled=autocast_enabled, dtype = autocast_dtype):            
            x_hat = F.grid_sample(y, flow.permute(0, 2, 3, 1), align_corners=False, mode = self.sample_mode)
            if self.has_displacement_emb:
                im_A_coords = torch.meshgrid(
                (
                    torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=x.device),
                    torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=x.device),
                ), indexing='ij'
                )
                im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]))
                im_A_coords = im_A_coords[None].expand(b, 2, hs, ws)
                in_displacement = flow-im_A_coords
                emb_in_displacement = self.disp_emb(40/32 * scale_factor * in_displacement)
                d = torch.cat((x, x_hat, emb_in_displacement), dim=1)
            else:
                if self.no_im_B_fm:
                    x_hat = torch.zeros_like(x)
                d = torch.cat((x, x_hat), dim=1)
            if self.concat_logits:
                d = torch.cat((d, logits), dim=1)
            d = self.block1(d)
            d = self.hidden_blocks(d)
        d = self.out_conv(d.float())
        displacement, certainty = d[:, :-1], d[:, -1:]
        return displacement, certainty

dw = True
hidden_blocks = 8
kernel_size = 5
displacement_emb = "linear"
disable_local_corr_grad = True


class decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_refiner = nn.ModuleDict({            
            "2": ConvRefiner(
                2 * 64+16,
                128+16,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=16,
                amp = True,
                disable_local_corr_grad = disable_local_corr_grad,
                bn_momentum = 0.01,
            ),
            "1": ConvRefiner(
                2 * 9 + 6,
                24,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks = hidden_blocks,
                displacement_emb = displacement_emb,
                displacement_emb_dim = 6,
                amp = True,
                disable_local_corr_grad = disable_local_corr_grad,
                bn_momentum = 0.01,
            )})
        self.proj= nn.ModuleDict({
            "2":nn.Sequential(nn.Conv2d(128, 64, 1, 1), nn.BatchNorm2d(64)),
            "1":nn.Sequential(nn.Conv2d(64, 9, 1, 1), nn.BatchNorm2d(9))})
        self.refine_init = 4
        self.flow_upsample_mode = "bilinear"
    def forward(self,f1, f2, flow,certainty,symmetric):

        corresps = {}
        all_scales = ["2", "1"] 
        sizes = {scale: f1[scale].shape[-2:] for scale in f1}
        h, w = sizes[1]
        coarsest_scale = int(all_scales[0])

        if flow.shape[2]!=sizes[coarsest_scale][0] or flow.shape[2]!=sizes[coarsest_scale][0]:
            flow = F.interpolate(
                    flow,
                    size=sizes[coarsest_scale],
                    align_corners=False,
                    mode="bilinear",
                )
            certainty = F.interpolate(
                    certainty,
                    size=sizes[coarsest_scale],
                    align_corners=False,
                    mode="bilinear",
                )
        
        displacement = 0.0
        for new_scale in all_scales:
            ins = int(new_scale)
            corresps[ins] = {}
            f1_s_, f2_s_ = f1[ins], f2[ins]
            if symmetric:
                f1_s = torch.cat([f1_s_,f2_s_],dim=0)
                f2_s = torch.cat([f2_s_,f1_s_],dim=0)
            else:
                f1_s = f1_s_
                f2_s = f2_s_
            if new_scale in self.proj:
                autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(f1_s.device, str(f1_s)=='cuda', torch.float16)
                with torch.autocast(autocast_device, enabled=autocast_enabled, dtype = autocast_dtype):
                    if not autocast_enabled:
                        f1_s, f2_s = f1_s.to(torch.float32), f2_s.to(torch.float32)
                    f1_s, f2_s = self.proj[new_scale](f1_s), self.proj[new_scale](f2_s)

            if new_scale in self.conv_refiner:
                corresps[ins].update({"flow_pre_delta": flow}) if self.training else None
                delta_flow, delta_certainty = self.conv_refiner[new_scale](
                    f1_s, f2_s, flow, scale_factor = 1, logits = certainty,
                )                    
                corresps[ins].update({"delta_flow": delta_flow,}) if self.training else None
                displacement = ins*torch.stack((delta_flow[:, 0].float() / (self.refine_init * w),
                                                delta_flow[:, 1].float() / (self.refine_init * h),),dim=1,)
                flow = flow + displacement
                certainty = (
                    certainty + delta_certainty
                )  # predict both certainty and displacement
            corresps[ins].update({
                "certainty": certainty,
                "flow": flow,             
            })
            if new_scale != "1":
                flow = F.interpolate(
                    flow,
                    size=sizes[ins // 2],
                    mode=self.flow_upsample_mode,
                )
                certainty = F.interpolate(
                    certainty,
                    size=sizes[ins // 2],
                    mode=self.flow_upsample_mode,
                )        
        return corresps
import sys
from copy import deepcopy
import torchvision.transforms.functional as TF
weight_urls = {
    "roma": {
        "outdoor": "https://github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth",
        "indoor": "https://github.com/Parskatt/storage/releases/download/roma/roma_indoor.pth",
    },
    "tiny_roma_v1": {
        "outdoor": "https://github.com/Parskatt/storage/releases/download/roma/tiny_roma_v1_outdoor.pth",
    },
    "dinov2": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth", #hopefully this doesnt change :D
}

class DenseHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = encoder()
        self.decoder = decoder()    
        pretrained_dict =  torch.hub.load_state_dict_from_url(weight_urls["roma"]["outdoor"],
                                                     map_location="cpu")
        model_dict = self.state_dict()
        pretrained_dict_ = {}
        for key,value in pretrained_dict.items():
            if key in model_dict:
                # print(key)
                pretrained_dict_[key]=value
        model_dict.update(pretrained_dict_) 
        self.load_state_dict(model_dict)


        # torch.save(self.state_dict(),"head.pt")
    def forward(self,im1,im2,flow,certainty,symmetric=False):
        im1 = TF.normalize(im1,mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        im2 = TF.normalize(im2,mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        feats1 = self.encoder(im1)
        feats2 = self.encoder(im2)
        corres = self.decoder(feats1,feats2,flow,certainty,symmetric)
        return corres