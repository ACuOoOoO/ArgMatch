import torch
import torch.nn.functional as F

def local_correlation(
    feature0,
    feature1,
    local_radius,
    padding_mode="zeros",
    flow = None,
    sample_mode = "bilinear",
):
    r = local_radius
    K = (2*r+1)**2
    B, c, h, w = feature0.size()
    _,_,h_y, w_y = feature1.size()
    corr = torch.empty((B,K,h,w), device = feature0.device, dtype=feature0.dtype)

    coords = flow.permute(0,2,3,1) # If using flow, sample around flow target.
    local_window = torch.meshgrid(
                (
                    torch.linspace(-2*local_radius/h_y, 2*local_radius/h_y, 2*r+1, device=feature0.device),
                    torch.linspace(-2*local_radius/w_y, 2*local_radius/w_y, 2*r+1, device=feature0.device),
                ),
                indexing = 'ij'
                )
    local_window = torch.stack((local_window[1], local_window[0]), dim=-1)[
            None
        ].expand(1, 2*r+1, 2*r+1, 2).reshape(1, (2*r+1)**2, 2)
    for _ in range(B):
        with torch.no_grad():
            local_window_coords = (coords[_,:,:,None]+local_window[:,None,None]).reshape(1,h,w*(2*r+1)**2,2)
            window_feature = F.grid_sample(
                feature1[_:_+1], local_window_coords, padding_mode=padding_mode, align_corners=False, mode = sample_mode, #
            )
            window_feature = window_feature.reshape(c,h,w,(2*r+1)**2)
        corr[_] = (feature0[_,...,None]/(c**.5)*window_feature).sum(dim=0).permute(2,0,1)
    return corr
