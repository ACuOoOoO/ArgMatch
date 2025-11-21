import torch
import kornia.geometry as KG
import kornia.geometry.epipolar as KGE
from kornia.geometry.linalg import transform_points
from typing import Tuple

def compute_rotation_error(T0, T1, reduce=True,thres=0.5,valid_mask=None):
    # use diagonal and sum to compute trace of a batch of matrices
    cos_a = ((T0[..., :3, :3].transpose(-1, -2) @ T1[..., :3, :3]).diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) \
        - 1.) / 2.
    cos_a = torch.clamp(cos_a, -1.+1e-3, 1.-1e-3) # avoid nan
    abs_acos_a = torch.abs(torch.arccos(cos_a))
    if reduce:
        return abs_acos_a.clamp(min=0.02,max=thres).mean()
    else:
        return abs_acos_a

def compute_translation_error_as_angle(T0, T1, reduce=True,thres=0.5,valid_mask=None):
    n = (torch.linalg.norm(T0[..., :3, 3], dim=-1) * torch.linalg.norm(T1[..., :3, 3], dim=-1))
    # valid_n = n > 1e-6
    T0_dot_T1 = (T0[..., :3, 3] * T1[..., :3, 3]).sum(-1)
    err = torch.abs(torch.arccos((T0_dot_T1 / n).clamp(-1.+1e-3, 1.-1e-3)))
    if reduce:
        return err.clamp(min=0.02,max=thres).mean()
    else:
        return err

def normalize_transformation(M: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r"""Normalize a given transformation matrix.

    The function trakes the transformation matrix and normalize so that the value in
    the last row and column is one.

    Args:
        M: The transformation to be normalized of any shape with a minimum size of 2x2.
        eps: small value to avoid unstabilities during the backpropagation.

    Returns:
        the normalized transformation matrix with same shape as the input.

    """
    if len(M.shape) < 2:
        raise AssertionError(M.shape)
    norm_val: torch.Tensor = M[..., -1:, -1:]
    return torch.where(norm_val.abs() > eps, M / (norm_val + eps), M)


def normalize_points(points: torch.Tensor, score: torch.Tensor=None, eps: float = 1e-8) :
    b,n,_ = points.shape
    if score is not None:
        score = score.reshape(b,n)
        score_sum = score.sum(dim=1) 
        weighted_points = points*score.unsqueeze(-1)
        x_mean = torch.sum(weighted_points, dim=1, keepdim=True)/score_sum[:,None,None]  # Bx1x2

        scale = (points - x_mean).norm(dim=-1, p=2)  # Bxn
        scale = (score*scale).sum(dim=-1)/score_sum
        scale = torch.sqrt(torch.tensor(2.0)) / (scale + eps)  # B

        ones, zeros = torch.torch.ones_like(scale), torch.zeros_like(scale)
    else:
        x_mean = torch.mean(points, dim=1, keepdim=True)  # Bx1x2

        scale = (points - x_mean).norm(dim=-1, p=2).mean(dim=-1)  # B
        scale = torch.sqrt(torch.tensor(2.0)) / (scale + eps)  # B

        ones, zeros = torch.torch.ones_like(scale), torch.zeros_like(scale)
    transform = torch.torch.stack(
        [scale, zeros, -scale * x_mean[..., 0, 0], zeros, scale, -scale * x_mean[..., 0, 1], zeros, zeros, ones], dim=-1
    )  # Bx9

    transform = transform.view(-1, 3, 3)  # Bx3x3
    points_norm = transform_points(transform, points)  # BxNx2

    return (points_norm, transform)

def find_fundamental(points1: torch.Tensor, points2: torch.Tensor, weights: torch.Tensor,weight_norm=None) -> torch.Tensor:

    points1_norm, transform1 = normalize_points(points1,weight_norm)
    points2_norm, transform2 = normalize_points(points2,weight_norm)

    x1, y1 = torch.chunk(points1_norm, dim=-1, chunks=2)  # Bx1xN
    x2, y2 = torch.chunk(points2_norm, dim=-1, chunks=2)  # Bx1xN

    ones = torch.torch.ones_like(x1)
    X = torch.cat([x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, ones], dim=-1)  # BxNx9
    weights = weights/weights.sum()*100

    X_w = (X*weights.unsqueeze(-1)).transpose(-1,-2)*10
    XX = torch.bmm(X_w,X_w.transpose(-1,-2))

    _, _, V = torch.svd(XX)
    V2 = V[...,-1]
    V2= V2/V2[:,-1:]
    F_mat = V[..., -1].view(-1, 3, 3)
    U, S, V = torch.svd(F_mat)
    rank_mask = torch.tensor([1.0, 1.0, 0.0],
                           device=F_mat.device,
                           dtype=F_mat.dtype)
    F_projected = U @ (torch.diag_embed(S * rank_mask) @ V.transpose(-2, -1))
    F_est = transform2.transpose(-2, -1) @ (F_projected @ transform1)
    F_est = F_est/F_est[:,-1,-1][:,None,None]
    return F_est


def intr_normalize(kpts, intr):
    n_kpts = torch.zeros_like(kpts)
    fx, fy, cx, cy = intr[..., 0, 0], intr[..., 1, 1], intr[..., 0, 2], intr[..., 1, 2]
    n_kpts[..., 0] = (kpts[..., 0] - cx[:,None]) / fx[:,None]
    n_kpts[..., 1] = (kpts[..., 1] - cy[:,None]) / fy[:,None]
    return n_kpts


def estimate_pose(x1, x2, intr0, intr1, weights, T_021=None):
    dev = intr0.device
    bs = intr0.shape[0]
    kpts0_norm = intr_normalize(x1, intr0)
    kpts1_norm = intr_normalize(x2, intr1)
    Es = find_fundamental(kpts0_norm,kpts1_norm,weights=weights)
    Rs, ts = KGE.motion_from_essential(Es)
    min_err = torch.full((bs,), 1e6, device=dev)
    min_err_pred_T021 = torch.torch.eye(4, device=dev).unsqueeze(0).repeat(bs, 1, 1)
    for R, t in zip(Rs.permute(1, 0, 2, 3), ts.permute(1, 0, 2, 3)):
        pred_T021 = torch.torch.eye(4, device=dev).unsqueeze(0).repeat(bs, 1, 1)
        pred_T021[:, :3, :3] = R
        pred_T021[:, :3, 3] = t.squeeze(-1)
        curr_err = compute_rotation_error(pred_T021, T_021, reduce=False) + compute_translation_error_as_angle(pred_T021, T_021, reduce=False)
        update_mask = curr_err < min_err
        min_err[update_mask] = curr_err[update_mask]
        min_err_pred_T021[update_mask] = pred_T021[update_mask]
    return min_err_pred_T021



            