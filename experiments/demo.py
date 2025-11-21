import os
import argparse

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as tvf

from argmatch.models.model_zoo import ArgMatch, ArgMatch_plus


def parse_args():
    parser = argparse.ArgumentParser(
        description="ArgMatch demo: dense matching and visualization."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="workspace/checkpoints/ckpt_clean.pth",
        help="Path to the ArgMatch checkpoint (.pth).",
    )
    parser.add_argument(
        "--image_A",
        type=str,
        default="_assets/im_A.jpg",
        help="Path to the first image.",
    )
    parser.add_argument(
        "--image_B",
        type=str,
        default="_assets/im_B.jpg",
        help="Path to the second image.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="_assets/outputs",
        help="Directory to save visualization results.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load model and checkpoint
    model = ArgMatch().to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # 2. Load and preprocess input images
    im_A = Image.open(args.image_A).convert("RGB")
    im_B = Image.open(args.image_B).convert("RGB")

    # Network input resolution
    net_h, net_w = 608, 800
    vis_h, vis_w = 600, 800  # visualization resolution

    im_A_resized = tvf.resize(im_A, [net_h, net_w])
    im_B_resized = tvf.resize(im_B, [net_h, net_w])

    im_A_tensor = tvf.to_tensor(im_A_resized)[None].to(device)  # (1,3,H,W)
    im_B_tensor = tvf.to_tensor(im_B_resized)[None].to(device)  # (1,3,H,W)

    batch = {"im_A": im_A_tensor, "im_B": im_B_tensor}

    # 3. Forward pass
    with torch.no_grad():
        dense_matches = model(batch)

    # 4. Resize for visualization
    im_A_vis = tvf.resize(im_A_tensor, [vis_h, vis_w])
    im_B_vis = tvf.resize(im_B_tensor, [vis_h, vis_w])

    flow = tvf.resize(dense_matches[2]["flow"], [vis_h, vis_w])           # (1,2,H,W)
    certainty = tvf.resize(dense_matches[2]["certainty"], [vis_h, vis_w]) # (1,1,H,W)
    certainty = certainty.sigmoid()

    # 5. Warp B to A using the flow
    # grid_sample expects grid in [-1, 1] with shape (N,H,W,2)
    grid = flow.permute(0, 2, 3, 1)  # (1,H,W,2)
    im_B_warp = F.grid_sample(
        im_B_vis, grid, mode="bilinear", align_corners=False
    )  # (1,3,H,W)

    # Apply certainty mask
    im_B_warp = (im_B_warp * (certainty > 0.2))[0]  # (3,H,W)
    blk_mask = (im_B_warp > 0.01).count_nonzero(dim=0) < 1  # (H,W)
    im_B_warp[:, blk_mask] = 1.0  # set invalid pixels to white

    # Concatenate A, B, and warped B side by side
    im_cat = torch.cat([im_A_vis[0], im_B_vis[0], im_B_warp], dim=-1)  # (3,H,3W)

    # 6. Visualize flow as RGB
    # Duplicate first channel to make 3 channels: (1,3,H,W)
    flow_viz = torch.cat([flow, flow[:, :1]], dim=1)
    flow_viz = (flow_viz + 1).clamp(max=2.0) / 2.0  # map [-1,1] -> [0,1]

    # 7. Save outputs
    flow_path = os.path.join(args.output_dir, "flow.png")
    cert_path = os.path.join(args.output_dir, "certainty.png")
    cat_path = os.path.join(args.output_dir, "concat.png")

    tvf.to_pil_image(flow_viz[0].cpu()).save(flow_path)
    tvf.to_pil_image(certainty[0].cpu()).save(cert_path)
    tvf.to_pil_image(im_cat.cpu()).save(cat_path)

    print(f"Saved flow visualization to: {flow_path}")
    print(f"Saved certainty map to:      {cert_path}")
    print(f"Saved concatenated image to: {cat_path}")


if __name__ == "__main__":
    main()
