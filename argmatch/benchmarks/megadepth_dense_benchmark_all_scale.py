import torch
import numpy as np
import tqdm
from argmatch.datasets.megadepth import MegadepthScene
from argmatch.utils.utils import warp_kpts, create_meshgrid
from torch.utils.data import ConcatDataset
import argmatch
import os
import torchvision.transforms.functional as TF
import torch.nn.functional as F
class MegadepthDenseScaleBenchmark:
    def __init__(
        self, data_root="data/megadepth",   scene_names = None,
    ) -> None:
        if scene_names is None:
            self.scene_names = [
                "0015_0.1_0.3.npz",
                "0015_0.3_0.5.npz",
                "0022_0.1_0.3.npz",
                "0022_0.3_0.5.npz",
                "0022_0.5_0.7.npz",
            ]
        else:
            self.scene_names = scene_names
        scenes = []
        for scene_name in self.scene_names:
            scene_info = np.load(os.path.join(data_root, scene_name), allow_pickle=True)
            scenes.append(
                MegadepthScene(
                    data_root,
                    scene_info,
                    min_overlap=0,
                    scene_name=scene_name,
                    normalize=False,
                    ht=672,
                    wt=672,
                )
            )
        self.dataset = ConcatDataset(scenes)  # fixed resolution of 384,512

    def dense_geometric_dist(self, depth1, depth2, T_1to2, K1, K2, flow):
        b, h1, w1, d = flow.shape
        with torch.no_grad():
            x1 = create_meshgrid(h1, w1, flow.device).reshape(1, -1, 2).repeat(b, 1, 1)
            x2, prob = warp_kpts(
                x1.double(),
                depth1.double(),
                depth2.double(),
                T_1to2.double(),
                K1.double(),
                K2.double(),
            )
            x2 = torch.stack(
                (w1 * (x2[..., 0] + 1) / 2, h1 * (x2[..., 1] + 1) / 2), dim=-1
            )
            prob = prob.float().reshape(b, h1, w1)
        x2_hat = flow
        x2_hat = torch.stack(
            (w1 * (x2_hat[..., 0] + 1) / 2, h1 * (x2_hat[..., 1] + 1) / 2), dim=-1
        )
        gd = (x2_hat - x2.reshape(b, h1, w1, 2)).norm(dim=-1)
        return gd, prob

    def sparse_geometric_dist(self, depth1, depth2, T_1to2, K1, K2, matches, b, h1, w1):
        x1 = matches[:, :2][None]
        with torch.no_grad():
            mask, x2, prob = warp_kpts(
                x1.double(),
                depth1.double(),
                depth2.double(),
                T_1to2.double(),
                K1.double(),
                K2.double(),
            )
            x2 = torch.stack(
                (w1 * (x2[..., 0] + 1) / 2, h1 * (x2[..., 1] + 1) / 2), dim=-1
            )
            prob = mask.float().reshape(b, 1, h1, w1)
        prob_sampled = torch.nn.functional.grid_sample(prob, x1, mode="bilinear").view(
            -1
        )
        x2 = x2.reshape(-1, 2)
        x2_hat = matches[:, 2:]
        x2_hat = torch.stack(
            (w1 * (x2_hat[..., 0] + 1) / 2, h1 * (x2_hat[..., 1] + 1) / 2), dim=-1
        )
        gd = (x2_hat - x2.reshape(b, h1, w1, 2)).norm(dim=-1)
        gd = gd[prob_sampled > 0.99]
        return gd

    def benchmark(self, model, batch_size=4):
        model.train(False)
        model.eval()
        with torch.no_grad():
            B = batch_size
            dataloader = torch.utils.data.DataLoader(
                self.dataset, batch_size=B, num_workers=batch_size
            )
            pcks = torch.zeros(len(self.dataset), 5, 4).cuda()
            precs = torch.zeros(len(self.dataset), 5).cuda()
            for idx, data in tqdm.tqdm(enumerate(dataloader)):
                im_A, im_B, depth1, depth2, T_1to2, K1, K2 = (
                    data["im_A"].cuda(),
                    data["im_B"].cuda(),
                    data["im_A_depth"].cuda(),
                    data["im_B_depth"].cuda(),
                    data["T_1to2"].cuda(),
                    data["K1"].cuda(),
                    data["K2"].cuda(),
                )
                batch = {"im_A": im_A, "im_B": im_B}
                desne_matches = model(batch)
                # for m in range(3):
                for j, scale in enumerate([16, 8, 4, 2, 1]):
                    if not scale in desne_matches.keys():
                        continue
                    flow = desne_matches[scale]["flow"] 
                    gd, prob = self.dense_geometric_dist(
                        depth1,
                        depth2,
                        T_1to2,
                        K1,
                        K2,
                        flow.permute(0, 2, 3, 1),
                    )

                    for k in range(gd.shape[0]):
                        pcks[idx * B + k, j, 0] = (
                            (gd[k][prob[k] > 0.99] < 0.5).float().mean()
                        )
                        pcks[idx * B + k, j, 1] = (
                            (gd[k][prob[k] > 0.99] < 1).float().mean()
                        )
                        pcks[idx * B + k, j, 2] = (
                            (gd[k][prob[k] > 0.99] < 3).float().mean()
                        )
                        pcks[idx * B + k, j, 3] = (
                            (gd[k][prob[k] > 0.99] < 5).float().mean()
                        )
                        
                        if scale == 2:
                            if isinstance(desne_matches[scale]["flow"],list):
                                flow = desne_matches[scale]["flow"][-1]
                            else:
                                flow = desne_matches[scale]["flow"] 
                            b, c, h, w = flow.shape
                            device = flow.device
                            grid = create_meshgrid(h, w, device=device)
                            matches = torch.cat(
                                [
                                    grid[0],
                                    flow[k].permute(1, 2, 0),
                                ],
                                dim=-1,
                            )
                            sparse_matches, certaitny = model.sample(
                                matches[None],
                                desne_matches[2]["certainty"][k],
                                5000,
                            )
                            x1 = sparse_matches[:, :2]
                            gd_sampled = torch.nn.functional.grid_sample(
                                gd[k][None, None].float(), x1[None, :, None]
                            )
                            prob_sampled = torch.nn.functional.grid_sample(
                                prob[k][None, None], x1[None, :, None]
                            )
                            t = gd_sampled[prob_sampled > 0.99]
                            precs[idx * B + k, 0] = sparse_matches.shape[0]
                            precs[idx * B + k, 1] = torch.count_nonzero(prob_sampled > 0.99)
                            precs[idx * B + k, 2] = torch.count_nonzero(t < 0.5)
                            precs[idx * B + k, 3] = torch.count_nonzero(t < 1)
                            precs[idx * B + k, 4] = torch.count_nonzero(t < 2)
        print(pcks.mean(dim=0))
        return pcks,precs
