import torch
import os

from traitlets import default
from pathlib import Path
import json
from argmatch.benchmarks import ScanNetBenchmark, MegaDepthPoseEstimationBenchmark,MegadepthDenseScaleBenchmark






def test_mega_8_scenes(model, name):
    mega_8_scenes_benchmark = MegaDepthPoseEstimationBenchmark(
        "data/megadepth",
        scene_names=[
            'mega_8_scenes_0019_0.1_0.3.npz',
            'mega_8_scenes_0025_0.1_0.3.npz',
            'mega_8_scenes_0021_0.1_0.3.npz',
            'mega_8_scenes_0008_0.1_0.3.npz',
            'mega_8_scenes_0032_0.1_0.3.npz',
            'mega_8_scenes_1589_0.1_0.3.npz',
            'mega_8_scenes_0063_0.1_0.3.npz',
            'mega_8_scenes_0024_0.1_0.3.npz',
            'mega_8_scenes_0019_0.3_0.5.npz',
            'mega_8_scenes_0025_0.3_0.5.npz',
            'mega_8_scenes_0021_0.3_0.5.npz',
            'mega_8_scenes_0008_0.3_0.5.npz',
            'mega_8_scenes_0032_0.3_0.5.npz',
            'mega_8_scenes_1589_0.3_0.5.npz',
            'mega_8_scenes_0063_0.3_0.5.npz',
            'mega_8_scenes_0024_0.3_0.5.npz'
        ]
    )
    mega_8_scenes_results = mega_8_scenes_benchmark.benchmark(model, model_name=name)
    print(mega_8_scenes_results)
    json.dump(mega_8_scenes_results, open(f"results/mega_8_scenes_{name}.json", "w"))


def test_mega1500(model, name):
    mega1500_benchmark = MegaDepthPoseEstimationBenchmark("data/megadepth")
    mega1500_results = mega1500_benchmark.benchmark(model, model_name=name)
    json.dump(mega1500_results, open(f"results/mega1500_{name}.json", "w"))


def test_megdepth_dense_scale(model, name):
    model.symmetric = False
    model.h = 672
    model.w = 672
    benchmark = MegadepthDenseScaleBenchmark('data/megadepth')
    pcks, precs = benchmark.benchmark(model)



def test_megdepth_dense_scale_8(model, name):
    benchmark = MegadepthDenseScaleBenchmark(
        'data/megadepth',
        scene_names=[
            'mega_8_scenes_0019_0.1_0.3.npz',
            'mega_8_scenes_0025_0.1_0.3.npz',
            'mega_8_scenes_0021_0.1_0.3.npz',
            'mega_8_scenes_0008_0.1_0.3.npz',
            'mega_8_scenes_0032_0.1_0.3.npz',
            'mega_8_scenes_1589_0.1_0.3.npz',
            'mega_8_scenes_0063_0.1_0.3.npz',
            'mega_8_scenes_0024_0.1_0.3.npz',
            'mega_8_scenes_0019_0.3_0.5.npz',
            'mega_8_scenes_0025_0.3_0.5.npz',
            'mega_8_scenes_0021_0.3_0.5.npz',
            'mega_8_scenes_0008_0.3_0.5.npz',
            'mega_8_scenes_0032_0.3_0.5.npz',
            'mega_8_scenes_1589_0.3_0.5.npz',
            'mega_8_scenes_0063_0.3_0.5.npz',
            'mega_8_scenes_0024_0.3_0.5.npz'
        ]
    )
    pcks, precs = benchmark.benchmark(model)
    torch.save({"pcks": pcks, "precs": precs}, "{}.pt".format(name))



def test_scannet(model, name):
    model.h = 480
    model.w = 640
    model.sample_thres = 0.1
    model.symetric = True
    scannet_benchmark = ScanNetBenchmark("data/scannet")
    scannet_results = scannet_benchmark.benchmark(model)
    json.dump(scannet_results, open(f"results/scannet_{name}.json", "w"))


if __name__ == "__main__":
    import os
    import argparse
    from pathlib import Path
    import torch

    from argmatch.models.model_zoo import ArgMatch, ArgMatch_plus
    # model = ArgMatch()
    # Here we assume these test functions are in the same namespace
    # (same file or already imported into the global scope).
    # If they are in other modules, please import them as needed.
    # from argmatch.models.high_reso_head import *

    # -------------------------
    # Parse command line arguments
    # -------------------------
    parser = argparse.ArgumentParser(
        description="Run ArgMatch tests with selectable GPU and test entry."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="workspace/checkpoints/ckpt_clean.pth",
        help="Path to model weights (.pth)."
    )
    parser.add_argument(
        "--tests",
        type=str,
        nargs="+",
        default=["test_mega1500"],
        help=(
            "One or more test function names to run, e.g.: "
            "test_scannet test_mega1500 test_megdepth_dense_scale test_hpatches_dense "
        ),
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU id to use (e.g., 0). Use -1 for CPU."
    )
    # Optional model-parameter overrides
    parser.add_argument("--symmetric", type=int, default=1, help="1/0 -> True/False")
    parser.add_argument("--upsample", type=int, default=0, help="1/0 -> True/False")
    parser.add_argument("--sample_thresh", type=float, default=0.03)
    parser.add_argument("--h", type=int, default=608)
    parser.add_argument("--w", type=int, default=800)

    # Allow passing extra identifier such as experiment name
    parser.add_argument("--exp-name", type=str, default="github", help="Experiment name.")

    args = parser.parse_args()

    # -------------------------
    # Device selection
    # -------------------------
    if args.gpu is None or args.gpu < 0:
        device = torch.device("cpu")
        cuda_msg = "CPU"
    else:
        if torch.cuda.is_available():
            # Optionally set the current process default GPU
            torch.cuda.set_device(args.gpu)
            device = torch.device(f"cuda:{args.gpu}")
            cuda_msg = f"CUDA:{args.gpu}"
        else:
            print("[WARN] CUDA is not available, falling back to CPU.")
            device = torch.device("cpu")
            cuda_msg = "CPU"

    # -------------------------
    # Load weights
    # -------------------------
    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"weights not found: {args.weights}")

    ckpt = torch.load(args.weights, map_location="cpu")

    model = ArgMatch()
    # Support two checkpoint formats: with a 'model' key or a raw state_dict
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[INFO] Missing keys: {len(missing)} (showing first 10) -> {missing[:10]}")
    if unexpected:
        print(f"[INFO] Unexpected keys: {len(unexpected)} (showing first 10) -> {unexpected[:10]}")

    model = model.to(device)
    model.eval()

    # Override model attributes from CLI
    model.symmetric = bool(args.symmetric)
    model.upsample = bool(args.upsample)
    model.sample_thresh = args.sample_thresh
    model.h = args.h
    model.w = args.w

    # Print experiment / model info
    experiment_name = args.exp_name or Path(__file__).stem
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Experiment: {experiment_name}")
    print(f"[INFO] Device: {cuda_msg}")
    print(f"[INFO] Params: {total_params:,}")
    print(
        f"[INFO] symmetric={model.symmetric} upsample={model.upsample} "
        f"sample_thresh={model.sample_thresh} h={model.h} w={model.w}"
    )

    # -------------------------
    # Run selected test functions
    # -------------------------
    # Look up test functions by name in the global namespace;
    # this can be replaced with an explicit mapping if desired.
    available = {}
    available.update(globals())

    for name in args.tests:
        fn = available.get(name, None)
        if fn is None or not callable(fn):
            raise ValueError(
                f"Test function '{name}' does not exist or is not callable. "
                "Please make sure it is imported into the current namespace or check the spelling."
            )
        print(f"[RUN] {name} ...")
        # Most existing signatures look like test_xxx(model, tag),
        # so we try (model, tag) first and then fall back to (model,)
        try:
            fn(model, experiment_name)
        except TypeError:
            fn(model)
        print(f"[DONE] {name}")
