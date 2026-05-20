"""Post-hoc cleanup of a trained gsplat checkpoint.

Three filters (any of them flags a Gaussian for removal):
  1. visibility: not projected with radii>0 from ANY training/val camera
  2. opacity:    sigmoid(opacities) < --opacity-threshold
  3. scale:      max(exp(scales)) > --scale-max-mult * scene_scale

Usage example mirroring the lidar training command:

    python clean_ckpt.py \
        --ckpt results/atlas_v3_reg/ckpts/ckpt_29999_rank0.pt \
        --data-dir data/atlas \
        --data-factor 2 \
        --no-normalize-world-space \
        --opacity-threshold 0.05 \
        --scale-max-mult 0.1 \
        --output results/atlas_v3_reg/clean
"""

import argparse
import math
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from datasets.colmap import Parser

from gsplat import export_splats
from gsplat.rendering import rasterization


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="path to ckpt_*.pt (must contain 'splats' state_dict)")
    p.add_argument("--data-dir", required=True, help="dataset root (with sparse/0/)")
    p.add_argument("--data-factor", type=int, default=2, help="image downsample factor used in training (default 2)")
    p.add_argument("--test-every", type=int, default=8, help="must match training (default 8)")
    p.add_argument("--normalize-world-space", dest="normalize_world_space", action="store_true")
    p.add_argument("--no-normalize-world-space", dest="normalize_world_space", action="store_false")
    p.set_defaults(normalize_world_space=False)
    p.add_argument("--cameras", choices=["train", "val", "all"], default="all",
                   help="which cameras to use for visibility test (default all)")
    p.add_argument("--opacity-threshold", type=float, default=0.05,
                   help="drop Gaussians with sigmoid(opacity) below this (default 0.05)")
    p.add_argument("--scale-max-mult", type=float, default=0.1,
                   help="drop Gaussians where max(exp(scales)) > this * scene_scale (default 0.1, matches DefaultStrategy.prune_scale3d)")
    p.add_argument("--no-visibility", action="store_true", help="disable visibility filter")
    p.add_argument("--no-opacity", action="store_true", help="disable opacity filter")
    p.add_argument("--no-scale", action="store_true", help="disable scale filter")
    p.add_argument("--global-scale", type=float, default=1.0, help="must match training (default 1.0)")
    p.add_argument("--output", required=True, help="output directory")
    p.add_argument("--export-splat", action="store_true", help="also export .splat (antimatter15)")
    p.add_argument("--save-clean-ckpt", action="store_true", help="also save cleaned .pt checkpoint")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def infer_sh_degree(shN: Tensor) -> int:
    """shN is (N, K-1, 3). Total SH coeffs K = sh0 (1) + shN. sh_degree d satisfies K = (d+1)^2."""
    k_total = shN.shape[1] + 1
    d = int(math.isqrt(k_total)) - 1
    assert (d + 1) ** 2 == k_total, f"shN shape {tuple(shN.shape)} not a valid SH set"
    return d


def load_splats(ckpt_path: str, device: str) -> Dict[str, Tensor]:
    blob = torch.load(ckpt_path, map_location=device)
    if "splats" not in blob:
        raise KeyError(f"'splats' key not found in {ckpt_path}; got {list(blob.keys())}")
    s = blob["splats"]
    needed = ["means", "scales", "quats", "opacities", "sh0", "shN"]
    for k in needed:
        if k not in s:
            raise KeyError(f"splat field '{k}' missing in checkpoint")
    return {k: s[k].to(device) for k in needed}


def collect_visibility(
    splats: Dict[str, Tensor],
    camtoworlds: np.ndarray,
    Ks: List[np.ndarray],
    sizes: List[Tuple[int, int]],
    sh_degree: int,
    device: str,
) -> Tensor:
    """Returns bool tensor [N] — True if Gaussian was projected with radii>0 from any camera."""
    means = splats["means"]
    quats = splats["quats"]
    scales = torch.exp(splats["scales"])
    opacities = torch.sigmoid(splats["opacities"])
    colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)  # [N, K, 3]
    N = means.shape[0]
    visible = torch.zeros(N, dtype=torch.bool, device=device)

    for i in tqdm(range(len(camtoworlds)), desc="visibility"):
        c2w = torch.from_numpy(camtoworlds[i]).float().to(device)
        viewmat = torch.linalg.inv(c2w).unsqueeze(0)  # [1, 4, 4]
        K = torch.from_numpy(Ks[i]).float().to(device).unsqueeze(0)  # [1, 3, 3]
        W, H = sizes[i]
        _, _, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat,
            Ks=K,
            width=W,
            height=H,
            sh_degree=sh_degree,
            packed=True,
            render_mode="RGB",
        )
        gids = info["gaussian_ids"]  # [nnz], indices in [0, N)
        if gids is not None and gids.numel() > 0:
            visible[gids.long()] = True

    return visible


@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    device = args.device

    # 1. Load checkpoint
    splats = load_splats(args.ckpt, device)
    N = splats["means"].shape[0]
    sh_degree = infer_sh_degree(splats["shN"])
    print(f"Loaded {N} Gaussians from {args.ckpt} (sh_degree={sh_degree})")

    # 2. Load camera intrinsics/extrinsics via the same Parser the trainer used
    parser = Parser(
        data_dir=args.data_dir,
        factor=args.data_factor,
        normalize=args.normalize_world_space,
        test_every=args.test_every,
        load_exposure=False,
    )
    scene_scale = parser.scene_scale * 1.1 * args.global_scale
    print(f"Scene scale: {scene_scale:.4f}")

    indices_all = np.arange(len(parser.image_names))
    if args.cameras == "train":
        cam_indices = indices_all[indices_all % parser.test_every != 0]
    elif args.cameras == "val":
        cam_indices = indices_all[indices_all % parser.test_every == 0]
    else:
        cam_indices = indices_all
    print(f"Using {len(cam_indices)} cameras ({args.cameras} split)")

    camtoworlds = parser.camtoworlds[cam_indices]  # [C, 4, 4]
    Ks = [parser.Ks_dict[parser.camera_ids[i]] for i in cam_indices]
    sizes = [parser.imsize_dict[parser.camera_ids[i]] for i in cam_indices]  # (W, H)

    # 3. Build per-filter masks (True = drop)
    drop_visibility = torch.zeros(N, dtype=torch.bool, device=device)
    drop_opacity = torch.zeros(N, dtype=torch.bool, device=device)
    drop_scale = torch.zeros(N, dtype=torch.bool, device=device)

    if not args.no_visibility:
        visible = collect_visibility(splats, camtoworlds, Ks, sizes, sh_degree, device)
        drop_visibility = ~visible
        print(f"  visibility filter: {drop_visibility.sum().item()} not seen by any camera")

    if not args.no_opacity:
        opa = torch.sigmoid(splats["opacities"]).flatten()
        drop_opacity = opa < args.opacity_threshold
        print(f"  opacity filter:    {drop_opacity.sum().item()} below sigmoid={args.opacity_threshold}")

    if not args.no_scale:
        scl = torch.exp(splats["scales"]).max(dim=-1).values
        drop_scale = scl > args.scale_max_mult * scene_scale
        print(f"  scale filter:      {drop_scale.sum().item()} above {args.scale_max_mult}*scene_scale = {args.scale_max_mult*scene_scale:.4f}")

    drop = drop_visibility | drop_opacity | drop_scale
    keep = ~drop
    n_keep = keep.sum().item()
    print(f"Total: keep {n_keep} / {N} ({100*n_keep/N:.1f}%); drop {N - n_keep}")

    # 4. Apply
    cleaned = {k: v[keep].contiguous() for k, v in splats.items()}

    # 5. Export PLY
    ply_path = os.path.join(args.output, "clean.ply")
    export_splats(
        means=cleaned["means"],
        scales=cleaned["scales"],
        quats=cleaned["quats"],
        opacities=cleaned["opacities"].flatten(),
        sh0=cleaned["sh0"],
        shN=cleaned["shN"],
        format="ply",
        save_to=ply_path,
    )
    print(f"Wrote {ply_path}")

    if args.export_splat:
        splat_path = os.path.join(args.output, "clean.splat")
        export_splats(
            means=cleaned["means"],
            scales=cleaned["scales"],
            quats=cleaned["quats"],
            opacities=cleaned["opacities"].flatten(),
            sh0=cleaned["sh0"],
            shN=cleaned["shN"],
            format="splat",
            save_to=splat_path,
        )
        print(f"Wrote {splat_path}")

    if args.save_clean_ckpt:
        ckpt_path = os.path.join(args.output, "ckpt_clean.pt")
        torch.save({"step": -1, "splats": cleaned}, ckpt_path)
        print(f"Wrote {ckpt_path}  (load with simple_viewer.py --ckpt ...)")


if __name__ == "__main__":
    main()
