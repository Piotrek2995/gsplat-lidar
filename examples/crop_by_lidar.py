"""Crop a gsplat PLY to a reference point cloud (e.g. LiDAR).

Two modes:
  - aabb     : keep splats inside the axis-aligned bbox of the reference (+ padding)
  - distance : keep splats whose mean is within --max_distance of the
               nearest reference point (uses scipy cKDTree)

Usage:
    # AABB
    python crop_by_lidar.py --mode aabb \
        --reference atlas_wyciete_local.ply \
        --splats   results/atlas_v3_reg/clean/clean.ply \
        --output   results/atlas_v3_reg/clean/clean_cropped.ply \
        --pad_pct  0.05

    # Distance-based (exact shape)
    python crop_by_lidar.py --mode distance \
        --reference atlas_wyciete_local.ply \
        --splats   results/atlas_v3_reg/clean/clean.ply \
        --output   results/atlas_v3_reg/clean/clean_shape.ply \
        --max_distance 0.10
"""

import argparse
import os
import struct
import numpy as np
import torch

from gsplat import export_splats


# ---------- generic PLY parser (handles mixed dtypes per property) ----------

PROP_DTYPES = {
    "float":    "<f4",
    "float32":  "<f4",
    "double":   "<f8",
    "float64":  "<f8",
    "uchar":    "u1",
    "uint8":    "u1",
    "char":     "i1",
    "int8":     "i1",
    "ushort":   "<u2",
    "uint16":   "<u2",
    "short":    "<i2",
    "int16":    "<i2",
    "uint":     "<u4",
    "uint32":   "<u4",
    "int":      "<i4",
    "int32":    "<i4",
}


def parse_ply(path):
    """Parses a binary_little_endian PLY with a single vertex element.

    Returns numpy structured array with one record per vertex.
    """
    with open(path, "rb") as f:
        header_bytes = b""
        while b"end_header\n" not in header_bytes:
            chunk = f.read(1)
            if not chunk:
                raise RuntimeError("Unexpected EOF in PLY header")
            header_bytes += chunk
        header = header_bytes.decode("ascii", errors="replace")
        lines = [ln.strip() for ln in header.splitlines()]
        if not any("format binary_little_endian 1.0" in ln for ln in lines):
            raise RuntimeError(f"Unsupported PLY format:\n{header}")
        n = None
        props = []  # (name, dtype_str)
        in_vertex = False
        for ln in lines:
            if ln.startswith("element vertex"):
                in_vertex = True
                n = int(ln.split()[-1])
            elif ln.startswith("element"):
                in_vertex = False
            elif ln.startswith("property") and in_vertex:
                parts = ln.split()
                if parts[1] == "list":
                    raise RuntimeError(f"List properties not supported in this parser: {ln}")
                tname = parts[1]
                pname = parts[2]
                if tname not in PROP_DTYPES:
                    raise RuntimeError(f"Unknown property type '{tname}'")
                props.append((pname, PROP_DTYPES[tname]))
        if n is None:
            raise RuntimeError("Missing 'element vertex'")
        dt = np.dtype({"names": [p[0] for p in props], "formats": [p[1] for p in props]})
        raw = f.read(n * dt.itemsize)
    return np.frombuffer(raw, dtype=dt, count=n)


# ---------- gsplat PLY → splat-dict (raw, pre-activation) ----------

def load_gsplat_ply(path):
    rec = parse_ply(path)
    names = rec.dtype.names

    def col(*ns):
        return np.stack([rec[n] for n in ns], axis=-1)

    for req in ("x", "y", "z", "f_dc_0", "opacity", "scale_0", "rot_0"):
        if req not in names:
            raise RuntimeError(f"'{req}' missing in {path} — not a gsplat PLY (props={names})")

    means    = col("x", "y", "z").astype(np.float32)
    sh0      = col("f_dc_0", "f_dc_1", "f_dc_2").astype(np.float32)[:, None, :]
    opacity  = rec["opacity"].astype(np.float32)
    scale    = col("scale_0", "scale_1", "scale_2").astype(np.float32)
    rot      = col("rot_0", "rot_1", "rot_2", "rot_3").astype(np.float32)

    # f_rest_*: channel-major flat; reshape (N, 3, K-1) then transpose to (N, K-1, 3)
    rest_names = sorted([n for n in names if n.startswith("f_rest_")],
                        key=lambda x: int(x.split("_")[-1]))
    if rest_names:
        rest_flat = np.stack([rec[n] for n in rest_names], axis=-1).astype(np.float32)
        n_pts = rest_flat.shape[0]
        k_minus_1 = len(rest_names) // 3
        assert k_minus_1 * 3 == len(rest_names), f"f_rest count {len(rest_names)} not divisible by 3"
        shN = rest_flat.reshape(n_pts, 3, k_minus_1).transpose(0, 2, 1)
    else:
        shN = np.zeros((means.shape[0], 0, 3), dtype=np.float32)

    return {
        "means":     torch.from_numpy(means),
        "sh0":       torch.from_numpy(sh0),
        "shN":       torch.from_numpy(shN),
        "opacities": torch.from_numpy(opacity),
        "scales":    torch.from_numpy(scale),
        "quats":     torch.from_numpy(rot),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["aabb", "distance"], default="aabb",
                    help="aabb = bbox crop; distance = keep splats near reference cloud")
    ap.add_argument("--reference", required=True, help="reference PLY")
    ap.add_argument("--splats", required=True, help="input gsplat PLY")
    ap.add_argument("--output", required=True, help="output gsplat PLY")
    ap.add_argument("--pad", type=float, default=0.0,
                    help="[aabb] padding added to each side of AABB (in scene units)")
    ap.add_argument("--pad_pct", type=float, default=0.0,
                    help="[aabb] padding as percentage of AABB size (per axis)")
    ap.add_argument("--max_distance", type=float, default=0.10,
                    help="[distance] keep splats within this distance of nearest ref point")
    ap.add_argument("--ref_downsample", type=int, default=1,
                    help="[distance] use every N-th reference point (1 = all, faster if >1)")
    args = ap.parse_args()

    print(f"[1/4] reference: {args.reference}")
    ref = parse_ply(args.reference)
    if not all(c in ref.dtype.names for c in ("x", "y", "z")):
        raise RuntimeError(f"reference PLY lacks x/y/z props (got {ref.dtype.names})")
    rx = np.stack([ref["x"], ref["y"], ref["z"]], axis=-1).astype(np.float64)
    print(f"    {rx.shape[0]} reference points")

    print(f"[2/4] splats: {args.splats}")
    splats = load_gsplat_ply(args.splats)
    means = splats["means"].numpy().astype(np.float64)
    N = means.shape[0]
    print(f"    {N} splats loaded")

    print(f"[3/4] filtering ({args.mode})...")
    if args.mode == "aabb":
        bb_min = rx.min(axis=0)
        bb_max = rx.max(axis=0)
        size = bb_max - bb_min
        pad_abs = args.pad + args.pad_pct * size
        bb_min -= pad_abs
        bb_max += pad_abs
        print(f"    AABB min: {bb_min}")
        print(f"    AABB max: {bb_max}")
        print(f"    AABB size: {bb_max - bb_min}  (pad={args.pad}+{args.pad_pct*100:.1f}%)")
        inside = np.all((means >= bb_min) & (means <= bb_max), axis=1)
    else:
        from scipy.spatial import cKDTree
        if args.ref_downsample > 1:
            rx_use = rx[::args.ref_downsample]
            print(f"    downsampled reference: {rx_use.shape[0]} pts (every {args.ref_downsample}-th)")
        else:
            rx_use = rx
        print(f"    building KD-tree on {rx_use.shape[0]} ref points...")
        tree = cKDTree(rx_use)
        print(f"    querying nearest distance for {N} splats (threshold={args.max_distance})...")
        dists, _ = tree.query(means, k=1, workers=-1)
        inside = dists <= args.max_distance
        print(f"    distance stats: min={dists.min():.4f}, median={np.median(dists):.4f}, "
              f"p95={np.quantile(dists, 0.95):.4f}, max={dists.max():.4f}")

    n_keep = int(inside.sum())
    print(f"    inside: {n_keep} / {N} ({100*n_keep/N:.1f}%)")
    if n_keep == 0:
        raise RuntimeError("Empty result — check coordinate systems / units of both PLY files match.")

    mask = torch.from_numpy(inside)
    out = {
        "means":     splats["means"][mask].contiguous(),
        "sh0":       splats["sh0"][mask].contiguous(),
        "shN":       splats["shN"][mask].contiguous(),
        "opacities": splats["opacities"][mask].contiguous(),
        "scales":    splats["scales"][mask].contiguous(),
        "quats":     splats["quats"][mask].contiguous(),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    print(f"[4/4] writing {args.output}")
    export_splats(
        means=out["means"],
        scales=out["scales"],
        quats=out["quats"],
        opacities=out["opacities"].flatten(),
        sh0=out["sh0"],
        shN=out["shN"],
        format="ply",
        save_to=args.output,
    )
    sz_mb = os.path.getsize(args.output) / 1e6
    print(f"    done — {sz_mb:.1f} MB, {n_keep} splats")


if __name__ == "__main__":
    main()
