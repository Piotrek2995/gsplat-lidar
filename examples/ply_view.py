"""Read-only web viewer for gsplat .ply files.

Renders splats with the standard GsplatViewer UI (camera, render quality
options) but no Edit panel, no selection box, no buttons. Use ply_editor.py
if you want to edit; use this when you just want to look.

Usage:
    python ply_view.py --ply results/atlas_v3_reg/clean/clean_shape.ply --port 8088
"""

import argparse
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import viser
from nerfview import CameraState, RenderTabState

from gsplat.rendering import rasterization
from gsplat_viewer import GsplatViewer, GsplatRenderTabState


def load_gsplat_ply(path: str) -> Dict[str, np.ndarray]:
    with open(path, "rb") as f:
        header_lines: List[bytes] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"unexpected EOF in PLY header: {path}")
            header_lines.append(line)
            if line.strip() == b"end_header":
                break
        header = b"".join(header_lines).decode("ascii", errors="replace")

        fmt_line = next((ln for ln in header.splitlines() if ln.startswith("format")), "")
        if "binary_little_endian" not in fmt_line:
            raise ValueError(f"only binary_little_endian PLY supported, got: {fmt_line!r}")

        n = None
        props: List[Tuple[str, str]] = []
        for ln in header.splitlines():
            if ln.startswith("element vertex"):
                n = int(ln.split()[-1])
            elif ln.startswith("property"):
                parts = ln.split()
                props.append((parts[1], parts[2]))
        if n is None:
            raise ValueError("no 'element vertex' line in PLY header")
        if any(dtype != "float" for dtype, _ in props):
            raise ValueError(f"only float32 PLY properties supported, got: {props}")

        record_size = 4 * len(props)
        raw = f.read(n * record_size)

    arr = np.frombuffer(raw, dtype="<f4").reshape(n, len(props)).copy()
    idx = {name: i for i, (_, name) in enumerate(props)}

    def take(*names):
        return arr[:, [idx[k] for k in names]]

    means = take("x", "y", "z")
    sh0 = take("f_dc_0", "f_dc_1", "f_dc_2").reshape(n, 1, 3)

    f_rest_names = sorted(
        [k for k in idx if k.startswith("f_rest_")],
        key=lambda s: int(s.split("_")[-1]),
    )
    if f_rest_names:
        f_rest_flat = arr[:, [idx[k] for k in f_rest_names]]
        k_minus_1 = len(f_rest_names) // 3
        shN = f_rest_flat.reshape(n, 3, k_minus_1).transpose(0, 2, 1).copy()
    else:
        shN = np.zeros((n, 0, 3), dtype=np.float32)

    opacities = arr[:, idx["opacity"]].copy()
    scales = take("scale_0", "scale_1", "scale_2")
    quats = take("rot_0", "rot_1", "rot_2", "rot_3")

    return {
        "means": means.astype(np.float32),
        "sh0": sh0.astype(np.float32),
        "shN": shN.astype(np.float32),
        "opacities": opacities.astype(np.float32),
        "scales": scales.astype(np.float32),
        "quats": quats.astype(np.float32),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = args.device

    raw = load_gsplat_ply(args.ply)
    means     = torch.from_numpy(raw["means"]).to(device)
    quats     = F.normalize(torch.from_numpy(raw["quats"]).to(device), dim=-1)
    scales    = torch.exp(torch.from_numpy(raw["scales"]).to(device))
    opacities = torch.sigmoid(torch.from_numpy(raw["opacities"]).to(device))
    colors    = torch.cat(
        [torch.from_numpy(raw["sh0"]).to(device), torch.from_numpy(raw["shN"]).to(device)],
        dim=1,
    )
    sh_degree = int(math.isqrt(colors.shape[1])) - 1
    N = means.shape[0]

    center = means.median(dim=0).values
    d = (means - center).norm(dim=-1)
    scene_radius = float(d.quantile(0.95).item())
    scene_center = center.cpu().numpy()
    print(f"Loaded {N} splats from {args.ply}  (sh_degree={sh_degree}, scene_radius={scene_radius:.3f})")

    @torch.no_grad()
    def render_fn(camera_state: CameraState, render_tab_state: RenderTabState):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            W, H = render_tab_state.render_width, render_tab_state.render_height
        else:
            W, H = render_tab_state.viewer_width, render_tab_state.viewer_height

        c2w = torch.from_numpy(camera_state.c2w).float().to(device)
        K = torch.from_numpy(camera_state.get_K((W, H))).float().to(device)
        viewmat = c2w.inverse()

        try:
            render_colors, _, _ = rasterization(
                means=means, quats=quats, scales=scales,
                opacities=opacities, colors=colors,
                viewmats=viewmat[None], Ks=K[None], width=W, height=H,
                sh_degree=min(render_tab_state.max_sh_degree, sh_degree),
                near_plane=render_tab_state.near_plane,
                far_plane=render_tab_state.far_plane,
                radius_clip=render_tab_state.radius_clip,
                eps2d=render_tab_state.eps2d,
                backgrounds=torch.tensor([list(render_tab_state.backgrounds)],
                                         device=device, dtype=torch.float32),
                packed=False,
                rasterize_mode=render_tab_state.rasterize_mode,
                camera_model=render_tab_state.camera_model,
            )
            render_tab_state.total_gs_count = N
            render_tab_state.rendered_gs_count = N
            return render_colors[0, ..., :3].cpu().numpy()
        except Exception as e:
            print(f"[render] {e}")
            return np.full((H, W, 3), 0.2, dtype=np.float32)

    server = viser.ViserServer(port=args.port, verbose=False)
    GsplatViewer(
        server=server,
        render_fn=render_fn,
        output_dir=Path("/tmp"),
        mode="rendering",
    )

    cam_target = np.asarray(scene_center, dtype=np.float32)
    cam_pos = cam_target + np.array([0.0, -3.0 * scene_radius, 1.5 * scene_radius], dtype=np.float32)

    @server.on_client_connect
    def _on_connect(client):
        client.camera.look_at = tuple(map(float, cam_target))
        client.camera.position = tuple(map(float, cam_pos))
        client.camera.near = 1e-3
        client.camera.far = max(50.0, 20.0 * scene_radius)

    print(f"Viewer running on http://localhost:{args.port}")
    print("Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Bye.")


if __name__ == "__main__":
    main()
