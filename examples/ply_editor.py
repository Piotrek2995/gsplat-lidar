"""Interactive web editor for gsplat .ply files.

Loads a Gaussian-splat PLY (as produced by gsplat.export_splats), lets you:
  * preview live opacity / scale filters with sliders
  * place a 3D box (translate + rotate, resize via sliders) and delete splats
    inside or outside
  * undo, reset, focus camera
  * save a new PLY with the current edits applied

Usage:
  python ply_editor.py \
      --ply results/atlas_v3_reg/clean/clean.ply \
      --output_dir results/atlas_v3_reg/edited \
      --port 8088
"""

import argparse
import math
import os
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import viser
from nerfview import CameraState, RenderTabState

from gsplat import export_splats
from gsplat.rendering import rasterization
from gsplat_viewer import GsplatViewer, GsplatRenderTabState


# ---------- PLY I/O ----------

def load_gsplat_ply(path: str) -> Dict[str, np.ndarray]:
    """Read a gsplat-format PLY (float32 binary little-endian) → numpy dict.

    Returns raw (pre-activation) tensors with these keys:
      means     (N,3), sh0 (N,1,3), shN (N,K-1,3),
      opacities (N,),  scales (N,3), quats (N,4)
    """
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
    sh0_flat = take("f_dc_0", "f_dc_1", "f_dc_2")  # (N,3)
    sh0 = sh0_flat.reshape(n, 1, 3)

    # shN: f_rest_* are channel-major (export_splats does permute(0,2,1).reshape(N,-1))
    f_rest_names = sorted(
        [k for k in idx if k.startswith("f_rest_")],
        key=lambda s: int(s.split("_")[-1]),
    )
    if f_rest_names:
        f_rest_flat = arr[:, [idx[k] for k in f_rest_names]]  # (N, 3*(K-1))
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


def quat_to_rotmat(wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = wxyz
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


# ---------- editor state ----------

class EditorState:
    def __init__(self, ply_path: str, device: str):
        self.device = device
        self.ply_path = ply_path
        raw = load_gsplat_ply(ply_path)
        self.raw_means     = torch.from_numpy(raw["means"]).to(device)
        self.raw_sh0       = torch.from_numpy(raw["sh0"]).to(device)
        self.raw_shN       = torch.from_numpy(raw["shN"]).to(device)
        self.raw_opacities = torch.from_numpy(raw["opacities"]).to(device)
        self.raw_scales    = torch.from_numpy(raw["scales"]).to(device)
        self.raw_quats     = torch.from_numpy(raw["quats"]).to(device)

        self.means     = self.raw_means
        self.quats     = F.normalize(self.raw_quats, dim=-1)
        self.scales    = torch.exp(self.raw_scales)
        self.opacities = torch.sigmoid(self.raw_opacities)
        self.colors    = torch.cat([self.raw_sh0, self.raw_shN], dim=1)
        self.sh_degree = int(math.isqrt(self.colors.shape[1])) - 1
        self.N = self.means.shape[0]

        self.active: torch.Tensor = torch.ones(self.N, dtype=torch.bool, device=device)
        self.history: List[torch.Tensor] = []
        self.opacity_min = 0.0
        self.scale_max = float("inf")

        center = self.means.median(dim=0).values
        d = (self.means - center).norm(dim=-1)
        self.scene_radius = float(d.quantile(0.95).item())
        self.scene_center = center.cpu().numpy()

    def view_mask(self) -> torch.Tensor:
        m = self.active
        if self.opacity_min > 0:
            m = m & (self.opacities >= self.opacity_min)
        if math.isfinite(self.scale_max):
            m = m & (self.scales.max(dim=-1).values <= self.scale_max)
        return m

    def push_undo(self):
        self.history.append(self.active.clone())
        if len(self.history) > 30:
            self.history.pop(0)

    def undo(self) -> bool:
        if not self.history:
            return False
        self.active = self.history.pop()
        return True

    def points_in_box(self, pos: np.ndarray, wxyz: np.ndarray, size: Tuple[float, float, float]) -> torch.Tensor:
        R = torch.from_numpy(quat_to_rotmat(np.asarray(wxyz))).float().to(self.device)
        p = torch.from_numpy(np.asarray(pos)).float().to(self.device)
        half = torch.tensor([size[0] / 2, size[1] / 2, size[2] / 2], device=self.device)
        local = (self.means - p) @ R
        return (local.abs() <= half).all(dim=-1)


# ---------- safe-callback wrapper (don't crash the server on errors) ----------

def safe(fn):
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            print(f"[editor] callback {fn.__name__} failed: {e}")
            traceback.print_exc()
    wrapper.__name__ = fn.__name__
    return wrapper


# ---------- main ----------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ply", required=True, help="input .ply (gsplat format)")
    p.add_argument("--output_dir", required=True, help="where to save edited .ply outputs")
    p.add_argument("--port", type=int, default=8088)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device

    state = EditorState(args.ply, device)
    print(f"Loaded {state.N} splats from {args.ply}  (sh_degree={state.sh_degree}, scene_radius={state.scene_radius:.3f})")

    @torch.no_grad()
    def viewer_render_fn(camera_state: CameraState, render_tab_state: RenderTabState):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            W, H = render_tab_state.render_width, render_tab_state.render_height
        else:
            W, H = render_tab_state.viewer_width, render_tab_state.viewer_height

        mask = state.view_mask()
        if mask.sum() == 0:
            return np.zeros((H, W, 3), dtype=np.float32)

        means = state.means[mask]
        quats = state.quats[mask]
        scales = state.scales[mask]
        opa = state.opacities[mask]
        colors = state.colors[mask]

        c2w = torch.from_numpy(camera_state.c2w).float().to(device)
        K = torch.from_numpy(camera_state.get_K((W, H))).float().to(device)
        viewmat = c2w.inverse()

        try:
            render_colors, _, _ = rasterization(
                means=means, quats=quats, scales=scales, opacities=opa, colors=colors,
                viewmats=viewmat[None], Ks=K[None], width=W, height=H,
                sh_degree=min(render_tab_state.max_sh_degree, state.sh_degree),
                near_plane=render_tab_state.near_plane,
                far_plane=render_tab_state.far_plane,
                radius_clip=render_tab_state.radius_clip,
                eps2d=render_tab_state.eps2d,
                backgrounds=torch.tensor([list(render_tab_state.backgrounds)], device=device, dtype=torch.float32),
                packed=False,
                rasterize_mode=render_tab_state.rasterize_mode,
                camera_model=render_tab_state.camera_model,
            )
            render_tab_state.total_gs_count = int(state.active.sum().item())
            render_tab_state.rendered_gs_count = int(mask.sum().item())
            return render_colors[0, ..., :3].cpu().numpy()
        except Exception as e:
            print(f"[render] {e}")
            return np.full((H, W, 3), 0.2, dtype=np.float32)

    server = viser.ViserServer(port=args.port, verbose=False)
    viewer = GsplatViewer(
        server=server,
        render_fn=viewer_render_fn,
        output_dir=Path(args.output_dir),
        mode="rendering",
    )

    cam_target = np.asarray(state.scene_center, dtype=np.float32)
    cam_pos = cam_target + np.array([0.0, -3.0 * state.scene_radius, 1.5 * state.scene_radius], dtype=np.float32)

    def focus_clients(target: np.ndarray, position: np.ndarray):
        for c in server.get_clients().values():
            c.camera.look_at = tuple(map(float, target))
            c.camera.position = tuple(map(float, position))

    @server.on_client_connect
    def _on_connect(client):
        client.camera.look_at = tuple(map(float, cam_target))
        client.camera.position = tuple(map(float, cam_pos))
        client.camera.near = 1e-3
        client.camera.far = max(50.0, 20.0 * state.scene_radius)

    # ===== Edit panel =====
    with server.gui.add_folder("Edit"):
        status_md = server.gui.add_markdown(f"**Active:** {state.N}/{state.N}  |  **history:** 0")

        with server.gui.add_folder("Live thresholds"):
            opa_slider = server.gui.add_slider(
                "opacity min", min=0.0, max=1.0, step=0.005, initial_value=0.0,
                hint="Hide splats with sigmoid(opacity) below this. Non-destructive until 'Bake'.",
            )
            scl_slider = server.gui.add_slider(
                "scale max (×radius)", min=0.001, max=2.0, step=0.001, initial_value=2.0,
                hint="Hide splats whose max scale exceeds this × scene_radius.",
            )
            bake_btn = server.gui.add_button("Bake thresholds → mask")

        with server.gui.add_folder("Box selection"):
            init_size = max(state.scene_radius * 0.5, 0.1)
            tc = server.scene.add_transform_controls(
                "/edit_box",
                scale=init_size,
                position=tuple(state.scene_center.tolist()),
                wxyz=(1.0, 0.0, 0.0, 0.0),
                disable_sliders=True,
            )
            box_name = "/edit_box/wire"
            current_dims = [init_size, init_size, init_size]

            def rebuild_box():
                server.scene.add_box(
                    box_name, color=(255, 220, 0),
                    dimensions=tuple(current_dims),
                    wireframe=True, opacity=0.5,
                )
            rebuild_box()

            box_show = server.gui.add_checkbox("Show box", initial_value=True)
            bw = server.gui.add_number("width",  initial_value=current_dims[0], min=0.01, step=0.05)
            bh = server.gui.add_number("height", initial_value=current_dims[1], min=0.01, step=0.05)
            bd = server.gui.add_number("depth",  initial_value=current_dims[2], min=0.01, step=0.05)

            def _update_dims(_=None):
                current_dims[0] = float(bw.value)
                current_dims[1] = float(bh.value)
                current_dims[2] = float(bd.value)
                rebuild_box()
            bw.on_update(_update_dims)
            bh.on_update(_update_dims)
            bd.on_update(_update_dims)

            center_btn  = server.gui.add_button("Center box on scene")
            del_in_btn  = server.gui.add_button("Delete INSIDE box", color="red")
            del_out_btn = server.gui.add_button("Delete OUTSIDE box", color="red")

        with server.gui.add_folder("Camera"):
            focus_scene_btn = server.gui.add_button("Focus on scene center")
            focus_box_btn   = server.gui.add_button("Focus on box")
            zoom_factor = server.gui.add_slider(
                "distance (×radius)", min=0.5, max=10.0, step=0.1, initial_value=3.0,
            )

        with server.gui.add_folder("Actions"):
            undo_btn     = server.gui.add_button("Undo last")
            reset_btn    = server.gui.add_button("Reset mask")
            save_ply_btn = server.gui.add_button("Save .ply")

    def refresh_status():
        n_active = int(state.active.sum().item())
        n_view = int(state.view_mask().sum().item())
        status_md.content = (
            f"**Active:** {n_active}/{state.N}  |  "
            f"**Rendered:** {n_view}  |  "
            f"**history:** {len(state.history)}"
        )

    refresh_status()

    # ===== callbacks =====
    @opa_slider.on_update
    @safe
    def _(_):
        state.opacity_min = float(opa_slider.value)
        refresh_status()
        viewer.rerender(_)

    @scl_slider.on_update
    @safe
    def _(_):
        v = float(scl_slider.value)
        state.scale_max = v * state.scene_radius if v < 2.0 else float("inf")
        refresh_status()
        viewer.rerender(_)

    @bake_btn.on_click
    @safe
    def _(_):
        state.push_undo()
        state.active = state.view_mask().clone()
        state.opacity_min = 0.0
        state.scale_max = float("inf")
        opa_slider.value = 0.0
        scl_slider.value = 2.0
        refresh_status()
        viewer.rerender(_)

    @box_show.on_update
    @safe
    def _(_):
        if box_show.value:
            rebuild_box()
        else:
            server.scene.remove_by_name(box_name)

    @center_btn.on_click
    @safe
    def _(_):
        tc.position = tuple(state.scene_center.tolist())
        tc.wxyz = (1.0, 0.0, 0.0, 0.0)

    @del_in_btn.on_click
    @safe
    def _(_):
        state.push_undo()
        inside = state.points_in_box(np.asarray(tc.position), np.asarray(tc.wxyz), tuple(current_dims))
        n = int(inside.sum().item())
        state.active = state.active & (~inside)
        print(f"Deleted INSIDE: {n}")
        refresh_status()
        viewer.rerender(_)

    @del_out_btn.on_click
    @safe
    def _(_):
        state.push_undo()
        inside = state.points_in_box(np.asarray(tc.position), np.asarray(tc.wxyz), tuple(current_dims))
        n = int((~inside).sum().item())
        state.active = state.active & inside
        print(f"Deleted OUTSIDE: {n}")
        refresh_status()
        viewer.rerender(_)

    @focus_scene_btn.on_click
    @safe
    def _(_):
        d = float(zoom_factor.value) * state.scene_radius
        offset = np.array([0.0, -d, d * 0.5], dtype=np.float32)
        focus_clients(cam_target, cam_target + offset)

    @focus_box_btn.on_click
    @safe
    def _(_):
        target = np.asarray(tc.position, dtype=np.float32)
        d = float(zoom_factor.value) * max(*current_dims) * 1.5
        offset = np.array([0.0, -d, d * 0.5], dtype=np.float32)
        focus_clients(target, target + offset)

    @undo_btn.on_click
    @safe
    def _(_):
        if state.undo():
            print("Undo OK")
        else:
            print("Nothing to undo")
        refresh_status()
        viewer.rerender(_)

    @reset_btn.on_click
    @safe
    def _(_):
        state.push_undo()
        state.active = torch.ones(state.N, dtype=torch.bool, device=device)
        refresh_status()
        viewer.rerender(_)

    @save_ply_btn.on_click
    @safe
    def _(_):
        keep = state.view_mask()
        out = os.path.join(args.output_dir, f"edited_{int(time.time())}.ply")
        export_splats(
            means=state.raw_means[keep],
            scales=state.raw_scales[keep],
            quats=state.raw_quats[keep],
            opacities=state.raw_opacities[keep],
            sh0=state.raw_sh0[keep],
            shN=state.raw_shN[keep],
            format="ply",
            save_to=out,
        )
        print(f"Wrote {out}  ({int(keep.sum().item())} splats)")

    print(f"Editor running on http://localhost:{args.port}")
    print("Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Bye.")


if __name__ == "__main__":
    main()
