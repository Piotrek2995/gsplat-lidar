import math
import os
import time
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import tyro
from PIL import Image
from torch import Tensor, optim

from gsplat import rasterization, rasterization_2dgs


class SimpleTrainer:
    """Trains random gaussians to fit an image."""

    def __init__(
        self,
        gt_image: Tensor,
        num_points: int = 2000,
        device: Literal["auto", "cpu", "cuda"] = "auto",
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device but CUDA is not available.")
        self.device = torch.device("cuda:0" if device == "cuda" else "cpu")
        self.gt_image = gt_image.to(device=self.device)
        self.num_points = num_points

        fov_x = math.pi / 2.0
        self.H, self.W = gt_image.shape[0], gt_image.shape[1]
        self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x)
        self.img_size = torch.tensor([self.W, self.H, 1], device=self.device)

        self._init_gaussians()
        self._pixel_grid = None

    def _sync_if_cuda(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def _cpu_render(self, K: Tensor) -> Tensor:
        """Simple differentiable Gaussian fallback for CPU-only demo runs."""
        means_c = (
            torch.einsum("ij,nj->ni", self.viewmat[:3, :3], self.means)
            + self.viewmat[:3, 3][None, :]
        )
        z = means_c[:, 2].clamp(min=1e-3)
        u = K[0, 0] * means_c[:, 0] / z + K[0, 2]
        v = K[1, 1] * means_c[:, 1] / z + K[1, 2]

        sxy = self.scales[:, :2].mean(dim=-1).clamp(min=1e-3)
        sigma = (sxy * K[0, 0] / z).clamp(min=0.7, max=12.0)

        if self._pixel_grid is None:
            ys = torch.arange(self.H, device=self.device, dtype=self.means.dtype)
            xs = torch.arange(self.W, device=self.device, dtype=self.means.dtype)
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            self._pixel_grid = torch.stack([gx, gy], dim=-1)

        grid = self._pixel_grid
        colors = torch.sigmoid(self.rgbs)
        opacities = torch.sigmoid(self.opacities)

        num = torch.zeros((self.H, self.W, colors.shape[-1]), device=self.device)
        den = torch.zeros((self.H, self.W, 1), device=self.device)

        chunk = 256
        for i in range(0, self.num_points, chunk):
            uu = u[i : i + chunk][:, None, None]
            vv = v[i : i + chunk][:, None, None]
            ss = sigma[i : i + chunk][:, None, None].clamp(min=1e-3)
            cc = colors[i : i + chunk]
            aa = opacities[i : i + chunk][:, None, None, None]

            dx = grid[None, :, :, 0] - uu
            dy = grid[None, :, :, 1] - vv
            dist2 = dx * dx + dy * dy
            w = aa * torch.exp(-0.5 * dist2 / (ss * ss))[..., None]
            num = num + (w * cc[:, None, None, :]).sum(dim=0)
            den = den + w.sum(dim=0)

        out = num / den.clamp(min=1e-6)
        bg = torch.sigmoid(self.background)[None, None, :]
        return out * torch.clamp(den, 0.0, 1.0) + bg * (1.0 - torch.clamp(den, 0.0, 1.0))

    def _init_gaussians(self):
        """Random gaussians"""
        bd = 2

        self.means = bd * (torch.rand(self.num_points, 3, device=self.device) - 0.5)
        self.scales = torch.rand(self.num_points, 3, device=self.device)
        d = 3
        self.rgbs = torch.rand(self.num_points, d, device=self.device)

        u = torch.rand(self.num_points, 1, device=self.device)
        v = torch.rand(self.num_points, 1, device=self.device)
        w = torch.rand(self.num_points, 1, device=self.device)

        self.quats = torch.cat(
            [
                torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
                torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
                torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
                torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
            ],
            -1,
        )
        self.opacities = torch.ones((self.num_points), device=self.device)

        self.viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 8.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
        )
        self.background = torch.zeros(d, device=self.device)

        self.means.requires_grad = True
        self.scales.requires_grad = True
        self.quats.requires_grad = True
        self.rgbs.requires_grad = True
        self.opacities.requires_grad = True
        self.viewmat.requires_grad = False

    def train(
        self,
        iterations: int = 1000,
        lr: float = 0.01,
        save_imgs: bool = False,
        model_type: Literal["3dgs", "2dgs"] = "3dgs",
    ):
        optimizer = optim.Adam(
            [self.rgbs, self.means, self.scales, self.opacities, self.quats], lr
        )
        mse_loss = torch.nn.MSELoss()
        frames = []
        times = [0] * 2  # rasterization, backward
        K = torch.tensor(
            [
                [self.focal, 0, self.W / 2],
                [0, self.focal, self.H / 2],
                [0, 0, 1],
            ],
            device=self.device,
        )

        if model_type == "3dgs":
            rasterize_fnc = rasterization
        elif model_type == "2dgs":
            rasterize_fnc = rasterization_2dgs
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        for iter in range(iterations):
            start = time.time()

            if self.device.type == "cuda":
                renders = rasterize_fnc(
                    self.means,
                    self.quats / self.quats.norm(dim=-1, keepdim=True),
                    self.scales,
                    torch.sigmoid(self.opacities),
                    torch.sigmoid(self.rgbs),
                    self.viewmat[None],
                    K[None],
                    self.W,
                    self.H,
                    packed=False,
                )[0]
                out_img = renders[0]
            else:
                out_img = self._cpu_render(K)

            self._sync_if_cuda()
            times[0] += time.time() - start
            loss = mse_loss(out_img, self.gt_image)
            optimizer.zero_grad()
            start = time.time()
            loss.backward()
            self._sync_if_cuda()
            times[1] += time.time() - start
            optimizer.step()
            print(f"Iteration {iter + 1}/{iterations}, Loss: {loss.item()}")

            if save_imgs and iter % 5 == 0:
                frames.append((out_img.detach().cpu().numpy() * 255).astype(np.uint8))
        if save_imgs:
            # save them as a gif with PIL
            frames = [Image.fromarray(frame) for frame in frames]
            out_dir = os.path.join(os.getcwd(), "results")
            os.makedirs(out_dir, exist_ok=True)
            frames[0].save(
                f"{out_dir}/training.gif",
                save_all=True,
                append_images=frames[1:],
                optimize=False,
                duration=5,
                loop=0,
            )
        print(f"Total(s):\nRasterization: {times[0]:.3f}, Backward: {times[1]:.3f}")
        print(
            f"Per step(s):\nRasterization: {times[0]/iterations:.5f}, Backward: {times[1]/iterations:.5f}"
        )


def image_path_to_tensor(image_path: Path):
    import torchvision.transforms as transforms

    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).permute(1, 2, 0)[..., :3]
    return img_tensor


def main(
    height: int = 256,
    width: int = 256,
    num_points: int = 100000,
    save_imgs: bool = True,
    img_path: Optional[Path] = None,
    iterations: int = 1000,
    lr: float = 0.01,
    model_type: Literal["3dgs", "2dgs"] = "3dgs",
    device: Literal["auto", "cpu", "cuda"] = "auto",
) -> None:
    if img_path:
        gt_image = image_path_to_tensor(img_path)
    else:
        gt_image = torch.ones((height, width, 3)) * 1.0
        # make top left and bottom right red, blue
        gt_image[: height // 2, : width // 2, :] = torch.tensor([1.0, 0.0, 0.0])
        gt_image[height // 2 :, width // 2 :, :] = torch.tensor([0.0, 0.0, 1.0])

    trainer = SimpleTrainer(gt_image=gt_image, num_points=num_points, device=device)
    trainer.train(
        iterations=iterations,
        lr=lr,
        save_imgs=save_imgs,
        model_type=model_type,
    )


if __name__ == "__main__":
    tyro.cli(main)
