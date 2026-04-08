import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tyro
from PIL import Image
from torch import Tensor, optim


@dataclass
class Config:
    # Optional path to input image; if None, a synthetic 2-color target is used.
    img_path: Optional[Path] = None
    # Output directory for artifacts.
    result_dir: str = "results/cpu_trainer"
    # Number of optimization steps.
    iterations: int = 200
    # Number of Gaussian points.
    num_points: int = 1500
    # Learning rate.
    lr: float = 0.01
    # Target resolution (used only for synthetic target).
    height: int = 256
    width: int = 256
    # Save progress GIF.
    save_imgs: bool = True


class CPUSimpleTrainer:
    def __init__(self, gt_image: Tensor, num_points: int) -> None:
        self.device = torch.device("cpu")
        self.gt_image = gt_image.to(self.device)
        self.num_points = num_points

        self.H, self.W = gt_image.shape[0], gt_image.shape[1]
        fov_x = math.pi / 2.0
        self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x)

        self._init_gaussians()
        self._pixel_grid = None

    def _init_gaussians(self) -> None:
        bd = 2.0
        d = 3
        self.means = bd * (torch.rand(self.num_points, 3, device=self.device) - 0.5)
        self.scales = torch.rand(self.num_points, 3, device=self.device)
        self.rgbs = torch.rand(self.num_points, d, device=self.device)
        self.opacities = torch.ones(self.num_points, device=self.device)

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

        for t in (self.means, self.scales, self.rgbs, self.opacities):
            t.requires_grad = True

    def _cpu_render(self, K: Tensor) -> Tensor:
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
        alpha = torch.clamp(den, 0.0, 1.0)
        return out * alpha + bg * (1.0 - alpha)

    def train(self, iterations: int, lr: float, save_imgs: bool, result_dir: str) -> None:
        optimizer = optim.Adam([self.rgbs, self.means, self.scales, self.opacities], lr)
        mse_loss = torch.nn.MSELoss()

        K = torch.tensor(
            [
                [self.focal, 0, self.W / 2],
                [0, self.focal, self.H / 2],
                [0, 0, 1],
            ],
            device=self.device,
        )

        frames = []
        times = [0.0, 0.0]
        os.makedirs(result_dir, exist_ok=True)

        for step in range(iterations):
            t0 = time.time()
            out_img = self._cpu_render(K)
            times[0] += time.time() - t0

            loss = mse_loss(out_img, self.gt_image)
            optimizer.zero_grad()
            t1 = time.time()
            loss.backward()
            optimizer.step()
            times[1] += time.time() - t1

            print(f"Iteration {step + 1}/{iterations}, Loss: {loss.item():.6f}")
            if save_imgs and step % 5 == 0:
                frames.append((out_img.detach().cpu().numpy() * 255).astype(np.uint8))

        if save_imgs and frames:
            pil_frames = [Image.fromarray(frame) for frame in frames]
            out_gif = os.path.join(result_dir, "training_cpu.gif")
            pil_frames[0].save(
                out_gif,
                save_all=True,
                append_images=pil_frames[1:],
                optimize=False,
                duration=5,
                loop=0,
            )
            print(f"Saved visualization: {out_gif}")

        print(f"Total(s): rasterization={times[0]:.3f}, backward={times[1]:.3f}")
        print(
            f"Per step(s): rasterization={times[0]/iterations:.5f}, backward={times[1]/iterations:.5f}"
        )


def image_path_to_tensor(image_path: Path) -> Tensor:
    import torchvision.transforms as transforms

    img = Image.open(image_path)
    transform = transforms.ToTensor()
    return transform(img).permute(1, 2, 0)[..., :3]


def make_synthetic_target(height: int, width: int) -> Tensor:
    gt_image = torch.ones((height, width, 3))
    gt_image[: height // 2, : width // 2, :] = torch.tensor([1.0, 0.0, 0.0])
    gt_image[height // 2 :, width // 2 :, :] = torch.tensor([0.0, 0.0, 1.0])
    return gt_image


def main(cfg: Config) -> None:
    if cfg.img_path:
        gt_image = image_path_to_tensor(cfg.img_path)
    else:
        gt_image = make_synthetic_target(cfg.height, cfg.width)

    trainer = CPUSimpleTrainer(gt_image=gt_image, num_points=cfg.num_points)
    trainer.train(
        iterations=cfg.iterations,
        lr=cfg.lr,
        save_imgs=cfg.save_imgs,
        result_dir=cfg.result_dir,
    )


if __name__ == "__main__":
    main(tyro.cli(Config))

