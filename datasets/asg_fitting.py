from pathlib import Path

import click
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import TQDMProgressBar
from scipy.optimize import linear_sum_assignment
from torch.utils.data import Dataset

from utils import (
    IMAGE_SUFFIXES,
    cv,
    distributed_items,
    image_paths,
    prediction_dataloader,
    read_image,
    rotation_output_names,
    select_average_params,
    update_best_params,
    write_image,
)

TINY = 1e-8
IMAGE_EXT = ".png"


def tangent_frame(normal):
    nx, ny, nz = np.split(normal, 3, axis=-1)
    denom = np.sqrt(nx * nx + nz * nz + TINY)
    tangent = np.concatenate((-nx * ny / denom, denom, -ny * nz / denom), axis=-1)
    tangent /= np.linalg.norm(tangent, axis=-1, keepdims=True)
    return tangent, np.cross(normal, tangent)


def rotate_vectors(vectors, angle):
    cosine = np.cos(angle)
    sine = np.sin(angle)
    result = vectors.copy()
    result[..., 0] = cosine * vectors[..., 0] + sine * vectors[..., 2]
    result[..., 2] = -sine * vectors[..., 0] + cosine * vectors[..., 2]
    return result


def rotate_asg(params, normals, index, rotate_count):
    if index == 0:
        return params.copy()
    angle = 2 * np.pi * index / rotate_count
    source_tangent, source_cross = tangent_frame(normals)
    lobe_tangent = (
        np.cos(params[:, :1]) * source_tangent + np.sin(params[:, :1]) * source_cross
    )
    rotated_normals = rotate_vectors(normals, angle)
    rotated_tangent = rotate_vectors(lobe_tangent, angle)
    source_indices, target_indices = linear_sum_assignment(
        -(rotated_normals @ normals.T)
    )

    target_tangent, target_cross = tangent_frame(normals)
    result = np.empty_like(params)
    result[target_indices] = params[source_indices]
    aligned_tangent = target_tangent[target_indices]
    aligned_cross = target_cross[target_indices]
    transformed = rotated_tangent[source_indices]
    result[target_indices, 0] = np.remainder(
        np.arctan2(
            np.sum(transformed * aligned_cross, axis=-1),
            np.sum(transformed * aligned_tangent, axis=-1),
        ),
        2 * np.pi,
    )
    return result


class ASGFitDataset(Dataset):
    def __init__(
        self,
        input_path,
        save_path,
        image_path,
        width,
        rotate_count,
        skip_existing,
        max_count,
    ):
        self.save_path = Path(save_path)
        self.image_path = Path(image_path)
        self.rotate_count = rotate_count
        all_paths = image_paths(input_path, IMAGE_SUFFIXES)
        all_names = {path.stem for path in all_paths}
        img_list = all_paths
        if rotate_count > 1:
            img_list = [path for path in img_list if path.stem.endswith("_00")]
            missing = [
                name
                for path in img_list
                for name in rotation_output_names(path.stem, rotate_count)
                if name not in all_names
            ]
            if missing:
                raise ValueError(
                    "missing ASG rotation inputs:\n" + "\n".join(missing[:50])
                )
        if max_count is not None:
            img_list = img_list[:max_count]
        if skip_existing:
            img_list = [p for p in img_list if not self.output_exists(p)]
        self.img_list = distributed_items([path.as_posix() for path in img_list])
        self.width = width
        self.height = width // 2

    def output_exists(self, img_path):
        names = rotation_output_names(Path(img_path).stem, self.rotate_count)
        return all(
            (self.save_path / f"{name}.npy").is_file()
            and (self.image_path / f"{name}{IMAGE_EXT}").is_file()
            for name in names
        )

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img = read_image(img_path, cv.IMREAD_UNCHANGED)
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=-1)
        img = img[..., :3].astype(np.float32) / 255.0
        img = cv.resize(img, (self.width, self.height), interpolation=cv.INTER_AREA)
        img = cv.GaussianBlur(img, (11, 11), 5)
        img = np.nan_to_num(img, nan=0.0).astype(np.float32)
        return torch.from_numpy(img).float(), Path(img_path).stem


class ASGFitDataModule(L.LightningDataModule):
    def __init__(
        self,
        input_path,
        save_path,
        image_path,
        width,
        batch_size,
        num_workers,
        rotate_count,
        skip_existing,
        max_count,
    ):
        super().__init__()
        self.input_path = input_path
        self.save_path = save_path
        self.image_path = image_path
        self.width = width
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.rotate_count = rotate_count
        self.skip_existing = skip_existing
        self.max_count = max_count
        self.predict_data = None

    def setup(self, stage=None):
        self.predict_data = ASGFitDataset(
            self.input_path,
            self.save_path,
            self.image_path,
            self.width,
            self.rotate_count,
            self.skip_existing,
            self.max_count,
        )

    def predict_dataloader(self):
        return prediction_dataloader(
            self.predict_data,
            self.batch_size,
            self.num_workers,
        )


class ASGEnvOptim:
    def __init__(self, steps, width, asg_num, batch_size, lr, avg_steps, device):
        self.steps = steps
        self.width = width
        self.height = width // 2
        self.asg_num = asg_num
        self.batch_size = batch_size
        self.lr = lr
        self.avg_steps = avg_steps
        self.device = device
        self.param = None
        self.opt = None

        lobe_path = (
            Path(__file__).resolve().parents[1]
            / "lighting_est"
            / f"fib_lobes_{asg_num}.npy"
        )
        fib_xyz = np.load(lobe_path)
        fib = torch.from_numpy(fib_xyz.astype(np.float32)).to(device)
        fib = F.normalize(fib + TINY, dim=-1)
        self.normal = fib[None].expand(batch_size, -1, -1)
        self.normal_tan, self.normal_cross = self.get_tangent_frame(self.normal)
        self.ls = self.get_view_dirs().to(device).view(-1, 3)
        self.ls_t = self.ls.t()
        self.smooth = torch.clamp_min(torch.matmul(self.normal, self.ls_t), 0.0)

    def get_view_dirs(self):
        az = torch.arange(self.width) / self.width * 2 * torch.pi
        el = torch.arange(self.height) / self.height * torch.pi
        el, az = torch.meshgrid(el, az, indexing="ij")
        return torch.stack(
            [
                torch.sin(el) * torch.cos(az),
                torch.cos(el),
                torch.sin(el) * torch.sin(az),
            ],
            dim=-1,
        )

    def get_tangent_frame(self, normal):
        nx, ny, nz = torch.split(normal, [1, 1, 1], dim=-1)
        denom = torch.sqrt(nx * nx + nz * nz + TINY)
        normal_tan = torch.cat([-nx * ny / denom, denom, -ny * nz / denom], dim=-1)
        normal_tan = F.normalize(normal_tan, dim=-1)
        normal_cross = torch.cross(normal, normal_tan, dim=-1)
        return normal_tan, normal_cross

    def initialize(self):
        weight = torch.randn(self.batch_size, self.asg_num, 3, device=self.device) / 7.5
        lamb = torch.randn(self.batch_size, self.asg_num, 1, device=self.device) * 20
        mu = torch.randn(self.batch_size, self.asg_num, 1, device=self.device) * 20
        angle = torch.randn(self.batch_size, self.asg_num, 1, device=self.device)

        self.param = torch.cat([angle, lamb, mu, weight], dim=-1)
        self.param = self.param.detach()
        self.param.requires_grad = True
        self.opt = torch.optim.AdamW([self.param], lr=self.lr)

    def render_asg(self, angle, lamb, mu, weight):
        tan = torch.cos(angle) * self.normal_tan + torch.sin(angle) * self.normal_cross
        bi_tan = torch.cross(self.normal, tan, dim=-1)
        tan_dot = torch.matmul(tan, self.ls_t)
        bi_dot = torch.matmul(bi_tan, self.ls_t)
        lamb = lamb.squeeze(-1).unsqueeze(-1)
        mu = mu.squeeze(-1).unsqueeze(-1)
        e_power = lamb * tan_dot.square() + mu * bi_dot.square()
        basis = self.smooth * torch.exp(-e_power)
        envmap = torch.einsum("bnp,bnc->bcp", basis, weight)
        return envmap.reshape(self.batch_size, 3, self.height, self.width)

    def de_parameterize(self):
        angle, lamb, mu, weight = torch.split(
            self.param.view(self.batch_size, self.asg_num, 6), [1, 1, 1, 3], dim=-1
        )
        angle = angle * 2
        lamb = torch.abs(lamb)
        mu = torch.abs(mu)
        weight = torch.abs(weight)
        return angle, lamb, mu, weight

    def optimize(self, envmap):
        envmap = envmap.to(self.device).permute(0, 3, 1, 2).float()
        self.initialize()
        best_loss = torch.full((self.batch_size,), float("inf"), device=self.device)
        best_params = [torch.zeros_like(x) for x in self.de_parameterize()]
        tail_angle_sin = torch.zeros(
            self.batch_size, self.asg_num, 1, device=self.device
        )
        tail_angle_cos = torch.zeros(
            self.batch_size, self.asg_num, 1, device=self.device
        )
        tail_lamb = torch.zeros(self.batch_size, self.asg_num, 1, device=self.device)
        tail_mu = torch.zeros(self.batch_size, self.asg_num, 1, device=self.device)
        tail_weight = torch.zeros(self.batch_size, self.asg_num, 3, device=self.device)
        tail_count = 0

        for idx in range(self.steps):
            self.opt.zero_grad(set_to_none=True)
            params = self.de_parameterize()
            pano = self.render_asg(*params)
            loss_item = (
                F.l1_loss(pano, envmap, reduction="none")
                .reshape(pano.shape[0], -1)
                .mean(1)
            )
            loss = loss_item.sum()
            loss.backward()
            self.opt.step()

            clean_params = [x.detach() for x in self.de_parameterize()]
            best_params, best_loss = update_best_params(
                best_params, best_loss, clean_params, loss_item.detach()
            )
            if self.avg_steps > 0 and idx + self.avg_steps >= self.steps:
                tail_angle_sin = tail_angle_sin + torch.sin(clean_params[0])
                tail_angle_cos = tail_angle_cos + torch.cos(clean_params[0])
                tail_lamb = tail_lamb + clean_params[1]
                tail_mu = tail_mu + clean_params[2]
                tail_weight = tail_weight + clean_params[3]
                tail_count += 1

        valid = torch.isfinite(best_loss)
        if not valid.any():
            return None

        avg_params = best_params
        if tail_count:
            angle = torch.atan2(
                tail_angle_sin / tail_count, tail_angle_cos / tail_count
            )
            avg_params = (
                torch.remainder(angle, 2 * torch.pi),
                tail_lamb / tail_count,
                tail_mu / tail_count,
                tail_weight / tail_count,
            )
            avg_loss = (
                F.l1_loss(self.render_asg(*avg_params), envmap, reduction="none")
                .reshape(envmap.shape[0], -1)
                .mean(1)
            )
            avg_params = select_average_params(
                avg_params, best_params, avg_loss, best_loss
            )
        rec_map = (
            self.render_asg(*avg_params)
            .permute(0, 2, 3, 1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        save_npy = (
            torch.cat(avg_params, dim=-1).detach().cpu().numpy().astype(np.float32)
        )
        return np.clip(rec_map, 0, 1), save_npy, valid.cpu().numpy()


class ASGFitModule(L.LightningModule):
    def __init__(
        self,
        save_path,
        image_path,
        width,
        output_width,
        asg_num,
        steps,
        lr,
        avg_steps,
        rotate_count,
    ):
        super().__init__()
        self.save_path = save_path
        self.image_path = image_path
        self.width = width
        self.output_width = output_width
        self.asg_num = asg_num
        self.steps = steps
        self.lr = lr
        self.avg_steps = avg_steps
        self.rotate_count = rotate_count
        fib_path = (
            Path(__file__).resolve().parents[1]
            / "lighting_est"
            / f"fib_lobes_{asg_num}.npy"
        )
        normals = np.load(fib_path).astype(np.float32) + TINY
        self.normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)

    def on_predict_start(self):
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        Path(self.image_path).mkdir(parents=True, exist_ok=True)

    def render_output(self, save_npys):
        batch_size = save_npys.shape[0]
        optim = ASGEnvOptim(
            1,
            self.output_width,
            self.asg_num,
            batch_size,
            self.lr,
            self.avg_steps,
            self.device,
        )
        params = torch.from_numpy(save_npys).to(self.device)
        return optim.render_asg(*torch.split(params, [1, 1, 1, 3], dim=-1)).permute(
            0, 2, 3, 1
        )

    def predict_step(self, batch, batch_idx):
        envmaps, names = batch
        with torch.enable_grad():
            optim = ASGEnvOptim(
                self.steps,
                self.width,
                self.asg_num,
                envmaps.shape[0],
                self.lr,
                self.avg_steps,
                self.device,
            )
            result = optim.optimize(envmaps)
            if result is None:
                return []
            _, save_npys, valid = result
            for save_npy, name, is_valid in zip(save_npys, names, valid):
                if not is_valid:
                    continue
                if save_npy.shape != (self.asg_num, 6):
                    raise ValueError(
                        f"invalid ASG parameter shape for {name}: {save_npy.shape}"
                    )
                names = rotation_output_names(name, self.rotate_count)
                params = np.stack(
                    [
                        rotate_asg(save_npy, self.normals, index, self.rotate_count)
                        for index in range(self.rotate_count)
                    ]
                ).astype(np.float32)
                if not np.isfinite(params).all():
                    raise ValueError(f"non-finite ASG parameters for {name}")
                images = (
                    self.render_output(params).detach().cpu().numpy().astype(np.float32)
                )
                for output_name, output_params, image in zip(names, params, images):
                    np.save(Path(self.save_path, output_name + ".npy"), output_params)
                    image = np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)
                    write_image(Path(self.image_path, output_name + IMAGE_EXT), image)
        return []


@click.command()
@click.option("--dataset-root", required=True)
@click.option("--input-dir", default="pano_ldr_1024")
@click.option("--output-dir", default="asg_npy")
@click.option("--image-dir", default="asg_png")
@click.option("--width", type=int, default=256)
@click.option("--output_width", type=int, default=512)
@click.option("--batch_size", type=int, default=16)
@click.option("--num_workers", type=int, default=4)
@click.option("--steps", type=int, default=500)
@click.option("--lr", type=float, default=2e-2)
@click.option("--asg_num", type=int, default=128)
@click.option("--avg_steps", type=int, default=10)
@click.option("--rotate-count", default=1)
@click.option("--max_count", type=int, default=None)
@click.option("--skip_existing/--overwrite", default=True)
def main(
    dataset_root,
    input_dir,
    output_dir,
    image_dir,
    width,
    output_width,
    batch_size,
    num_workers,
    steps,
    lr,
    asg_num,
    avg_steps,
    rotate_count,
    max_count,
    skip_existing,
):
    if not 1 <= rotate_count <= 100:
        raise click.UsageError("--rotate-count must be in [1, 100]")
    root = Path(dataset_root).expanduser()
    input_path = root / input_dir
    save_path = root / output_dir
    image_path = root / image_dir
    datamodule = ASGFitDataModule(
        input_path,
        save_path,
        image_path,
        width,
        batch_size,
        num_workers,
        rotate_count,
        skip_existing,
        max_count,
    )
    model = ASGFitModule(
        save_path,
        image_path,
        width,
        output_width,
        asg_num,
        steps,
        lr,
        avg_steps,
        rotate_count,
    )
    trainer = L.Trainer(
        logger=False,
        enable_checkpointing=False,
        callbacks=[TQDMProgressBar()],
        inference_mode=False,
        use_distributed_sampler=False,
    )
    trainer.predict(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
