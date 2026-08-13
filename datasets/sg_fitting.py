from pathlib import Path

import click
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import TQDMProgressBar
from torch.utils.data import Dataset

from utils import (
    HDR_SUFFIXES,
    cv,
    distributed_items,
    image_paths,
    prediction_dataloader,
    read_image,
    render_sg_luminance,
    rotation_output_names,
    select_average_params,
    sg_luminance_params,
    update_best_params,
    write_image,
)


def rotate_sg(params, index, rotate_count):
    angle = 2 * np.pi * index / rotate_count
    cosine = np.cos(angle)
    sine = np.sin(angle)
    result = params.copy()
    x = params[..., 0]
    z = params[..., 2]
    result[..., 0] = cosine * x + sine * z
    result[..., 2] = -sine * x + cosine * z
    return result


class SGFitDataset(Dataset):
    def __init__(
        self,
        input_path,
        width,
        percentile,
        skip_existing,
        save_path,
        image_path,
        rotate_count,
        max_count,
    ):
        self.save_path = Path(save_path)
        self.image_path = Path(image_path)
        self.rotate_count = rotate_count
        all_paths = image_paths(input_path, HDR_SUFFIXES)
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
                    "missing SG rotation inputs:\n" + "\n".join(missing[:50])
                )
        if max_count is not None:
            img_list = img_list[:max_count]
        if skip_existing:
            img_list = [p for p in img_list if not self.output_exists(p)]
        self.img_list = distributed_items([path.as_posix() for path in img_list])
        self.width = width
        self.height = width // 2
        self.percentile = percentile
        self.lum_weight = np.asarray([0.0722, 0.7152, 0.2126])[None, None]

    def output_exists(self, img_path):
        names = rotation_output_names(Path(img_path).stem, self.rotate_count)
        return all(
            (self.save_path / f"{name}.npy").is_file()
            and (self.image_path / f"{name}.jpg").is_file()
            for name in names
        )

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img = read_image(img_path, cv.IMREAD_UNCHANGED)
        img = cv.resize(img, (self.width, self.height), interpolation=cv.INTER_AREA)
        img = np.nan_to_num(img, nan=0.0).astype(np.float32)
        lum = np.sum(img * self.lum_weight, -1, keepdims=True)
        lum_threshold = np.percentile(lum, self.percentile)
        img[lum[..., 0] < lum_threshold] = 0
        return torch.from_numpy(img).float(), Path(img_path).stem


class SGFitDataModule(L.LightningDataModule):
    def __init__(
        self,
        input_path,
        save_path,
        image_path,
        width,
        percentile,
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
        self.percentile = percentile
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.rotate_count = rotate_count
        self.skip_existing = skip_existing
        self.max_count = max_count
        self.predict_data = None

    def setup(self, stage=None):
        self.predict_data = SGFitDataset(
            self.input_path,
            self.width,
            self.percentile,
            self.skip_existing,
            self.save_path,
            self.image_path,
            self.rotate_count,
            self.max_count,
        )

    def predict_dataloader(self):
        return prediction_dataloader(
            self.predict_data,
            self.batch_size,
            self.num_workers,
        )


class SGEnvOptim:
    def __init__(self, steps, width, sg_num, batch_size, lr, avg_steps, device):
        self.steps = steps
        self.width = width
        self.height = width // 2
        self.sg_num = sg_num
        self.batch_size = batch_size
        self.avg_steps = avg_steps
        self.device = device
        self.ls = self.get_view_dirs().to(device).view(-1, 3)
        self.ls_t = self.ls.t()
        self.param = None
        self.opt = None

    def get_view_dirs(self):
        phi, theta = torch.meshgrid(
            [
                torch.linspace(0.0, torch.pi, self.height),
                torch.linspace(0.0, 2 * torch.pi, self.width),
            ],
            indexing="ij",
        )
        return torch.stack(
            [
                torch.cos(theta) * torch.sin(phi),
                torch.cos(phi),
                torch.sin(theta) * torch.sin(phi),
            ],
            dim=-1,
        ).float()

    def initialize(self, lr):
        p = torch.randn(self.batch_size, self.sg_num, 3, device=self.device)
        p = F.normalize(p, dim=-1)
        lamb = torch.randn(self.batch_size, self.sg_num, 1, device=self.device) * 20
        weight = torch.randn(self.batch_size, self.sg_num, 3, device=self.device)

        self.param = torch.cat([p, lamb, weight], dim=-1).detach()
        self.param.requires_grad = True
        self.opt = torch.optim.NAdam([self.param], lr=lr)

    def render_sg(self, position, lamb, weight):
        dot = torch.matmul(position, self.ls_t)
        e_item = torch.exp(lamb * (dot - 1.0))
        envmap = torch.einsum("bnp,bnc->bpc", e_item, weight)
        return envmap.reshape(self.batch_size, self.height, self.width, 3)

    def de_parameterize(self):
        p, lamb, weight = torch.split(
            self.param.view(self.batch_size, self.sg_num, 7), [3, 1, 3], dim=-1
        )
        p = F.normalize(p, dim=-1)
        lamb = torch.abs(lamb)
        weight = torch.abs(weight)
        return p, lamb, weight

    def optimize(self, envmap, lr):
        envmap = envmap.to(self.device).float()
        self.initialize(lr)
        best_loss = torch.full((self.batch_size,), float("inf"), device=self.device)
        best_params = [torch.zeros_like(x) for x in self.de_parameterize()]
        tail_p = torch.zeros(self.batch_size, self.sg_num, 3, device=self.device)
        tail_lamb = torch.zeros(self.batch_size, self.sg_num, 1, device=self.device)
        tail_weight = torch.zeros(self.batch_size, self.sg_num, 3, device=self.device)
        tail_count = 0

        for idx in range(self.steps):
            self.opt.zero_grad(set_to_none=True)
            params = self.de_parameterize()
            pano = self.render_sg(*params)
            loss_item = (
                F.mse_loss(pano, envmap, reduction="none")
                .reshape(self.batch_size, -1)
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
                tail_p = tail_p + clean_params[0]
                tail_lamb = tail_lamb + clean_params[1]
                tail_weight = tail_weight + clean_params[2]
                tail_count += 1

        valid = torch.isfinite(best_loss)
        if not valid.any():
            return None

        avg_params = best_params
        if tail_count:
            avg_params = (
                F.normalize(tail_p / tail_count, dim=-1),
                tail_lamb / tail_count,
                tail_weight / tail_count,
            )
            avg_loss = (
                F.mse_loss(self.render_sg(*avg_params), envmap, reduction="none")
                .reshape(self.batch_size, -1)
                .mean(1)
            )
            avg_params = select_average_params(
                avg_params, best_params, avg_loss, best_loss
            )
        save_npy = (
            torch.cat(avg_params, dim=-1).detach().cpu().numpy().astype(np.float32)
        )
        return save_npy, valid.cpu().numpy()


class SGFitModule(L.LightningModule):
    def __init__(
        self,
        save_path,
        image_path,
        width,
        output_width,
        sg_num,
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
        self.sg_num = sg_num
        self.steps = steps
        self.lr = lr
        self.avg_steps = avg_steps
        self.rotate_count = rotate_count

    def on_predict_start(self):
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        Path(self.image_path).mkdir(parents=True, exist_ok=True)

    def predict_step(self, batch, batch_idx):
        envmaps, names = batch
        with torch.enable_grad():
            optim = SGEnvOptim(
                self.steps,
                self.width,
                self.sg_num,
                envmaps.shape[0],
                self.lr,
                self.avg_steps,
                self.device,
            )
            result = optim.optimize(envmaps, self.lr)
            if result is None:
                return []
            save_npys, valid = result
            for save_npy, name, is_valid in zip(save_npys, names, valid):
                if not is_valid:
                    continue
                if save_npy.shape != (self.sg_num, 7):
                    raise ValueError(
                        f"invalid SG parameter shape for {name}: {save_npy.shape}"
                    )
                output_names_ = rotation_output_names(name, self.rotate_count)
                params = np.stack(
                    [
                        sg_luminance_params(
                            rotate_sg(save_npy, index, self.rotate_count)
                        )
                        for index in range(self.rotate_count)
                    ]
                ).astype(np.float32)
                if not np.isfinite(params).all():
                    raise ValueError(f"non-finite SG parameters for {name}")
                images = render_sg_luminance(
                    torch.from_numpy(params).to(self.device), self.output_width
                )
                images = images.detach().cpu().numpy()
                images = np.round(np.clip(images, 0, 1) ** (1 / 2.2) * 255).astype(
                    np.uint8
                )
                images = np.repeat(images, 3, axis=-1)
                for output_name, output_params, image in zip(
                    output_names_, params, images
                ):
                    np.save(Path(self.save_path, output_name + ".npy"), output_params)
                    write_image(Path(self.image_path, output_name + ".jpg"), image)
        return []


@click.command()
@click.option("--dataset-root", required=True)
@click.option("--input-dir", default="pano_hdr_1024")
@click.option("--output-dir", default="sg_npy")
@click.option("--image-dir", default="sg_jpg")
@click.option("--width", type=int, default=128)
@click.option("--output_width", type=int, default=512)
@click.option("--batch_size", type=int, default=16)
@click.option("--num_workers", type=int, default=4)
@click.option("--steps", type=int, default=10000)
@click.option("--lr", type=float, default=1e-1)
@click.option("--sg_num", type=int, default=12)
@click.option("--percentile", type=float, default=99)
@click.option("--avg_steps", type=int, default=100)
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
    sg_num,
    percentile,
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
    datamodule = SGFitDataModule(
        input_path,
        save_path,
        image_path,
        width,
        percentile,
        batch_size,
        num_workers,
        rotate_count,
        skip_existing,
        max_count,
    )
    model = SGFitModule(
        save_path,
        image_path,
        width,
        output_width,
        sg_num,
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
