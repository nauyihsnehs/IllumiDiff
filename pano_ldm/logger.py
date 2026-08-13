import os
from pathlib import Path

import numpy as np
import torch
import torchvision
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from PIL import Image


def image_names(batch):
    if not isinstance(batch, dict) or "img_name" not in batch:
        return None
    names = batch["img_name"]
    if isinstance(names, str):
        return [names]
    return list(names)


def comparison_index(key):
    if not key.startswith("comparison_"):
        return None
    suffix = key.rsplit("_", 1)[-1]
    return int(suffix) if suffix.isdigit() else None


class ImageLogger(Callback):
    def __init__(
        self,
        batch_frequency=2000,
        max_images=4,
        clamp=True,
        rescale=True,
        disabled=False,
        log_images_kwargs=None,
        image_format="jpg",
        log_first_step=False,
    ):
        super().__init__()
        image_format = image_format.lower()
        if image_format not in {"jpg", "jpeg", "png"}:
            raise ValueError(f"unsupported image format: {image_format}")
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.clamp = clamp
        self.disabled = disabled
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.image_format = image_format
        self.log_first_step = log_first_step
        self.first_step_logged = False
        self.last_step = None

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, names=None):
        root = os.path.join(save_dir, "image_log", split)
        for key in images:
            grid = torchvision.utils.make_grid(images[key], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = self.filename(key, global_step, names)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def filename(self, key, global_step, names=None):
        index = comparison_index(key)
        if names is not None and index is not None and index < len(names):
            stem = f"{Path(str(names[index])).stem}_gs-{global_step:06}"
        else:
            stem = f"{Path(str(key)).stem}_gs-{global_step:06}"
        return f"{stem}.{self.image_format}"

    def prepare_image(self, image):
        if not isinstance(image, torch.Tensor):
            return image
        if image.ndim == 4:
            image = image[: self.max_images]
        image = image.detach().cpu()
        if self.clamp:
            image = torch.clamp(image, -1.0, 1.0)
        return image

    @rank_zero_only
    def log_img(self, pl_module, batch, batch_idx, split="train", force=False):
        if not force and not self.check_frequency(pl_module.global_step):
            return
        if self.last_step == pl_module.global_step:
            return
        if not hasattr(pl_module, "log_images") or not callable(pl_module.log_images):
            return
        if self.max_images <= 0:
            return

        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        with torch.inference_mode():
            images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

        for key in images:
            images[key] = self.prepare_image(images[key])

        save_dir = pl_module.logger.log_dir
        self.log_local(
            save_dir, split, images, pl_module.global_step, image_names(batch)
        )
        self.last_step = pl_module.global_step

        if is_train:
            pl_module.train()
        if pl_module.device.type == "cuda":
            torch.cuda.empty_cache()

    def check_frequency(self, global_step):
        return (
            self.batch_freq > 0
            and global_step > 0
            and global_step % self.batch_freq == 0
        )

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if (
            not self.disabled
            and self.log_first_step
            and not self.first_step_logged
            and pl_module.global_step == 0
        ):
            self.log_img(pl_module, batch, batch_idx, split="train", force=True)
            self.first_step_logged = True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")


class WildImageLogger(ImageLogger):
    def __init__(
        self,
        batch=None,
        name=None,
        batch_frequency=2000,
        seed=3407,
        max_images=1,
        clamp=True,
        rescale=True,
        disabled=False,
        log_images_kwargs=None,
    ):
        super().__init__(
            batch_frequency=batch_frequency,
            max_images=max_images,
            clamp=clamp,
            rescale=rescale,
            disabled=disabled or batch is None,
            log_images_kwargs=log_images_kwargs,
        )
        self.batch = batch
        self.name = name
        self.seed = seed

    def names(self):
        if self.name is not None:
            return [self.name]
        return image_names(self.batch)

    @rank_zero_only
    def log_wild(self, pl_module):
        if self.disabled:
            return
        if self.last_step == pl_module.global_step:
            return
        if not hasattr(pl_module, "log_images") or not callable(pl_module.log_images):
            return

        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        save_dir = pl_module.logger.log_dir
        names = self.names()
        with torch.inference_mode(), torch.random.fork_rng():
            torch.manual_seed(self.seed)
            for index in range(len(self.batch["img_name"])):
                batch = {
                    key: value[index : index + 1]
                    if isinstance(value, (torch.Tensor, list))
                    else value
                    for key, value in self.batch.items()
                }
                images = pl_module.log_images(
                    batch, split="wild", **self.log_images_kwargs
                )
                for key in images:
                    images[key] = self.prepare_image(images[key])
                item_names = names[index : index + 1] if names is not None else None
                self.log_local(
                    save_dir,
                    "wild",
                    images,
                    pl_module.global_step,
                    item_names,
                )
                del images
                if pl_module.device.type == "cuda":
                    torch.cuda.empty_cache()

        self.last_step = pl_module.global_step

        if is_train:
            pl_module.train()

    def on_train_start(self, trainer, pl_module):
        self.log_wild(pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.check_frequency(pl_module.global_step):
            self.log_wild(pl_module)

    def on_fit_end(self, trainer, pl_module):
        self.log_wild(pl_module)
