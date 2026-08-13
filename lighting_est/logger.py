import random
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from utils import write_image


def extension(record):
    ext = record.get("ext", ".jpg")
    if ext.startswith("."):
        return ext
    return "." + ext


def to_numpy(image):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    return np.asarray(image)


def save_record(root, record, global_step):
    image = to_numpy(record["image"])
    name = Path(str(record.get("name", "image"))).stem or "image"
    ext = extension(record)
    path = Path(root, f"{name}_gs-{global_step:06}{ext}")
    write_image(path, image, record.get("params"))


def should_log(global_step, frequency):
    return frequency > 0 and global_step > 0 and global_step % frequency == 0


def save_records(root, records, global_step):
    for record in records:
        save_record(root, record, global_step)


def sample_batch(batch, index):
    return [
        value[index : index + 1].detach().clone()
        if isinstance(value, torch.Tensor)
        else value[index : index + 1]
        for value in batch
    ]


class LightingImageLogger(Callback):
    def __init__(self, train_frequency=500, disabled=False):
        super().__init__()
        self.train_frequency = train_frequency
        self.disabled = disabled
        self.val_batch_idx = None
        self.val_rng = None

    @rank_zero_only
    def log_batch(self, pl_module, batch, split):
        if self.disabled:
            return
        if not hasattr(pl_module, "log_images") or not callable(pl_module.log_images):
            return

        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        with torch.inference_mode():
            records = pl_module.log_images(batch, split=split)

        root = Path(pl_module.logger.log_dir, "image_log", split)
        save_records(root, records, pl_module.global_step)

        if is_train:
            pl_module.train()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if should_log(pl_module.global_step, self.train_frequency):
            self.log_batch(pl_module, batch, "train")

    def on_validation_epoch_start(self, trainer, pl_module):
        self.val_batch_idx = None
        self.val_rng = None
        if self.disabled or trainer.sanity_checking:
            return
        total_batches = trainer.num_val_batches[0]
        if total_batches:
            self.val_rng = random.Random(pl_module.global_step)
            self.val_batch_idx = self.val_rng.randrange(total_batches)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if (
            trainer.sanity_checking
            or dataloader_idx != 0
            or batch_idx != self.val_batch_idx
        ):
            return
        sample_index = self.val_rng.randrange(len(batch[-1]))
        self.log_batch(pl_module, sample_batch(batch, sample_index), "val")
        self.val_batch_idx = None


class LightingWildLogger(Callback):
    def __init__(self, batch=None, batch_frequency=500, seed=3407, disabled=False):
        super().__init__()
        self.batch = batch
        self.batch_frequency = batch_frequency
        self.seed = seed
        self.disabled = disabled or batch is None
        self.last_step = None

    def log_wild(self, pl_module):
        if self.disabled:
            return
        if self.last_step == pl_module.global_step:
            return
        if not hasattr(pl_module, "log_wild_images") or not callable(
            pl_module.log_wild_images
        ):
            return

        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        with torch.inference_mode(), torch.random.fork_rng():
            torch.manual_seed(self.seed)
            records = pl_module.log_wild_images(self.batch)

        save_records(
            Path(pl_module.logger.log_dir, "image_log", "wild"),
            records,
            pl_module.global_step,
        )
        self.last_step = pl_module.global_step

        if is_train:
            pl_module.train()

    def on_train_start(self, trainer, pl_module):
        self.log_wild(pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if should_log(pl_module.global_step, self.batch_frequency):
            self.log_wild(pl_module)

    def on_fit_end(self, trainer, pl_module):
        self.log_wild(pl_module)
