import random
from pathlib import Path

import click
import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from torch.utils.data import DataLoader, Dataset

from utils import (
    load_checkpoint,
    load_config,
    load_sg_luminance,
    loader_config,
    resolve_path,
    save_config_snapshot,
    stage2_sg_from_params,
    training_batch_size,
)
from pano_ldm.autoencoder import AutoencoderKL
from pano_ldm.logger import ImageLogger

TRANSFORM = "log1p(lum)*2-1"
RUN_NAME = "vae_1ch_sg"
VAE_PREFIX = "first_stage_model."
PURE_VAE_PREFIXES = (
    "encoder.",
    "decoder.",
    "quantize.",
    "quant_conv.",
    "post_quant_conv.",
)


def list_sg_params(root_path):
    image_paths = [
        p.as_posix()
        for p in Path(root_path).iterdir()
        if p.is_file() and p.suffix.lower() == ".npy"
    ]
    image_paths.sort()
    return image_paths


def print_list(title, values, limit=20):
    print(f"{title}: {len(values)}")
    for value in values[:limit]:
        print(f"  {value}")
    if len(values) > limit:
        print(f"  ... {len(values) - limit} more")


def strip_vae_prefix(key):
    if key.startswith(VAE_PREFIX):
        return key[len(VAE_PREFIX) :]
    if key.startswith(PURE_VAE_PREFIXES):
        return key
    return None


def create_vae():
    config = {
        "z_channels": 4,
        "embed_dim": 4,
        "resolution": 256,
        "ch": 128,
        "ch_mult": [1, 2, 4, 4],
        "num_res_blocks": 2,
        "dropout": 0.0,
    }
    return AutoencoderKL(config, channels=1)


def adapt_rgb_key(key, value, target):
    if (
        key == "encoder.conv_in.weight"
        and value.ndim == 4
        and value.shape[1] == 3
        and target.shape[1] == 1
    ):
        lum_weight = torch.as_tensor(
            [0.2126, 0.7152, 0.0722], dtype=value.dtype, device=value.device
        )
        return (value * lum_weight.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)

    if (
        key == "decoder.conv_out.weight"
        and value.ndim == 4
        and value.shape[0] == 3
        and target.shape[0] == 1
    ):
        lum_weight = torch.as_tensor(
            [0.2126, 0.7152, 0.0722], dtype=value.dtype, device=value.device
        )
        return (value * lum_weight.view(3, 1, 1, 1)).sum(dim=0, keepdim=True)

    if (
        key == "decoder.conv_out.bias"
        and value.ndim == 1
        and value.shape[0] == 3
        and target.shape[0] == 1
    ):
        lum_weight = torch.as_tensor(
            [0.2126, 0.7152, 0.0722], dtype=value.dtype, device=value.device
        )
        return (value * lum_weight).sum(dim=0, keepdim=True)

    return None


def build_vae_init_state(vae, ckpt_path):
    _, state_dict = load_checkpoint(ckpt_path)
    model_state = vae.state_dict()
    init_state = {}
    copied = []
    adapted = []
    unexpected = []
    shape_mismatch = []

    for ckpt_key, value in state_dict.items():
        key = strip_vae_prefix(ckpt_key)
        if key is None:
            continue
        if key not in model_state:
            unexpected.append(key)
            continue

        adapted_value = adapt_rgb_key(key, value, model_state[key])
        if adapted_value is not None:
            init_state[key] = adapted_value
            adapted.append(key)
            continue

        if model_state[key].shape == value.shape:
            init_state[key] = value
            copied.append(key)
        else:
            shape_mismatch.append(
                (key, tuple(value.shape), tuple(model_state[key].shape))
            )

    missing = [key for key in model_state if key not in init_state]
    return init_state, copied, adapted, missing, unexpected, shape_mismatch


def load_kl_f8_weights(vae, ckpt_path):
    init_state, copied, adapted, missing, unexpected, shape_mismatch = (
        build_vae_init_state(vae, ckpt_path)
    )

    print(f"copied VAE keys: {len(copied)}")
    print_list("adapted VAE keys", adapted)
    print_list("missing VAE keys", missing)
    print_list("unexpected VAE keys", unexpected)
    print_list("shape mismatched VAE keys", shape_mismatch)

    if not init_state:
        raise RuntimeError(f"No usable VAE weights found in [{ckpt_path}]")

    load_missing, load_unexpected = vae.load_state_dict(init_state, strict=False)
    print(f"Loaded 1ch VAE init weights from [{ckpt_path}]")
    print(f"load_state_dict missing keys: {len(load_missing)}")
    print(f"load_state_dict unexpected keys: {len(load_unexpected)}")


def export_vae(model, save_path):
    state_dict = {}
    for key, value in model.vae.state_dict().items():
        state_dict[key] = value.detach().cpu()

    payload = {
        "state_dict": state_dict,
        "metadata": {
            "channels": 1,
            "transform": TRANSFORM,
            "source": "pano_ldm/train_vae_1ch.py",
        },
    }
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, save_path)
    print(f"saved pure 1ch VAE checkpoint: {save_path}")


class SGLumDataset(Dataset):
    def __init__(self, sg_root, size=512, max_count=None, rotate=False, flip=False):
        super().__init__()
        self.W = size
        self.H = size // 2
        self.rotate = rotate
        self.flip = flip
        image_paths = list_sg_params(sg_root)
        if max_count:
            image_paths = image_paths[:max_count]
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def get_augmentation(self):
        shift = random.randrange(self.W) if self.rotate else 0
        do_flip = self.flip and random.random() < 0.5
        return shift, do_flip

    def __getitem__(self, i):
        path = self.image_paths[i]
        shift, do_flip = self.get_augmentation()
        return {
            "sg": load_sg_luminance(path),
            "sg_shift": shift,
            "sg_flip": do_flip,
            "img_name": Path(path).stem,
        }


class OneChannelVAEModule(L.LightningModule):
    def __init__(self, size=512, learning_rate=1e-5, kl_weight=1e-6):
        super().__init__()
        self.vae = create_vae()
        self.size = size
        self.learning_rate = learning_rate
        self.kl_weight = kl_weight

    def get_input(self, batch):
        x = stage2_sg_from_params(
            batch["sg"].to(self.device).float(),
            self.size,
            self.size // 2,
            batch.get("sg_shift"),
            batch.get("sg_flip"),
        ).permute(0, 3, 1, 2)
        return x.to(memory_format=torch.contiguous_format).float()

    def forward(self, x, sample_posterior=True):
        return self.vae(x, sample_posterior=sample_posterior)

    def shared_step(self, batch, split):
        inputs = self.get_input(batch)
        reconstructions, posterior = self(inputs)
        l1_loss = F.l1_loss(reconstructions, inputs)
        mse_loss = F.mse_loss(reconstructions, inputs)
        kl_loss = posterior.kl().mean()
        loss = l1_loss + 0.5 * mse_loss + self.kl_weight * kl_loss
        self.log(
            f"{split}/loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            f"{split}/l1",
            l1_loss,
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            f"{split}/mse",
            mse_loss,
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            f"{split}/kl",
            kl_loss,
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.vae.parameters(), lr=self.learning_rate, betas=(0.5, 0.9)
        )

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        inputs = self.get_input(batch).to(self.device)
        reconstructions, posterior = self(inputs, sample_posterior=False)
        return {
            "inputs": inputs,
            "reconstructions": reconstructions,
        }


@rank_zero_only
def export_rank_zero(model, save_path):
    export_vae(model, save_path)


@click.command()
@click.option(
    "--config",
    "config_file",
    default="pano_ldm/configs/train_vae.toml",
    show_default=True,
)
@click.option("--ckpt", "resume_ckpt", default=None)
def main(config_file, resume_ckpt):
    config_file = resolve_path(
        config_file,
        [Path.cwd(), Path(__file__).parent, Path(__file__).parent / "configs"],
    )
    config = load_config(config_file)
    if resume_ckpt is not None:
        config["resume_ckpt"] = resume_ckpt

    batch_size, accumulate = training_batch_size(config["batch"])
    model_config = config["model"]
    model = OneChannelVAEModule(
        size=config["dataset"]["size"],
        learning_rate=model_config["learning_rate"],
        kl_weight=model_config["kl_weight"],
    )
    if resume_ckpt is None:
        init_ckpt = resolve_path(
            model_config["init_ckpt"],
            [config_file.parent, Path(__file__).parent, Path.cwd()],
        )
        load_kl_f8_weights(model.vae, init_ckpt.as_posix())

    dataset_config = config["dataset"]
    data_root = resolve_path(
        dataset_config["data_root"],
        [config_file.parent, Path(__file__).parent, Path.cwd()],
    )
    dataset = SGLumDataset(
        sg_root=data_root / dataset_config["sg_path"],
        size=dataset_config["size"],
        max_count=dataset_config.get("max_count"),
        rotate=dataset_config["rotate"],
        flip=dataset_config["flip"],
    )
    dataloader = DataLoader(
        dataset,
        **loader_config(config["dataloader"], batch_size),
    )
    logging_config = config["logging"]
    tb_logger = TensorBoardLogger(save_dir=logging_config["save_dir"], name=RUN_NAME)
    tb_logger.log_hyperparams(config)
    save_config_snapshot(tb_logger.log_dir, config)
    image_logger = ImageLogger(
        batch_frequency=logging_config["image_frequency"],
        max_images=logging_config["max_images"],
    )
    checkpoint_config = dict(config["checkpoint"])
    checkpoint_config["dirpath"] = Path(tb_logger.log_dir, "checkpoints").as_posix()
    checkpoint = ModelCheckpoint(**checkpoint_config)
    trainer_config = dict(config["trainer"])
    trainer_config["accumulate_grad_batches"] = accumulate
    trainer_config["callbacks"] = [
        checkpoint,
        image_logger,
    ]
    trainer_config["logger"] = tb_logger
    trainer = L.Trainer(**trainer_config)
    trainer.fit(model, dataloader, ckpt_path=resume_ckpt)
    export_path = resolve_path(
        config["export"]["path"], [config_file.parent, Path.cwd()]
    )
    export_rank_zero(model, export_path.as_posix())


if __name__ == "__main__":
    main()
