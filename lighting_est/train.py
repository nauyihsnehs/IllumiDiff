from pathlib import Path

import click
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import default_collate

from utils import (
    load_config,
    resolve_path,
    save_config_snapshot,
    training_batch_size,
)
from lighting_est.dataset import (
    ASGNetDataModule,
    HDRNetDataModule,
    HDRNetPredictDataset,
    IDNetDataModule,
    SGNetDataModule,
    color_tensor,
    mask_tensor,
)
from lighting_est.logger import (
    LightingImageLogger,
    LightingWildLogger,
)
from lighting_est.modules import ASGNetModule, HDRNetModule, IDNetModule, SGNetModule

TASKS = {
    "id_net": (IDNetDataModule, IDNetModule),
    "sg_net": (SGNetDataModule, SGNetModule),
    "asg_net": (ASGNetDataModule, ASGNetModule),
    "hdr_net": (HDRNetDataModule, HDRNetModule),
}


def create_dataset(config, batch_size):
    params = dict(config["dataset"])
    if config["task"] == "hdr_net":
        params["max_hdr"] = config["model"]["max_hdr"]
        if "curriculum" in config:
            params.update(config["curriculum"])
    params.update(config["dataloader"])
    params["batch_size"] = batch_size
    return TASKS[config["task"]][0](**params)


def validate_curriculum(config, config_file):
    curriculum = config.get("curriculum")
    if curriculum is None:
        return
    if config["task"] != "hdr_net":
        raise ValueError("curriculum is only supported for hdr_net")

    epoch_keys = ("low_epochs", "medium_epochs", "hard_epochs")
    for key in epoch_keys:
        value = curriculum.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"curriculum.{key} must be a positive integer")
    max_epochs = config.get("trainer", {}).get("max_epochs")
    total_epochs = sum(curriculum[key] for key in epoch_keys)
    if max_epochs != total_epochs:
        raise ValueError(
            f"trainer.max_epochs must equal curriculum epochs ({total_epochs})"
        )

    for key in ("medium_train_list_path", "hard_train_list_path"):
        curriculum[key] = resolve_path(
            curriculum[key], [config_file.parent, Path.cwd()]
        ).as_posix()
    config["trainer"]["reload_dataloaders_every_n_epochs"] = 1


def create_model(config, output_path):
    params = dict(config["model"])
    return TASKS[config["task"]][1](**params, img_log_dir=output_path)


def create_checkpoints(config, directory):
    common = {
        "dirpath": directory,
        "auto_insert_metric_name": False,
        "save_last": False,
    }
    periodic = ModelCheckpoint(
        **common,
        filename="step-{step:08}-val_loss-{val_loss:.6f}",
        every_n_train_steps=config["every_n_train_steps"],
        save_top_k=-1,
    )
    best = ModelCheckpoint(
        **common,
        filename="best-step-{step:08}-val_loss-{val_loss:.6f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_on_train_epoch_end=False,
    )
    return periodic, best


def create_wild_batch(config):
    wild = config.get("wild", {})
    task = config["task"]
    if task == "hdr_net":
        wild_path = wild.get("wild_path")
        if not wild_path:
            return None
        wild_path = Path(wild_path).expanduser()
        dataset = HDRNetPredictDataset(
            wild_path / "pano_ldr_512",
            wild_path / "sg_npy",
            wild_path / "pano_ls_512",
            config["dataset"]["resolution"],
        )
        return default_collate([dataset[index] for index in range(len(dataset))])

    input_path = wild.get("input_path")
    if not input_path:
        return None

    resolution = config["dataset"]["resolution"]
    name = [wild.get("name") or Path(input_path).stem]
    if task == "id_net":
        return color_tensor(input_path, resolution)[None], name
    if task == "asg_net":
        return color_tensor(input_path, (256, 256))[None], name
    if task == "sg_net":
        return (
            color_tensor(input_path, (256, 256), signed=True)[None],
            mask_tensor(wild["mask_path"], (256, 256))[None],
            name,
        )


def train_from_config(
    config_file,
    resume_ckpt=None,
    train_list=None,
    max_steps=None,
):
    config_file = resolve_path(
        config_file,
        [Path.cwd(), Path(__file__).parent, Path(__file__).parent / "configs"],
    )
    config = load_config(config_file)
    if train_list is not None:
        config["dataset"]["train_list_path"] = train_list
        config.pop("curriculum", None)
    if max_steps is not None:
        config.setdefault("trainer", {})["max_steps"] = max_steps
    if resume_ckpt is not None:
        config["resume_ckpt"] = resume_ckpt
    validate_curriculum(config, config_file)

    batch_size, accumulate = training_batch_size(config["batch"])
    dataset = create_dataset(config, batch_size)
    log_root = Path(config["log_root"]).expanduser()
    tb_logger = TensorBoardLogger(save_dir=log_root.as_posix(), name=config["task"])
    tb_logger.log_hyperparams(config)
    save_config_snapshot(tb_logger.log_dir, config)
    model = create_model(config, tb_logger.log_dir)

    image_logger = LightingImageLogger(
        train_frequency=config["logging"]["image_frequency"]
    )
    wild_logger = LightingWildLogger(
        batch=create_wild_batch(config),
        batch_frequency=config["wild"]["frequency"],
        seed=config["wild"]["seed"],
    )
    checkpoints = create_checkpoints(
        config["checkpoint"],
        Path(tb_logger.log_dir, "checkpoints").as_posix(),
    )

    trainer_config = dict(config.get("trainer", {}))
    trainer_config.update(
        accumulate_grad_batches=accumulate,
        check_val_every_n_epoch=None,
        callbacks=[
            *checkpoints,
            image_logger,
            wild_logger,
        ],
        logger=tb_logger,
    )
    trainer = L.Trainer(**trainer_config)
    trainer.fit(model=model, datamodule=dataset, ckpt_path=resume_ckpt)


@click.command()
@click.option(
    "--config", "config_file", default="configs/sg_net.toml", show_default=True
)
@click.option("--ckpt", "resume_ckpt", default=None)
@click.option(
    "--train-list",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
)
@click.option("--max-steps", type=click.IntRange(min=1), default=None)
def train(config_file, resume_ckpt, train_list, max_steps):
    train_from_config(config_file, resume_ckpt, train_list, max_steps)


if __name__ == "__main__":
    train()
