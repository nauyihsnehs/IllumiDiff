import os
from pathlib import Path

import click
import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from PIL import Image
from torch.utils.data import DataLoader

from utils import (
    IMAGE_SUFFIXES,
    build_perspective_condition,
    cv,
    loader_config,
    load_config,
    load_model_weights,
    path_map,
    resolve_path,
    save_config_snapshot,
    training_batch_size,
)
from pano_ldm.dataset import (
    PanoConditionDataset,
    validate_res,
)
from pano_ldm.logger import (
    ImageLogger,
    WildImageLogger,
)
from pano_ldm.diffusion import create_model

PACKAGE_ROOT = Path(__file__).resolve().parent


def create_dataset(config, config_dir):
    data = config
    bases = [config_dir, PACKAGE_ROOT, PACKAGE_ROOT.parent]
    root = resolve_path(data["data_root"], bases)
    asg_image_path = data.get("asg_image_path")
    return PanoConditionDataset(
        input_root=root / data["input_path"],
        pano_root=root / data["pano_path"],
        sg_root=root / data["sg_path"],
        asg_image_root=root / asg_image_path if asg_image_path else None,
        sg_pre_root=root / data["sg_pre_path"],
        asg_pre_image_root=root / data["asg_pre_image_path"],
        use_pre_asg=data["use_pre_asg"],
        use_pre_sg=data.get("use_pre_sg", False),
        scene=data.get("scene", "all"),
        res=data["res"],
        projection_vfov=data["projection_vfov"],
    )


def load_rgb(path, size):
    with Image.open(path) as image:
        image = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
        return torch.from_numpy(np.array(image) / 255.0).float()


def create_wild_batch(config, config_dir):
    wild = config["wild_logger"]
    root = resolve_path(
        wild["root_path"],
        [config_dir, PACKAGE_ROOT, PACKAGE_ROOT.parent],
    )
    inputs = path_map(root / "pers_ldr_512", IMAGE_SUFFIXES)
    asg = path_map(root / "asg_net", {".npy"})
    sg = path_map(root / "sg_net", {".npy"})

    names = list(inputs)
    pano_size = validate_res(config["dataset"]["res"])
    perspectives = []
    known_masks = []
    asg_values = []
    sg_values = []
    for name in names:
        input_rgb = load_rgb(inputs[name], (256, 256)).numpy()
        perspective, known_mask = build_perspective_condition(
            input_rgb,
            pano_size,
            config["dataset"]["projection_vfov"],
        )
        asg_value = np.load(asg[name], allow_pickle=False)
        if (
            asg_value.dtype != np.float32
            or asg_value.ndim != 3
            or asg_value.shape[-1] != 3
        ):
            raise ValueError(
                f"invalid wild ASG {asg[name]}: "
                "expected float32 [H, W, 3], "
                f"got {asg_value.dtype} {asg_value.shape}"
            )
        if asg_value.shape[:2] != pano_size[::-1]:
            asg_value = cv.resize(
                asg_value,
                pano_size,
                interpolation=cv.INTER_LINEAR,
            )
        sg_value = np.load(sg[name], allow_pickle=False)
        if (
            sg_value.dtype != np.float32
            or sg_value.ndim != 2
            or sg_value.shape[-1] != 5
        ):
            raise ValueError(
                f"invalid wild SG {sg[name]}: expected [N, 5] float32, "
                f"got {sg_value.dtype} {sg_value.shape}"
            )
        perspectives.append(torch.from_numpy(perspective * 2 - 1).float())
        known_masks.append(torch.from_numpy(known_mask).float())
        asg_values.append(torch.from_numpy(asg_value * 2 - 1).float())
        sg_values.append(torch.from_numpy(sg_value))

    count = len(names)
    return {
        "perspective": torch.stack(perspectives),
        "known_mask": torch.stack(known_masks),
        "asg": torch.stack(asg_values),
        "sg": torch.stack(sg_values),
        "sg_shift": torch.zeros(count, dtype=torch.long),
        "sg_flip": torch.zeros(count, dtype=torch.bool),
        "img_name": names,
    }


@click.command()
@click.option(
    "--config",
    "config_file",
    default="pano_ldm/configs/train.toml",
    show_default=True,
)
@click.option("--ckpt", "resume_ckpt", default=None)
@click.option("--init-ckpt", default=None)
@click.option(
    "--init-ema",
    is_flag=True,
    help="Initialize diffusion weights from --init-ckpt EMA state.",
)
def main(config_file, resume_ckpt, init_ckpt, init_ema):
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True",
    )
    config_file = resolve_path(config_file, [Path.cwd()])
    config = load_config(config_file)
    config_dir = config_file.parent
    if resume_ckpt is not None and init_ckpt is not None:
        raise click.UsageError("--ckpt and --init-ckpt cannot be used together")
    if init_ema and init_ckpt is None:
        raise click.UsageError("--init-ema requires --init-ckpt")
    if resume_ckpt is not None:
        config["resume_ckpt"] = resume_ckpt
    if resume_ckpt is None and init_ckpt is None:
        raise click.UsageError(
            "Stage 2 training requires --init-ckpt for a new run or --ckpt to resume"
        )
    batch_size, accumulate = training_batch_size(config["batch"])
    model_config = resolve_path(
        config["model"]["config_path"], [config_dir, PACKAGE_ROOT, PACKAGE_ROOT.parent]
    )
    model = create_model(model_config.as_posix())
    if init_ckpt is not None:
        init_ckpt = resolve_path(init_ckpt, [Path.cwd(), config_dir])
        load_model_weights(model, init_ckpt, ema=init_ema)
        config["init_ckpt"] = init_ckpt.as_posix()
        config["init_ema"] = init_ema
        source = "EMA" if init_ema else "online"
        print(f"Initialized VQ-inpainting model from {source} [{init_ckpt}]")
    model.configure_training(config["model"])

    dataset = create_dataset(config["dataset"], config_dir)
    dataloader = DataLoader(
        dataset,
        **loader_config(config["dataloader"], batch_size),
    )

    logging_config = config["logging"]
    tb_logger = TensorBoardLogger(
        save_dir=logging_config["save_dir"],
        name=logging_config.get("name", PACKAGE_ROOT.name),
    )
    tb_logger.log_hyperparams(config)
    save_config_snapshot(tb_logger.log_dir, config)
    image_logger = ImageLogger(**config["image_logger"])
    wild_batch = create_wild_batch(config, config_dir)
    wild_config = config["wild_logger"]
    wild_logger = WildImageLogger(
        batch=wild_batch,
        batch_frequency=wild_config["batch_frequency"],
        seed=wild_config["seed"],
        max_images=len(wild_batch["img_name"]),
        log_images_kwargs={
            "ddim_steps": wild_config["ddim_steps"],
            "N": len(wild_batch["img_name"]),
        },
    )
    model_ckpt = ModelCheckpoint(**config["checkpoint"])
    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer_config = dict(config["trainer"])
    trainer = L.Trainer(
        **trainer_config,
        accumulate_grad_batches=accumulate,
        callbacks=[
            model_ckpt,
            lr_monitor,
            image_logger,
            wild_logger,
        ],
        logger=tb_logger,
    )
    trainer.fit(model, dataloader, ckpt_path=resume_ckpt)


if __name__ == "__main__":
    main()
