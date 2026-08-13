from pathlib import Path

import click
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from utils import (
    EXR_SAVE_PARAMS,
    LUM_WEIGHT_BGR,
    build_perspective_condition,
    chunks,
    inference_paths,
    load_inference_inputs,
    load_lightning_module,
    load_model_weights,
    max_luminance,
    render_sg_luminance,
    srgb_to_linear,
    to_signed,
    write_image,
)
from lighting_est.modules import (
    ASGNetModule,
    HDRNetModule,
    IDNetModule,
    SGNetModule,
    stretch_for_hdrnet,
)
from pano_ldm.diffusion import create_model

PERSPECTIVE_SIZE = (256, 256)
LDM_RES = (256, 256)
HDR_RES = (512, 256)
INFERENCE_MAX_HDR = 3e4
DEFAULT_LDM_CONFIG = "pano_ldm/configs/pano_ldm.toml"
DEFAULT_CHECKPOINTS = {
    "id": "ckpts/id_net-step020k.ckpt",
    "sg": "ckpts/sg_net-step080k.ckpt",
    "asg": "ckpts/asg_net-step100k.ckpt",
    "hdr": "ckpts/hdr_net-step038k.ckpt",
    "ldm": "ckpts/ldm-step015k.ckpt",
}
DDIM_ETA = 0.0
DEVICE = torch.device("cuda:0")


def output_dirs(output_path, save_complete):
    root = Path(output_path)
    root.mkdir(parents=True, exist_ok=True)
    if not save_complete:
        return {"hdr": root}
    directories = {
        "id_stage1": root / "id_net" / "stage1",
        "sg": root / "sg_net",
        "asg": root / "asg_net",
        "ldr": root / "pano_ldm",
        "id_stage3": root / "id_net" / "stage3",
        "hdr": root / "hdr_net",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def load_ldm(checkpoint):
    model = create_model(DEFAULT_LDM_CONFIG)
    load_model_weights(model, checkpoint)
    return model.to(DEVICE).eval()


def load_models(checkpoints):
    return {
        "id": load_lightning_module(IDNetModule, checkpoints["id"], DEVICE),
        "sg": load_lightning_module(
            SGNetModule,
            checkpoints["sg"],
            DEVICE,
            resolution=LDM_RES,
            backbone_weights=None,
        ),
        "asg": load_lightning_module(
            ASGNetModule,
            checkpoints["asg"],
            DEVICE,
            resolution=LDM_RES,
            backbone_weights=None,
        ),
        "hdr": load_lightning_module(
            HDRNetModule,
            checkpoints["hdr"],
            DEVICE,
            max_hdr=INFERENCE_MAX_HDR,
        ),
        "ldm": load_ldm(checkpoints["ldm"]),
    }


def build_conditions(input_rgb, asg_rgb, fov):
    perspectives = []
    known_masks = []
    for image in input_rgb.detach().cpu():
        image = image.permute(1, 2, 0).numpy()
        perspective, known_mask = build_perspective_condition(
            image,
            LDM_RES,
            fov,
        )
        perspectives.append(torch.from_numpy(perspective * 2 - 1))
        known_masks.append(torch.from_numpy(known_mask).float())
    return {
        "perspective": torch.stack(perspectives),
        "known_mask": torch.stack(known_masks),
        "asg": to_signed(asg_rgb).permute(0, 2, 3, 1),
    }


@torch.inference_mode()
def run_stage1(models, input_bgr):
    luminance = models["id"].model(input_bgr)
    direction, lamb, weight = models["sg"].model(to_signed(input_bgr), luminance)
    sg_params = torch.cat((direction, lamb, weight), dim=-1)
    angle, asg_lamb, mu, asg_weight = models["asg"].model(input_bgr)
    asg_bgr = models["asg"].asg_viewer((angle, asg_lamb, mu, asg_weight))
    return luminance, sg_params, asg_bgr.clamp(0, 1)


@torch.inference_mode()
def run_ldm(
    model,
    input_bgr,
    asg_bgr,
    sg_params,
    fov,
    ddim_steps,
    seeds,
):
    condition = build_conditions(
        input_bgr.flip(1),
        asg_bgr.flip(1),
        fov,
    )
    batch_size = input_bgr.shape[0]
    batch = dict(condition)
    batch.update(
        {
            "sg": sg_params,
            "sg_shift": torch.zeros(
                batch_size, device=input_bgr.device, dtype=torch.long
            ),
            "sg_flip": torch.zeros(
                batch_size, device=input_bgr.device, dtype=torch.bool
            ),
        }
    )
    samples = model.sample_batch(
        batch,
        ddim_steps=ddim_steps,
        eta=DDIM_ETA,
        seeds=seeds,
    )
    return samples.add(1).mul(0.5).clamp(0, 1)


@torch.inference_mode()
def run_stage3(models, ldr_rgb, sg_params, sg_scale):
    ldr_bgr = ldr_rgb.flip(1)
    luminance = models["id"].model(ldr_bgr)
    ldr_signed = to_signed(ldr_bgr)
    sg_image = models["hdr"].sg_image(sg_params, ldr_signed, scale=sg_scale)
    hdr_log = models["hdr"].model(
        ldr_signed,
        sg_image,
        to_signed(luminance),
    )
    return luminance, models["hdr"].linear_hdr(hdr_log)


def prepare_hdrnet_ldr(ldr_rgb):
    ldr_rgb = stretch_for_hdrnet(ldr_rgb, HDR_RES).clamp(0, 1)
    return torch.round(ldr_rgb * 255) / 255


def select_candidates(
    input_bgr,
    ldr_candidates,
    luminance_candidates,
    hdr_candidates,
):
    clean_hdr = torch.nan_to_num(
        hdr_candidates,
        nan=0,
        posinf=INFERENCE_MAX_HDR,
        neginf=0,
    ).clamp_min(0)
    weight = hdr_candidates.new_tensor(LUM_WEIGHT_BGR).view(1, 1, 3, 1, 1)
    luminance = (clean_hdr * weight).sum(dim=2).flatten(2).clamp_min(1e-8)
    median = torch.quantile(luminance, 0.5, dim=-1)
    highlight = torch.quantile(luminance, 0.999, dim=-1)
    dynamic_range = torch.log2(highlight / median.clamp_min(1e-8))

    input_linear = srgb_to_linear(input_bgr)
    input_tone = input_linear.mean(dim=(-2, -1))
    input_tone = input_tone / input_tone.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    hdr_tone = clean_hdr.mean(dim=(-2, -1))
    hdr_tone = hdr_tone / hdr_tone.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    tone_difference = torch.linalg.vector_norm(
        hdr_tone - input_tone.unsqueeze(0),
        dim=-1,
    )
    dynamic_rank = (dynamic_range[:, None, :] < dynamic_range[None, :, :]).sum(dim=1)
    tone_rank = (tone_difference[:, None, :] > tone_difference[None, :, :]).sum(dim=1)
    best = (dynamic_rank + tone_rank).argmin(dim=0)
    batch = torch.arange(best.shape[0], device=best.device)
    return (
        ldr_candidates[best, batch],
        luminance_candidates[best, batch],
        clean_hdr[best, batch],
    )


def save_luminance(values, names, directory):
    values = values.detach().cpu().permute(0, 2, 3, 1).numpy()
    for value, name in zip(values, names):
        image = np.round(np.clip(value, 0, 1) * 255).astype(np.uint8)
        write_image(directory / f"{name}.png", image)


def save_stage1(luminance, sg_params, asg_bgr, names, directories):
    save_luminance(luminance, names, directories["id_stage1"])
    sg_values = sg_params.detach().cpu().numpy().astype(np.float32)
    sg_images = render_sg_luminance(sg_params, *HDR_RES).detach().cpu().numpy()
    asg_bgr = asg_bgr.detach().cpu()
    asg_rgb = asg_bgr.flip(1).permute(0, 2, 3, 1).numpy().astype(np.float32)
    asg_previews = stretch_for_hdrnet(asg_bgr, HDR_RES).permute(0, 2, 3, 1).numpy()

    for index, name in enumerate(names):
        np.save(directories["sg"] / f"{name}.npy", sg_values[index])
        sg_preview = np.clip(sg_images[index, ..., 0] ** (1 / 2.2), 0, 1)
        write_image(
            directories["sg"] / f"{name}.jpg",
            np.round(sg_preview * 255).astype(np.uint8),
        )
        np.save(directories["asg"] / f"{name}.npy", asg_rgb[index])
        write_image(
            directories["asg"] / f"{name}.jpg",
            np.round(np.clip(asg_previews[index], 0, 1) * 255).astype(np.uint8),
        )


def save_ldr(values, names, directory):
    values = values.detach().cpu().permute(0, 2, 3, 1).numpy()
    for value, name in zip(values, names):
        image = np.round(np.clip(value, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(image).save(directory / f"{name}.jpg", quality=95)


def save_hdr(luminance, hdr_bgr, names, directories, save_complete):
    if save_complete:
        save_luminance(luminance, names, directories["id_stage3"])
    values = hdr_bgr.detach().cpu().permute(0, 2, 3, 1).numpy()
    results = []
    for value, name in zip(values, names):
        path = write_image(
            directories["hdr"] / f"{name}.exr",
            value,
            EXR_SAVE_PARAMS,
        )
        results.append((path, max_luminance(value)))
    return results


@click.command()
@click.option("--input-path", default="test_imgs/inputs/mini")
@click.option("--output-path", default="test_imgs/outputs")
@click.option("--batch-size", default=8)
@click.option("--seed", default=114514)
@click.option("--id-ckpt", default=DEFAULT_CHECKPOINTS["id"])
@click.option("--sg-ckpt", default=DEFAULT_CHECKPOINTS["sg"])
@click.option("--asg-ckpt", default=DEFAULT_CHECKPOINTS["asg"])
@click.option("--hdr-ckpt", default=DEFAULT_CHECKPOINTS["hdr"])
@click.option("--ldm-ckpt", default=DEFAULT_CHECKPOINTS["ldm"])
@click.option("--ddim-steps", default=50)
@click.option("--fov", default=50)
@click.option("--sg-scale", default=1.0)
@click.option("--ldm-repeat", default=3)
@click.option("--verbose", is_flag=True, default=False)
@click.option("--save-complete", is_flag=True, default=False)
def main(
    input_path,
    output_path,
    batch_size,
    seed,
    id_ckpt,
    sg_ckpt,
    asg_ckpt,
    hdr_ckpt,
    ldm_ckpt,
    ddim_steps,
    fov,
    sg_scale,
    ldm_repeat,
    verbose,
    save_complete,
):
    paths = inference_paths(input_path)
    torch.manual_seed(seed)
    checkpoints = {
        "id": id_ckpt,
        "sg": sg_ckpt,
        "asg": asg_ckpt,
        "hdr": hdr_ckpt,
        "ldm": ldm_ckpt,
    }
    directories = output_dirs(output_path, save_complete)
    models = load_models(checkpoints)

    with tqdm(total=len(paths), desc="inference", unit="image") as progress:
        for batch_index, path_batch in enumerate(chunks(paths, batch_size)):
            names = [path.stem for path in path_batch]
            input_bgr = load_inference_inputs(path_batch, PERSPECTIVE_SIZE, DEVICE)
            luminance, sg_params, asg_bgr = run_stage1(models, input_bgr)
            if save_complete:
                save_stage1(luminance, sg_params, asg_bgr, names, directories)

            ldr_candidates = []
            luminance_candidates = []
            hdr_candidates = []
            batch_start = batch_index * batch_size
            for repeat_index in range(ldm_repeat):
                seeds = [
                    seed + repeat_index * len(paths) + batch_start + index
                    for index in range(len(path_batch))
                ]
                ldr_rgb = run_ldm(
                    models["ldm"],
                    input_bgr,
                    asg_bgr,
                    sg_params,
                    fov,
                    ddim_steps,
                    seeds,
                )
                ldr_rgb = prepare_hdrnet_ldr(ldr_rgb)
                pano_luminance, hdr_bgr = run_stage3(
                    models,
                    ldr_rgb,
                    sg_params,
                    sg_scale,
                )
                ldr_candidates.append(ldr_rgb)
                luminance_candidates.append(pano_luminance)
                hdr_candidates.append(hdr_bgr)

            selected = select_candidates(
                input_bgr,
                torch.stack(ldr_candidates),
                torch.stack(luminance_candidates),
                torch.stack(hdr_candidates),
            )
            ldr_rgb, pano_luminance, hdr_bgr = selected
            if save_complete:
                save_ldr(ldr_rgb, names, directories["ldr"])
            results = save_hdr(
                pano_luminance,
                hdr_bgr,
                names,
                directories,
                save_complete,
            )
            if verbose:
                for name, (path, maximum) in zip(names, results):
                    progress.write(
                        f"{name}: max_luminance={maximum:.6g}, output={path.resolve()}"
                    )
            progress.update(len(path_batch))


if __name__ == "__main__":
    main()
