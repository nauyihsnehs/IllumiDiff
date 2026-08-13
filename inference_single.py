from pathlib import Path

import click
import numpy as np
import torch
from tqdm import tqdm

from utils import (
    EXR_SAVE_PARAMS,
    chunks,
    inference_paths,
    load_inference_inputs,
    load_lightning_module,
    max_luminance,
    render_sg_luminance,
    to_signed,
    write_image,
)
from lighting_est.modules import (
    ASGNetModule,
    IDNetModule,
    SGNetModule,
)

PERSPECTIVE_SIZE = (256, 256)
DEFAULT_CHECKPOINTS = {
    "id": "ckpts/id_net-step020k.ckpt",
    "sg": "ckpts/sg_net-step080k.ckpt",
    "asg": "ckpts/asg_net-step100k.ckpt",
}
DEVICE = torch.device("cuda:0")


def validate_resolution(resolution):
    width, height = resolution
    if width != 2 * height:
        raise click.BadParameter("output resolution must have a 2:1 aspect ratio")
    return resolution


def load_models(checkpoints, resolution):
    return {
        "id": load_lightning_module(IDNetModule, checkpoints["id"], DEVICE),
        "sg": load_lightning_module(
            SGNetModule,
            checkpoints["sg"],
            DEVICE,
            resolution=resolution,
            backbone_weights=None,
        ),
        "asg": load_lightning_module(
            ASGNetModule,
            checkpoints["asg"],
            DEVICE,
            resolution=resolution,
            backbone_weights=None,
        ),
    }


@torch.inference_mode()
def render_lighting(models, input_bgr, resolution):
    luminance = models["id"].model(input_bgr)
    direction, lamb, weight = models["sg"].model(
        to_signed(input_bgr),
        luminance,
    )
    sg_params = torch.cat((direction, lamb, weight), dim=-1)
    sg_bhwc = render_sg_luminance(sg_params, *resolution)

    angle, asg_lamb, mu, asg_weight = models["asg"].model(input_bgr)
    asg_bhwc = (
        models["asg"].asg_viewer((angle, asg_lamb, mu, asg_weight)).permute(0, 2, 3, 1)
    )
    lighting = asg_bhwc + sg_bhwc.expand(-1, -1, -1, 3)
    return luminance, sg_params, sg_bhwc, asg_bhwc, lighting


def output_dirs(output_path, save_complete):
    root = Path(output_path)
    root.mkdir(parents=True, exist_ok=True)
    if not save_complete:
        return {"lighting": root}
    directories = {
        "lighting": root,
        "id": root / "id_net",
        "sg": root / "sg_net",
        "asg": root / "asg_net",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def save_complete_results(luminance, sg_params, sg_bhwc, asg_bhwc, names, directories):
    luminance = luminance.detach().cpu().permute(0, 2, 3, 1).numpy()
    sg_params = sg_params.detach().cpu().numpy().astype(np.float32)
    sg_bhwc = sg_bhwc.detach().cpu().numpy()
    asg_bhwc = asg_bhwc.detach().cpu().numpy()
    for index, name in enumerate(names):
        id_image = np.round(np.clip(luminance[index], 0, 1) * 255).astype(np.uint8)
        write_image(directories["id"] / f"{name}.png", id_image)
        np.save(directories["sg"] / f"{name}.npy", sg_params[index])
        sg_preview = np.clip(sg_bhwc[index, ..., 0] ** (1 / 2.2), 0, 1)
        write_image(
            directories["sg"] / f"{name}.jpg",
            np.round(sg_preview * 255).astype(np.uint8),
        )
        np.save(directories["asg"] / f"{name}.npy", asg_bhwc[index])
        write_image(
            directories["asg"] / f"{name}.jpg",
            np.round(np.clip(asg_bhwc[index], 0, 1) * 255).astype(np.uint8),
        )


def save_lighting(values, names, output_path):
    values = values.detach().cpu().numpy().astype(np.float32)
    results = []
    for value, name in zip(values, names):
        path = write_image(output_path / f"{name}.exr", value, EXR_SAVE_PARAMS)
        results.append((path, max_luminance(value)))
    return results


@click.command()
@click.option("--input-path", default="test_imgs/inputs/mini")
@click.option("--output-path", default="test_imgs/outputs_single")
@click.option("--batch-size", default=10)
@click.option("--id-ckpt", default=DEFAULT_CHECKPOINTS["id"])
@click.option("--sg-ckpt", default=DEFAULT_CHECKPOINTS["sg"])
@click.option("--asg-ckpt", default=DEFAULT_CHECKPOINTS["asg"])
@click.option("--output-res", type=(int, int), default=(512, 256))
@click.option("--verbose", is_flag=True, default=False)
@click.option("--save-complete", is_flag=True, default=False)
def main(
    input_path,
    output_path,
    batch_size,
    id_ckpt,
    sg_ckpt,
    asg_ckpt,
    output_res,
    verbose,
    save_complete,
):
    if batch_size < 1:
        raise click.BadParameter("batch-size must be at least 1")
    output_res = validate_resolution(output_res)
    paths = inference_paths(input_path)
    checkpoints = {"id": id_ckpt, "sg": sg_ckpt, "asg": asg_ckpt}
    models = load_models(checkpoints, output_res)
    directories = output_dirs(output_path, save_complete)

    with tqdm(total=len(paths), desc="inference", unit="image") as progress:
        for path_batch in chunks(paths, batch_size):
            names = [path.stem for path in path_batch]
            input_bgr = load_inference_inputs(path_batch, PERSPECTIVE_SIZE, DEVICE)
            luminance, sg_params, sg_bhwc, asg_bhwc, lighting = render_lighting(
                models,
                input_bgr,
                output_res,
            )
            if save_complete:
                save_complete_results(
                    luminance,
                    sg_params,
                    sg_bhwc,
                    asg_bhwc,
                    names,
                    directories,
                )
            results = save_lighting(lighting, names, directories["lighting"])
            if verbose:
                for name, (path, maximum) in zip(names, results):
                    progress.write(
                        f"{name}: max_luminance={maximum:.6g}, output={path.resolve()}"
                    )
            progress.update(len(path_batch))


if __name__ == "__main__":
    main()
