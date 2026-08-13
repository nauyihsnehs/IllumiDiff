from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import click
import numpy as np
from tqdm import tqdm

from utils import (
    EXR_SAVE_PARAMS,
    HDR_SUFFIXES,
    cv,
    image_paths,
    linear_to_srgb,
    read_image,
    world_to_image,
    write_image,
)

PANO_WIDTH = 1024
PERS_SIZE = 512
PNG_PARAMS = [cv.IMWRITE_PNG_COMPRESSION, 1]
OUTPUTS = {
    "pano_hdr": ("pano_hdr_1024", ".exr"),
    "pano_ldr": ("pano_ldr_1024", ".png"),
    "pers_hdr": ("pers_hdr_512", ".exr"),
    "pers_ldr": ("pers_ldr_512", ".png"),
}


@lru_cache(maxsize=16)
def perspective_uv(size, fov):
    extent = np.tan(np.deg2rad(fov) / 2)
    axis = (np.arange(size, dtype=np.float32) + 0.5) / size
    x, y = np.meshgrid((axis * 2 - 1) * extent, (1 - axis * 2) * extent)
    norm = np.sqrt(x * x + y * y + 1)
    u, v = world_to_image(x / norm, y / norm, -1 / norm)
    return u.astype(np.float32), v.astype(np.float32)


def collect_sources(input_dirs, max_count):
    sources = []
    names = {}
    for input_dir in input_dirs:
        root = Path(input_dir)
        paths = image_paths(root, HDR_SUFFIXES)
        if not paths:
            raise ValueError(f"no HDR panoramas found: {root}")
        for path in paths:
            name = f"{path.stem}_{root.name}"
            if name in names:
                raise ValueError(
                    f"duplicate sample name {name!r}: {names[name]} and {path}"
                )
            names[name] = path
            sources.append((path, name))
    return sources if max_count is None else sources[:max_count]


def read_hdr(path):
    image = read_image(path, cv.IMREAD_UNCHANGED)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"HDR panorama must have at least 3 channels: {path}")
    height, width = image.shape[:2]
    if width != height * 2:
        raise ValueError(f"HDR panorama must be 2:1, got {width}x{height}: {path}")
    image = np.nan_to_num(image[..., :3], nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(image, 0).astype(np.float32)


def perspective_view(panorama, u, v, rotation, size):
    height, width = panorama.shape[:2]
    map_x = (np.remainder(u + rotation, 1) * width - 0.5).astype(np.float32)
    map_y = (v * height - 0.5).astype(np.float32)
    view = cv.remap(panorama, map_x, map_y, cv.INTER_LINEAR, borderMode=cv.BORDER_WRAP)
    return cv.resize(view, (size, size), interpolation=cv.INTER_AREA).astype(np.float32)


def exposure_from_perspective(perspective, name):
    luminance = perspective @ np.asarray([0.0722, 0.7152, 0.2126], dtype=np.float32)
    positive = luminance[luminance > 0]
    if not positive.size:
        raise ValueError(f"perspective HDR has no positive luminance: {name}")
    return np.float32(0.45**2.4 / max(float(np.median(positive)), 1e-8))


def make_ldr(hdr, exposure):
    mapped = linear_to_srgb(np.minimum(hdr, 1 / exposure) * exposure)
    return np.round(np.clip(mapped, 0.5 / 255, 1) * 255).astype(np.uint8)


def target_paths(output_dirs, name):
    return {
        key: output_dirs[key] / f"{name}{extension}"
        for key, (_, extension) in OUTPUTS.items()
    }


def write_outputs(paths, values, skip_existing):
    for key, value in values.items():
        if skip_existing and paths[key].is_file():
            continue
        params = EXR_SAVE_PARAMS if key.endswith("hdr") else PNG_PARAMS
        write_image(paths[key], value, params)


def process_source(args):
    path, source_name, output_dirs, fov, rotate_count, supersample, skip_existing = args
    all_paths = [
        target_paths(output_dirs, f"{source_name}_{index:02d}")
        for index in range(rotate_count)
    ]
    if skip_existing and all(
        all(path.is_file() for path in paths.values()) for paths in all_paths
    ):
        return 0

    panorama = read_hdr(path)
    pano_hdr = cv.resize(
        panorama,
        (PANO_WIDTH, PANO_WIDTH // 2),
        interpolation=cv.INTER_AREA,
    )
    u, v = perspective_uv(PERS_SIZE * supersample, fov)

    for index, paths in enumerate(all_paths):
        rotation = index / rotate_count
        shift = -round(rotation * PANO_WIDTH)
        rotated_pano = np.roll(pano_hdr, shift, axis=1)
        pers_hdr = perspective_view(
            panorama,
            u,
            v,
            rotation,
            PERS_SIZE,
        )
        exposure = exposure_from_perspective(pers_hdr, f"{source_name}_{index:02d}")
        values = {
            "pano_hdr": rotated_pano,
            "pano_ldr": make_ldr(rotated_pano, exposure),
            "pers_hdr": pers_hdr,
            "pers_ldr": make_ldr(pers_hdr, exposure),
        }
        write_outputs(paths, values, skip_existing)
    return rotate_count


def validate_options(
    input_dirs, rotate_count, fov, supersample, num_workers, max_count
):
    if not input_dirs:
        raise click.UsageError("at least one --input-dir is required")
    if not 1 <= rotate_count <= 100:
        raise click.UsageError("--rotate-count must be in [1, 100]")
    if not 1 <= fov < 180:
        raise click.UsageError("--fov must be in [1, 180)")
    if supersample < 1 or num_workers < 1:
        raise click.UsageError("--supersample and --num-workers must be >= 1")
    if max_count is not None and max_count < 1:
        raise click.UsageError("--max-count must be >= 1")


@click.command()
@click.option("--input-dir", "input_dirs", multiple=True, required=True)
@click.option("--dataset-root", required=True)
@click.option("--rotate-count", default=10)
@click.option("--fov", default=50.0)
@click.option("--supersample", default=2)
@click.option("--num-workers", default=1)
@click.option("--max-count", default=None)
@click.option("--skip-existing/--overwrite", default=True)
def main(
    input_dirs,
    dataset_root,
    rotate_count,
    fov,
    supersample,
    num_workers,
    max_count,
    skip_existing,
):
    validate_options(input_dirs, rotate_count, fov, supersample, num_workers, max_count)
    sources = collect_sources(input_dirs, max_count)
    if not sources:
        raise ValueError("no HDR panoramas found")

    root = Path(dataset_root).expanduser()
    output_dirs = {key: root / folder for key, (folder, _) in OUTPUTS.items()}
    for path in output_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    args = [
        (path, name, output_dirs, fov, rotate_count, supersample, skip_existing)
        for path, name in sources
    ]
    if num_workers == 1:
        written = sum(process_source(item) for item in tqdm(args, desc="build dataset"))
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            written = sum(
                tqdm(
                    executor.map(process_source, args),
                    total=len(args),
                    desc="build dataset",
                )
            )
    click.echo(f"sources: {len(sources)}, samples processed: {written}, output: {root}")


if __name__ == "__main__":
    main()
