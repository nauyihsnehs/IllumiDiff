import json
import os
import tomllib
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import DataLoader

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
HDR_SUFFIXES = {".exr", ".hdr"}
LUM_WEIGHT_BGR = np.asarray([0.0722, 0.7152, 0.2126], dtype=np.float32)
EXR_SAVE_PARAMS = [
    cv.IMWRITE_EXR_TYPE,
    cv.IMWRITE_EXR_TYPE_HALF,
    cv.IMWRITE_EXR_COMPRESSION,
    cv.IMWRITE_EXR_COMPRESSION_PIZ,
]


def resolve_path(path, bases=None):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    bases = [Path(base) for base in (bases or [Path.cwd()])]
    candidates = [base / path for base in bases]
    return next(
        (candidate.resolve() for candidate in candidates if candidate.exists()),
        candidates[0].resolve(),
    )


def load_config(path):
    with Path(path).expanduser().open("rb") as stream:
        return tomllib.load(stream)


def save_config_snapshot(log_dir, config):
    Path(log_dir, "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def training_batch_size(config):
    global_batch_size = config["global_batch_size"]
    max_batch_size = config["max_batch_size_per_device"]
    device_count = max(torch.cuda.device_count(), 1)
    if global_batch_size % device_count:
        raise ValueError(
            f"global batch size {global_batch_size} is not divisible by "
            f"{device_count} devices"
        )
    device_batch_size = global_batch_size // device_count
    batch_size = next(
        size
        for size in range(min(device_batch_size, max_batch_size), 0, -1)
        if device_batch_size % size == 0
    )
    return batch_size, device_batch_size // batch_size


def loader_config(config, batch_size):
    result = dict(config)
    result["batch_size"] = batch_size
    result["persistent_workers"] = bool(
        result.get("persistent_workers") and result.get("num_workers", 0) > 0
    )
    return result


def checkpoint_state_dict(checkpoint):
    state_dict = checkpoint
    while isinstance(state_dict, dict) and isinstance(
        state_dict.get("state_dict"), dict
    ):
        state_dict = state_dict["state_dict"]
    return dict(state_dict)


def load_checkpoint(path):
    checkpoint = torch.load(Path(path).expanduser(), map_location="cpu")
    return checkpoint, checkpoint_state_dict(checkpoint)


def apply_ema_weights(model, checkpoint, required=False):
    ema_state = (
        checkpoint.get("ema_state_dict") if isinstance(checkpoint, dict) else None
    )
    if ema_state is None:
        if required:
            raise KeyError("checkpoint has no EMA state")
        return False

    expected = {
        name: parameter
        for name, parameter in model.named_parameters()
        if name.startswith("model.")
    }
    missing = expected.keys() - ema_state.keys()
    unexpected = ema_state.keys() - expected.keys()
    mismatched = [
        name
        for name in expected.keys() & ema_state.keys()
        if expected[name].shape != ema_state[name].shape
    ]
    if missing or unexpected or mismatched:
        raise RuntimeError(
            "EMA state does not match the diffusion model: "
            f"missing={len(missing)}, unexpected={len(unexpected)}, "
            f"mismatched={len(mismatched)}"
        )
    state_dict = model.state_dict()
    state_dict.update(ema_state)
    model.load_state_dict(state_dict, strict=True)
    return True


def load_model_weights(model, path, ema=None):
    checkpoint, state_dict = load_checkpoint(path)
    model.load_state_dict(state_dict, strict=True)
    if ema is not False:
        apply_ema_weights(model, checkpoint, required=ema is True)
    return checkpoint


def load_lightning_module(module, checkpoint, device, **kwargs):
    _, state_dict = load_checkpoint(checkpoint)
    model = module(**kwargs)
    model.on_load_checkpoint({"state_dict": state_dict})
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def sg_luminance_params(value):
    if value.shape[-1] == 5:
        result = value
    elif value.shape[-1] == 7:
        if isinstance(value, torch.Tensor):
            luminance = torch.sum(
                value[..., 4:7] * value.new_tensor(LUM_WEIGHT_BGR),
                dim=-1,
                keepdim=True,
            )
            result = torch.cat((value[..., :4], luminance), dim=-1)
        else:
            luminance = np.sum(value[..., 4:7] * LUM_WEIGHT_BGR, axis=-1, keepdims=True)
            result = np.concatenate((value[..., :4], luminance), axis=-1)
    else:
        raise ValueError(f"SG parameters must end in 5 or 7 values, got {value.shape}")

    finite = (
        torch.isfinite(result).all()
        if isinstance(result, torch.Tensor)
        else np.isfinite(result).all()
    )
    if not finite:
        raise ValueError("SG parameters contain non-finite values")
    return result


def load_sg_luminance(path):
    value = np.load(Path(path), allow_pickle=False).astype(np.float32)
    if value.ndim != 2:
        raise ValueError(f"SG parameters must be a 2D array: {path}, got {value.shape}")
    return torch.from_numpy(sg_luminance_params(value).astype(np.float32))


@lru_cache(maxsize=16)
def sg_view_directions(width, height, device_type, device_index, dtype):
    device = torch.device(
        device_type if device_index is None else f"{device_type}:{device_index}"
    )
    phi = torch.linspace(0, torch.pi, height, device=device, dtype=dtype)
    theta = torch.linspace(0, 2 * torch.pi, width, device=device, dtype=dtype)
    phi, theta = torch.meshgrid(phi, theta, indexing="ij")
    return torch.stack(
        (
            torch.cos(theta) * torch.sin(phi),
            torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
        ),
        dim=-1,
    ).reshape(-1, 3)


def render_sg_luminance(params, width, height=None, shifts=None, flips=None):
    height = height or width // 2
    squeeze = params.ndim == 2
    params = params[None] if squeeze else params

    view_dirs = sg_view_directions(
        width,
        height,
        params.device.type,
        params.device.index,
        params.dtype,
    )
    direction, lamb, weight = torch.split(params, [3, 1, 1], dim=-1)
    lobes = torch.exp(lamb * (torch.matmul(direction, view_dirs.t()) - 1))
    panorama = torch.einsum("bnp,bnc->bpc", lobes, weight)
    panorama = panorama.reshape(params.shape[0], height, width, 1).clamp_min_(0)

    if shifts is not None:
        batch, _, _, channels = panorama.shape
        shifts = torch.as_tensor(shifts, device=panorama.device).reshape(batch)
        columns = torch.arange(width, device=panorama.device)[None] - shifts[:, None]
        columns = columns.remainder(width)[:, None, :, None].expand(
            batch, height, width, channels
        )
        panorama = torch.gather(panorama, 2, columns)
    if flips is not None:
        flips = (
            torch.as_tensor(flips, device=panorama.device)
            .bool()
            .reshape(params.shape[0], 1, 1, 1)
        )
        panorama = torch.where(flips, torch.flip(panorama, dims=[2]), panorama)
    return panorama[0] if squeeze else panorama


def stage2_sg_from_params(params, width, height=None, shifts=None, flips=None):
    panorama = render_sg_luminance(params, width, height, shifts, flips)
    return torch.log1p(panorama).mul_(2).sub_(1)


def chunks(values, size):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def inference_paths(input_path):
    input_path = Path(input_path)
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"unsupported input image: {input_path}")
        return [input_path]
    paths = sorted(
        path
        for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise FileNotFoundError(f"no input images found: {input_path}")
    names = [path.stem for path in paths]
    if len(names) != len(set(names)):
        raise ValueError("inference inputs contain duplicate file stems")
    return paths


def load_inference_inputs(paths, size, device):
    images = []
    for path in paths:
        image = read_image(path, cv.IMREAD_COLOR)
        image = cv.resize(image, size, interpolation=cv.INTER_AREA)
        images.append(torch.from_numpy(image.astype(np.float32) / 255).permute(2, 0, 1))
    return torch.stack(images).to(device)


def image_paths(root, suffixes=None):
    root = Path(root)
    paths = [path for path in root.iterdir() if path.is_file()]
    if suffixes is not None:
        paths = [path for path in paths if path.suffix.lower() in suffixes]
    return sorted(paths)


def path_map(root, suffixes=None):
    result = {}
    for path in image_paths(root, suffixes):
        if path.stem in result:
            raise ValueError(
                f"duplicate image stem: {path.stem}\n{result[path.stem]}\n{path}"
            )
        result[path.stem] = path
    return result


def matched_paths(primary_root, modality_roots, primary_suffixes=None):
    primary = path_map(primary_root, primary_suffixes)
    if not primary:
        raise FileNotFoundError(f"no input images found: {primary_root}")
    modalities = {
        name: path_map(root)
        for name, root in modality_roots.items()
        if root is not None
    }
    return [
        {
            "name": stem,
            "input": path,
            **{name: items[stem] for name, items in modalities.items()},
        }
        for stem, path in primary.items()
    ]


def read_image(path, flags=cv.IMREAD_UNCHANGED):
    return cv.imread(Path(path).as_posix(), flags)


def write_image(path, image, params=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv.imwrite(path.as_posix(), image, params or []):
        raise RuntimeError(f"failed to write image: {path}")
    return path


def to_signed(value):
    return value * 2 - 1


def linear_to_srgb(image):
    image = np.clip(image, 0, 1)
    return np.where(
        image <= 0.0031308,
        image * 12.92,
        1.055 * image ** (1 / 2.4) - 0.055,
    )


def srgb_to_linear(image):
    return torch.where(
        image <= 0.04045,
        image / 12.92,
        ((image + 0.055) / 1.055).pow(2.4),
    )


def max_luminance(image):
    luminance = image @ LUM_WEIGHT_BGR
    finite = luminance[np.isfinite(luminance)]
    return float(finite.max()) if finite.size else float("nan")


def world_to_image(x, y, z):
    u = 0.5 * (1 + np.arctan2(x, -z) / np.pi)
    v = np.arccos(np.clip(y, -1, 1)) / np.pi
    return u, v


@lru_cache(maxsize=16)
def perspective_condition_maps(
    input_height,
    input_width,
    pano_width,
    pano_height,
    vfov,
):
    aspect = input_width / input_height
    vertical = np.tan(np.deg2rad(vfov) / 2)
    horizontal = vertical * aspect
    columns = np.linspace(-horizontal, horizontal, input_width)
    rows = np.linspace(vertical, -vertical, input_height)
    x, y = np.meshgrid(columns, rows)
    norm = np.sqrt(x * x + y * y + 1)
    u, v = world_to_image(x / norm, y / norm, -1 / norm)
    pano_x = np.clip((u * pano_width).astype(np.int64), 0, pano_width - 1)
    pano_y = np.clip((v * pano_height).astype(np.int64), 0, pano_height - 1)

    columns = (np.arange(pano_width) + 0.5) / pano_width
    rows = (np.arange(pano_height) + 0.5) / pano_height
    u, v = np.meshgrid(columns, rows)
    theta = np.pi * (u * 2 - 1)
    phi = np.pi * v
    world_x = np.sin(phi) * np.sin(theta)
    world_y = np.cos(phi)
    world_z = -np.sin(phi) * np.cos(theta)
    depth = -world_z
    safe_depth = np.where(depth > 1e-8, depth, 1)
    valid = (
        (depth > 1e-8)
        & (np.abs(world_x / safe_depth) < horizontal)
        & (np.abs(world_y / safe_depth) < vertical)
    )

    return pano_y, pano_x, valid[..., None].astype(np.float32)


def build_perspective_condition(input_rgb, pano_size, vfov=92):
    input_height, input_width = input_rgb.shape[:2]
    pano_width, pano_height = pano_size
    pano_y, pano_x, valid = perspective_condition_maps(
        input_height,
        input_width,
        pano_width,
        pano_height,
        vfov,
    )
    perspective = np.zeros(
        (pano_height, pano_width, input_rgb.shape[-1]), dtype=input_rgb.dtype
    )
    perspective[pano_y.ravel(), pano_x.ravel()] = input_rgb.reshape(
        -1, input_rgb.shape[-1]
    )
    return perspective.astype(np.float32), valid


def distributed_items(items):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        return items[rank::world_size]
    return items


def rotation_base(name):
    base, separator, suffix = name.rpartition("_")
    return base if separator and len(suffix) == 2 and suffix.isdigit() else None


def rotation_output_names(name, rotate_count):
    if rotate_count == 1:
        return [name]
    base = rotation_base(name)
    if base is None or not name.endswith("_00"):
        raise ValueError(f"rotation reuse requires a _00 sample, got {name!r}")
    return [f"{base}_{index:02d}" for index in range(rotate_count)]


def update_best_params(best_params, best_loss, params, loss):
    mask = torch.isfinite(loss) & (loss < best_loss)
    mask_view = mask[:, None, None]
    best_params = [
        torch.where(mask_view, param, best) for param, best in zip(params, best_params)
    ]
    return best_params, torch.where(mask, loss, best_loss)


def select_average_params(avg_params, best_params, avg_loss, best_loss):
    mask = (~torch.isfinite(avg_loss)) | (avg_loss > best_loss * 1.05)
    mask_view = mask[:, None, None]
    return [
        torch.where(mask_view, best, avg) for avg, best in zip(avg_params, best_params)
    ]


def prediction_dataloader(data, batch_size, num_workers):
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
