import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import click
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from utils import IMAGE_SUFFIXES, cv, image_paths, world_to_image, write_image

BOTTOM_SIZE = 512
BOTTOM_VFOV = 90
BOTTOM_ROTATION = (math.pi / 2, 0.0, 0.0)
DEFAULT_PROMPT = "clean unobstructed floor or ground surface"
DEFAULT_NEGATIVE_PROMPT = (
    "tripod, camera, stand, person, text, watermark, object, blurry, distorted"
)
PNG_PARAMS = [cv.IMWRITE_PNG_COMPRESSION, 1]
MAP_CACHE = {}
WINDOW_NAME = "Manual Bottom Mask"


def collect_input_targets(input_dir, max_count):
    paths = image_paths(input_dir, IMAGE_SUFFIXES)
    paths = paths if max_count is None else paths[:max_count]
    return [
        {"path": path, "name": path.stem, "index": index}
        for index, path in enumerate(paths)
    ]


def list_output_images(path):
    root = Path(path)
    paths = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
    paths.sort()
    return paths


def collect_output_targets(dirs, folder, max_count):
    root = dirs[folder]
    if not root.is_dir():
        raise ValueError(f"missing {folder} folder: {root}")
    paths = list_output_images(root)
    if max_count is not None:
        paths = paths[:max_count]
    return [
        {"path": path, "name": path.stem, "index": index}
        for index, path in enumerate(paths)
    ]


def collect_stage_targets(stage, input_dir, dirs, max_count):
    if stage == "prepare":
        return collect_input_targets(input_dir, max_count)
    if stage in {"mask", "inpaint"}:
        return collect_output_targets(dirs, "bottom", max_count)
    if stage == "merge":
        return collect_output_targets(dirs, "bottom_inpainted", max_count)
    raise ValueError(f"unknown stage: {stage}")


def output_dirs(output_root):
    root = Path(output_root)
    return {
        "ldr": root / "ldr",
        "bottom": root / "bottom",
        "bottom_mask": root / "bottom_mask",
        "bottom_masked": root / "bottom_masked",
        "bottom_inpainted": root / "bottom_inpainted",
        "inpainted": root / "inpainted",
    }


def ensure_dirs(paths):
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def output_path(dirs, folder, name):
    return dirs[folder] / f"{name}.png"


def overwrite_or_missing(path, skip_existing):
    return not skip_existing or not path.is_file()


def read_bgr(path, label):
    image = cv.imread(Path(path).as_posix(), cv.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to read {label}: {path}")
    return image


def read_mask(path):
    mask = cv.imread(Path(path).as_posix(), cv.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"failed to read mask: {path}")
    return mask


def require_file(path, message):
    if not Path(path).is_file():
        raise ValueError(f"{message}: {path}")


def read_ldr(path, size=None):
    image = read_bgr(path, "LDR pano")
    height, width = image.shape[:2]
    if width != height * 2:
        raise ValueError(f"LDR panorama must be 2:1, got {width}x{height}: {path}")
    if size is None:
        return image
    actual = (image.shape[1], image.shape[0])
    if actual != size:
        raise ValueError(
            f"expected {size[0]}x{size[1]} LDR pano, got {actual[0]}x{actual[1]}: {path}"
        )
    return image


def fov_extents(vfov, width, height):
    vfov_rad = np.deg2rad(vfov)
    mv = np.tan(vfov_rad / 2)
    hfov = 2 * np.arctan(mv * width / height)
    mu = np.tan(hfov / 2)
    return mu, mv


def rotation_matrix(rotation_ear):
    el, az, ro = rotation_ear
    return R.from_euler("ZXY", [ro, el, -az]).as_matrix()


def image2world(u, v):
    theta = np.pi * (u * 2 - 1)
    phi = np.pi * v
    x = np.sin(phi) * np.sin(theta)
    y = np.cos(phi)
    z = -np.sin(phi) * np.cos(theta)
    return x, y, z


def bottom_to_erp_maps(pano_shape):
    key = ("bottom", pano_shape)
    if key in MAP_CACHE:
        return MAP_CACHE[key]

    pano_h, pano_w = pano_shape
    mu, mv = fov_extents(BOTTOM_VFOV, BOTTOM_SIZE, BOTTOM_SIZE)
    cols = np.linspace(-mu, mu, BOTTOM_SIZE)
    rows = np.linspace(mv, -mv, BOTTOM_SIZE)
    x, y = np.meshgrid(cols, rows)
    norm = np.sqrt(x * x + y * y + 1)
    local = np.vstack(
        (x.ravel() / norm.ravel(), y.ravel() / norm.ravel(), -1 / norm.ravel())
    )
    world = rotation_matrix(BOTTOM_ROTATION).T.dot(local)
    u, v = world_to_image(world[0], world[1], world[2])
    map_x = np.mod(u.reshape(BOTTOM_SIZE, BOTTOM_SIZE) * pano_w - 0.5, pano_w)
    map_y = np.clip(v.reshape(BOTTOM_SIZE, BOTTOM_SIZE) * pano_h - 0.5, 0, pano_h - 1)
    maps = map_x.astype(np.float32), map_y.astype(np.float32)
    MAP_CACHE[key] = maps
    return maps


def erp_to_bottom_maps(pano_shape):
    key = ("erp", pano_shape)
    if key in MAP_CACHE:
        return MAP_CACHE[key]

    pano_h, pano_w = pano_shape
    cols = (np.arange(pano_w) + 0.5) / pano_w
    rows = (np.arange(pano_h) + 0.5) / pano_h
    u, v = np.meshgrid(cols, rows)
    x, y, z = image2world(u, v)
    local = rotation_matrix(BOTTOM_ROTATION).dot(
        np.vstack((x.ravel(), y.ravel(), z.ravel()))
    )
    lx = local[0].reshape(pano_h, pano_w)
    ly = local[1].reshape(pano_h, pano_w)
    lz = local[2].reshape(pano_h, pano_w)
    denom = -lz
    safe_denom = np.where(denom > 1e-8, denom, 1.0)
    plane_x = lx / safe_denom
    plane_y = ly / safe_denom
    mu, mv = fov_extents(BOTTOM_VFOV, BOTTOM_SIZE, BOTTOM_SIZE)
    valid = (
        (denom > 1e-8)
        & (plane_x >= -mu)
        & (plane_x <= mu)
        & (plane_y >= -mv)
        & (plane_y <= mv)
    )
    map_x = (plane_x + mu) / (2 * mu) * (BOTTOM_SIZE - 1)
    map_y = (mv - plane_y) / (2 * mv) * (BOTTOM_SIZE - 1)
    map_x = np.where(valid, map_x, 0).astype(np.float32)
    map_y = np.where(valid, map_y, 0).astype(np.float32)
    maps = map_x, map_y, valid
    MAP_CACHE[key] = maps
    return maps


def crop_bottom(pano):
    map_x, map_y = bottom_to_erp_maps(pano.shape[:2])
    return cv.remap(pano, map_x, map_y, cv.INTER_LINEAR, borderMode=cv.BORDER_WRAP)


def project_bottom_to_erp(bottom, pano_shape, interpolation):
    map_x, map_y, valid = erp_to_bottom_maps(pano_shape)
    projected = cv.remap(
        bottom,
        map_x,
        map_y,
        interpolation,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=0,
    )
    projected[~valid] = 0
    return projected, valid


def circle_distance(size):
    center = (size - 1) / 2
    coords = np.arange(size)
    x, y = np.meshgrid(coords, coords)
    return np.sqrt((x - center) ** 2 + (y - center) ** 2)


def hard_circle_mask(size, radius):
    mask = circle_distance(size) <= radius
    return (mask * 255).astype(np.uint8)


def feather_binary_mask(mask, feather):
    mask = (mask > 0).astype(np.uint8)
    if feather <= 0:
        return mask.astype(np.float32)
    distance = cv.distanceTransform(mask, cv.DIST_L2, 5)
    return np.clip(distance / feather, 0, 1).astype(np.float32)


def save_mask_preview(crop, mask, save_path):
    image = crop.astype(np.float32)
    color = np.zeros_like(image)
    color[..., 2] = 255
    alpha = ((mask > 0).astype(np.float32) * 0.45)[..., None]
    preview = image * (1 - alpha) + color * alpha
    write_image(save_path, np.clip(preview, 0, 255).astype(np.uint8), PNG_PARAMS)


def cut_masked_pixels(image, mask):
    image = image.copy()
    image[mask > 0] = 0
    return image


def validate_options(mask_radius, mask_feather, batch_size, num_workers, max_count):
    if mask_radius <= 0 or mask_radius > BOTTOM_SIZE // 2:
        raise ValueError(f"--mask_radius must be in [1, {BOTTOM_SIZE // 2}]")
    if mask_feather < 0:
        raise ValueError("--mask_feather must be >= 0")
    if batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if num_workers < 1:
        raise ValueError("--num_workers must be >= 1")
    if max_count is not None and max_count < 1:
        raise ValueError("--max_count must be >= 1")


def prepare_target(args):
    target, dirs, mask_radius, skip_existing = args
    name = target["name"]
    ldr_path = output_path(dirs, "ldr", name)
    bottom_path = output_path(dirs, "bottom", name)
    mask_path = output_path(dirs, "bottom_mask", name)
    masked_path = output_path(dirs, "bottom_masked", name)
    outputs = [ldr_path, bottom_path, mask_path, masked_path]

    if skip_existing and all(path.is_file() for path in outputs):
        return name

    if overwrite_or_missing(ldr_path, skip_existing):
        ldr = read_ldr(target["path"])
        write_image(ldr_path, ldr, PNG_PARAMS)
    else:
        ldr = read_ldr(ldr_path)

    mask = hard_circle_mask(BOTTOM_SIZE, mask_radius)
    if overwrite_or_missing(bottom_path, skip_existing):
        bottom = crop_bottom(ldr)
        write_image(bottom_path, bottom, PNG_PARAMS)
    else:
        bottom = read_bgr(bottom_path, "bottom crop")

    if overwrite_or_missing(mask_path, skip_existing):
        write_image(mask_path, mask, PNG_PARAMS)
    if overwrite_or_missing(masked_path, skip_existing):
        save_mask_preview(bottom, mask, masked_path)
    return name


def prepare_targets(targets, dirs, mask_radius, num_workers, skip_existing):
    ensure_dirs(
        [dirs["ldr"], dirs["bottom"], dirs["bottom_mask"], dirs["bottom_masked"]]
    )
    args = [(target, dirs, mask_radius, skip_existing) for target in targets]
    if num_workers <= 1:
        for item in tqdm(args, desc="prepare"):
            prepare_target(item)
        return

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for _ in tqdm(
            executor.map(prepare_target, args), total=len(args), desc="prepare"
        ):
            pass


def normalize_mask(mask, shape):
    if mask.shape != shape:
        mask = cv.resize(mask, (shape[1], shape[0]), interpolation=cv.INTER_NEAREST)
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def overlay_image(image, mask, alpha):
    red = np.zeros_like(image)
    red[..., 2] = 255
    blend = image.astype(np.float32) * (1 - alpha) + red.astype(np.float32) * alpha
    return np.where((mask > 0)[..., None], blend, image).astype(np.uint8)


class MaskApp:
    def __init__(self, paths, mask_root, preview_root, brush_radius, alpha):
        self.paths = paths
        self.mask_root = Path(mask_root)
        self.preview_root = Path(preview_root)
        self.brush_radius = brush_radius
        self.alpha = alpha
        self.index = 0
        self.image = None
        self.mask = None
        self.undo = []
        self.mode = None
        self.running = True

    def load(self):
        path = self.paths[self.index]
        self.image = read_bgr(path, "bottom crop")
        mask_path = self.mask_root / path.name
        self.mask = (
            read_mask(mask_path)
            if mask_path.is_file()
            else hard_circle_mask(BOTTOM_SIZE, 125)
        )
        self.mask = normalize_mask(self.mask, self.image.shape[:2])
        self.undo = []
        cv.setWindowTitle(
            WINDOW_NAME, f"{path.stem} {self.index + 1}/{len(self.paths)}"
        )

    def save(self):
        path = self.paths[self.index]
        write_image(self.mask_root / path.name, self.mask, PNG_PARAMS)
        write_image(
            self.preview_root / path.name,
            overlay_image(self.image, self.mask, self.alpha),
            PNG_PARAMS,
        )

    def point(self, x, y):
        _, _, width, height = cv.getWindowImageRect(WINDOW_NAME)
        image_height, image_width = self.image.shape[:2]
        return (
            min(max(round(x * image_width / max(width, 1)), 0), image_width - 1),
            min(max(round(y * image_height / max(height, 1)), 0), image_height - 1),
        )

    def mouse(self, event, x, y, flags, param):
        x, y = self.point(x, y)
        if event in {cv.EVENT_LBUTTONDOWN, cv.EVENT_RBUTTONDOWN}:
            self.undo.append(self.mask.copy())
            self.mode = "draw" if event == cv.EVENT_LBUTTONDOWN else "erase"
        if event == cv.EVENT_MOUSEMOVE and not flags & (
            cv.EVENT_FLAG_LBUTTON | cv.EVENT_FLAG_RBUTTON
        ):
            return
        if (
            event in {cv.EVENT_LBUTTONDOWN, cv.EVENT_RBUTTONDOWN, cv.EVENT_MOUSEMOVE}
            and self.mode
        ):
            value = 255 if self.mode == "draw" else 0
            cv.circle(self.mask, (x, y), self.brush_radius, value, -1)
        if event in {cv.EVENT_LBUTTONUP, cv.EVENT_RBUTTONUP}:
            self.mode = None

    def run(self):
        self.mask_root.mkdir(parents=True, exist_ok=True)
        self.preview_root.mkdir(parents=True, exist_ok=True)
        cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)
        cv.setMouseCallback(WINDOW_NAME, self.mouse)
        self.load()
        while self.running:
            cv.imshow(WINDOW_NAME, overlay_image(self.image, self.mask, self.alpha))
            key = cv.waitKeyEx(20)
            if key in {10, 13, 32}:
                self.save()
                self.index += 1
                self.running = self.index < len(self.paths)
                if self.running:
                    self.load()
            if key in {ord("u"), ord("U")} and self.undo:
                self.mask = self.undo.pop()
            if key in {ord("q"), ord("Q"), 27}:
                self.running = False
        cv.destroyWindow(WINDOW_NAME)


def edit_masks(targets, dirs, brush_radius, alpha):
    paths = [dirs["bottom"] / f"{target['name']}.png" for target in targets]
    MaskApp(
        paths, dirs["bottom_mask"], dirs["bottom_masked"], brush_radius, alpha
    ).run()


def load_inpaint_pipeline(model_name):
    import torch
    from diffusers import DiffusionPipeline

    device = torch.device("cuda")
    pipe = DiffusionPipeline.from_pretrained(model_name, local_files_only=True)
    pipe = pipe.to(device)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    return pipe, torch


def batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def inpaint_targets(
    targets,
    dirs,
    model_name,
    seed,
    steps,
    guidance_scale,
    prompt,
    negative_prompt,
    batch_size,
    cut_masked_bottom,
    skip_existing,
):
    ensure_dirs([dirs["bottom_inpainted"]])
    pending = [
        target
        for target in targets
        if not skip_existing
        or not output_path(dirs, "bottom_inpainted", target["name"]).is_file()
    ]
    if not pending:
        return

    pipe, torch = load_inpaint_pipeline(model_name)
    device = pipe.device
    total_batches = math.ceil(len(pending) / batch_size)

    for batch in tqdm(
        batched(pending, batch_size), total=total_batches, desc="inpaint"
    ):
        images = []
        masks = []
        generators = []
        save_paths = []
        for target in batch:
            name = target["name"]
            bottom_path = output_path(dirs, "bottom", name)
            mask_path = output_path(dirs, "bottom_mask", name)
            require_file(bottom_path, "missing bottom crop, run --stage prepare first")
            require_file(mask_path, "missing bottom mask, run --stage prepare first")

            image = read_bgr(bottom_path, "bottom crop")
            mask = read_mask(mask_path)
            if cut_masked_bottom:
                image = cut_masked_pixels(image, mask)
            images.append(Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB)))
            masks.append(Image.fromarray(mask).convert("L"))
            generators.append(
                torch.Generator(device=device).manual_seed(seed + target["index"])
            )
            save_paths.append(output_path(dirs, "bottom_inpainted", name))

        result = pipe(
            prompt=[prompt] * len(images),
            negative_prompt=[negative_prompt] * len(images),
            image=images,
            mask_image=masks,
            width=BOTTOM_SIZE,
            height=BOTTOM_SIZE,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generators,
        ).images
        for image, save_path in zip(result, save_paths):
            image.save(save_path.as_posix())


def merge_target(args):
    target, dirs, mask_feather, skip_existing = args
    name = target["name"]
    ldr_path = output_path(dirs, "ldr", name)
    mask_path = output_path(dirs, "bottom_mask", name)
    bottom_path = output_path(dirs, "bottom_inpainted", name)
    save_path = output_path(dirs, "inpainted", name)

    if skip_existing and save_path.is_file():
        return name

    require_file(ldr_path, "missing LDR pano, run --stage prepare first")
    require_file(mask_path, "missing bottom mask, run --stage prepare first")
    require_file(
        bottom_path, "missing inpainted bottom crop, run --stage inpaint first"
    )

    pano = read_ldr(ldr_path)
    bottom = read_bgr(bottom_path, "inpainted bottom crop")
    mask = read_mask(mask_path)
    alpha_bottom = feather_binary_mask(mask, mask_feather)
    projected, _ = project_bottom_to_erp(bottom, pano.shape[:2], cv.INTER_LINEAR)
    alpha, _ = project_bottom_to_erp(alpha_bottom, pano.shape[:2], cv.INTER_LINEAR)
    alpha = np.clip(alpha, 0, 1)[..., None]
    merged = (
        pano.astype(np.float32) * (1 - alpha) + projected.astype(np.float32) * alpha
    )
    write_image(
        save_path, np.round(np.clip(merged, 0, 255)).astype(np.uint8), PNG_PARAMS
    )
    return name


def merge_targets(targets, dirs, mask_feather, num_workers, skip_existing):
    ensure_dirs([dirs["inpainted"]])
    args = [(target, dirs, mask_feather, skip_existing) for target in targets]
    if num_workers <= 1:
        for item in tqdm(args, desc="merge"):
            merge_target(item)
        return

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for _ in tqdm(executor.map(merge_target, args), total=len(args), desc="merge"):
            pass


def stage_sequence(stage):
    return ["prepare", "inpaint", "merge"] if stage == "all" else [stage]


@click.command()
@click.option("--input-dir", required=True)
@click.option("--output-root", required=True)
@click.option(
    "--stage",
    type=click.Choice(["prepare", "mask", "inpaint", "merge", "all"]),
    default="inpaint",
)
@click.option("--mask_radius", type=int, default=125, show_default=True)
@click.option("--mask_feather", type=int, default=16, show_default=True)
@click.option("--brush-radius", default=18)
@click.option("--mask-alpha", default=0.45)
@click.option(
    "--model_name",
    type=str,
    default="stabilityai/stable-diffusion-2-inpainting",
    show_default=True,
)
@click.option("--seed", type=int, default=114514, show_default=True)
@click.option("--num_inference_steps", type=int, default=30, show_default=True)
@click.option("--guidance_scale", type=float, default=7.5, show_default=True)
@click.option("--batch_size", type=int, default=4, show_default=True)
@click.option("--prompt", type=str, default=DEFAULT_PROMPT, show_default=True)
@click.option(
    "--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT, show_default=True
)
@click.option(
    "--cut_masked_bottom/--keep_masked_bottom", default=False, show_default=True
)
@click.option("--num_workers", type=int, default=1, show_default=True)
@click.option("--max_count", type=int, default=None)
@click.option("--skip_existing/--overwrite", default=True, show_default=True)
def main(
    input_dir,
    output_root,
    stage,
    mask_radius,
    mask_feather,
    brush_radius,
    mask_alpha,
    model_name,
    seed,
    num_inference_steps,
    guidance_scale,
    batch_size,
    prompt,
    negative_prompt,
    cut_masked_bottom,
    num_workers,
    max_count,
    skip_existing,
):
    validate_options(mask_radius, mask_feather, batch_size, num_workers, max_count)
    if brush_radius < 1 or not 0 <= mask_alpha <= 1:
        raise click.UsageError(
            "--brush-radius must be positive and --mask-alpha in [0, 1]"
        )
    dirs = output_dirs(output_root)
    click.echo(f"output root: {Path(output_root)}")

    for item in stage_sequence(stage):
        targets = collect_stage_targets(item, input_dir, dirs, max_count)
        if not targets:
            raise ValueError(f"no {item} targets found")
        click.echo(f"{item} targets: {len(targets)}")
        if item == "prepare":
            prepare_targets(targets, dirs, mask_radius, num_workers, skip_existing)
        elif item == "mask":
            edit_masks(targets, dirs, brush_radius, mask_alpha)
        elif item == "inpaint":
            inpaint_targets(
                targets,
                dirs,
                model_name,
                seed,
                num_inference_steps,
                guidance_scale,
                prompt,
                negative_prompt,
                batch_size,
                cut_masked_bottom,
                skip_existing,
            )
        elif item == "merge":
            merge_targets(targets, dirs, mask_feather, num_workers, skip_existing)


if __name__ == "__main__":
    main()
