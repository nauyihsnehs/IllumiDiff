import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils import (
    IMAGE_SUFFIXES,
    build_perspective_condition,
    cv,
    load_sg_luminance,
    path_map,
    read_image,
)

SCENE_SOURCES = {
    "indoor": {"indoor", "laval-indoor"},
    "outdoor": {"outdoor", "laval-outdoor"},
}
SCENES = {"all", *SCENE_SOURCES}
SOURCE_NAMES = set().union(*SCENE_SOURCES.values())
MODEL_DOWNSAMPLE_FACTOR = 128


def validate_res(res):
    if not isinstance(res, (list, tuple)) or len(res) != 2:
        raise ValueError("res must be [width, height]")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in res):
        raise ValueError("res width and height must be integers")
    if any(value <= 0 for value in res):
        raise ValueError("res width and height must be positive")
    if any(value % MODEL_DOWNSAMPLE_FACTOR for value in res):
        raise ValueError(
            f"res width and height must be multiples of {MODEL_DOWNSAMPLE_FACTOR}"
        )
    return tuple(res)


def sample_parts(name):
    parts = name.rsplit("_", 2)
    if (
        len(parts) != 3
        or not parts[0]
        or parts[1] not in SOURCE_NAMES
        or not parts[2].isdigit()
    ):
        return None
    return parts


def filter_scene_names(names, scene):
    if scene not in SCENES:
        raise ValueError(f"unknown scene {scene!r}; choose from {sorted(SCENES)}")
    names = list(names)
    if scene == "all":
        return names
    parsed = {name: sample_parts(name) for name in names}
    invalid = [name for name, parts in parsed.items() if parts is None]
    if invalid:
        suffix = f"\n... and {len(invalid) - 50} more" if len(invalid) > 50 else ""
        raise ValueError(
            "invalid panorama sample names for scene filtering:\n"
            + "\n".join(invalid[:50])
            + suffix
        )
    selected = [
        name for name, parts in parsed.items() if parts[1] in SCENE_SOURCES[scene]
    ]
    if not selected:
        raise ValueError(f"no {scene} panorama condition samples found")
    return selected


def resize_bgr(path, size):
    image = read_image(path, cv.IMREAD_COLOR)
    return cv.resize(image, size, interpolation=cv.INTER_AREA)


def load_rgb(path, size):
    with Image.open(path) as image:
        image = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float32) / 255


def sg_energy(params):
    lamb = params[..., 3].clamp_min(1e-6)
    area = -torch.expm1(-2 * lamb) / lamb
    return (params[..., 4] * area).sum()


def match_sg_energy(value, reference):
    result = value.clone()
    result[..., 4] *= sg_energy(reference) / sg_energy(value).clamp_min(1e-8)
    return result


class PanoConditionDataset(Dataset):
    def __init__(
        self,
        input_root,
        pano_root,
        sg_root,
        asg_image_root,
        res,
        projection_vfov=92,
        sg_pre_root=None,
        asg_pre_image_root=None,
        use_pre_asg=False,
        use_pre_sg=False,
        scene="all",
    ):
        super().__init__()
        self.pano_size = validate_res(res)
        self.width, self.height = self.pano_size
        self.projection_vfov = projection_vfov
        self.use_pre_asg = use_pre_asg
        self.use_pre_sg = use_pre_sg
        self.scene = scene
        self.input_map = path_map(input_root, IMAGE_SUFFIXES)
        self.modalities = {
            "pano": path_map(pano_root, IMAGE_SUFFIXES),
            "sg_pre": path_map(sg_pre_root, {".npy"}),
        }
        if not use_pre_sg:
            self.modalities["sg"] = path_map(sg_root, {".npy"})
        if asg_image_root is not None:
            self.modalities["asg_image"] = path_map(
                asg_image_root,
                IMAGE_SUFFIXES,
            )
        if use_pre_asg:
            self.modalities.update(
                asg_image_pre=path_map(asg_pre_image_root, IMAGE_SUFFIXES),
            )
        self.names = filter_scene_names(self.input_map, scene)
        if not self.names:
            raise ValueError(f"no panorama condition inputs found: {input_root}")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        input_bgr = resize_bgr(self.input_map[name], (256, 256))
        input_bgr = input_bgr.astype(np.float32) / 255
        input_rgb = input_bgr[..., ::-1].copy()
        perspective, known_mask = build_perspective_condition(
            input_rgb,
            self.pano_size,
            self.projection_vfov,
        )

        pano = load_rgb(self.modalities["pano"][name], self.pano_size)
        asg_key = "asg_image_pre" if self.use_pre_asg else "asg_image"
        asg_rgb = load_rgb(
            self.modalities[asg_key][name],
            self.pano_size,
        )
        sg_pre = load_sg_luminance(self.modalities["sg_pre"][name])
        if self.use_pre_sg:
            sg = sg_pre
        else:
            sg = load_sg_luminance(self.modalities["sg"][name])
            sg = match_sg_energy(sg, sg_pre)
        return {
            "pano": torch.from_numpy(pano * 2 - 1).float(),
            "perspective": torch.from_numpy(perspective * 2 - 1).float(),
            "known_mask": torch.from_numpy(known_mask).float(),
            "asg": torch.from_numpy(asg_rgb * 2 - 1).float(),
            "sg": sg,
            "img_name": name,
        }
