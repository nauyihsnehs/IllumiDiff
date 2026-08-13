import os
import random
from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from utils import (
    HDR_SUFFIXES,
    IMAGE_SUFFIXES,
    LUM_WEIGHT_BGR,
    cv,
    linear_to_srgb,
    load_sg_luminance,
    matched_paths,
    read_image,
    world_to_image,
)

PERS_RES = (256, 256)


def resolve_data_path(base_path, path):
    if path is None:
        return None
    path = Path(path).expanduser()
    return path if path.is_absolute() else Path(base_path).expanduser() / path


def limit(records, max_count):
    return records if max_count is None else records[:max_count]


def resize_image(image, size):
    if not isinstance(size, int):
        return cv.resize(image, size, interpolation=cv.INTER_AREA)
    short_edge = min(image.shape[:2])
    if short_edge <= size:
        return image
    scale = size / short_edge
    target = (round(image.shape[1] * scale), round(image.shape[0] * scale))
    return cv.resize(image, target, interpolation=cv.INTER_AREA)


def color_tensor(path, size, signed=False):
    image = read_image(path, cv.IMREAD_COLOR)
    image = resize_image(image, size).astype(np.float32) / 255
    tensor = torch.from_numpy(image).permute(2, 0, 1)
    return tensor * 2 - 1 if signed else tensor


def mask_tensor(path, size, signed=False):
    image = read_image(path, cv.IMREAD_GRAYSCALE)
    image = resize_image(image, size).astype(np.float32) / 255
    tensor = torch.from_numpy(image)[None]
    return tensor * 2 - 1 if signed else tensor


class BaseDataModule(L.LightningDataModule):
    def __init__(self, batch_size=16, num_workers=None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = (
            min(os.cpu_count() or 1, 8) if num_workers is None else num_workers
        )
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.predict_data = None
        self.data_num = None

    def loader(self, data, training=False):
        return DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=training,
            drop_last=training,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self):
        return self.loader(self.train_data, training=True)

    def val_dataloader(self):
        return self.loader(self.val_data)

    def test_dataloader(self):
        return self.loader(self.test_data)

    def predict_dataloader(self):
        return self.loader(self.predict_data)

    def fit_data(self, dataset):
        self.train_data, self.val_data = random_split(
            dataset,
            [0.95, 0.05],
            torch.Generator().manual_seed(42),
        )
        self.data_num = [len(self.train_data), len(self.val_data)]


class SGNetDataset(Dataset):
    def __init__(self, rgb_path, input_ls_path=None, sg_path=None, max_count=None):
        modalities = {"mask": input_ls_path}
        if sg_path is not None:
            modalities["sg"] = sg_path
        self.records = limit(
            matched_paths(rgb_path, modalities, IMAGE_SUFFIXES), max_count
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        rgb = color_tensor(record["input"], PERS_RES, signed=True)
        mask = mask_tensor(record["mask"], PERS_RES)
        if "sg" not in record:
            return rgb, mask, record["name"]

        return rgb, mask, load_sg_luminance(record["sg"]), record["name"]


class SGNetDataModule(BaseDataModule):
    def __init__(
        self,
        base_path,
        input_path,
        input_ls_path=None,
        sg_path=None,
        resolution=(256, 256),
        batch_size=1,
        id_net_ckpt_path=None,
        max_count=None,
        num_workers=None,
    ):
        super().__init__(batch_size, num_workers)
        self.input_path = resolve_data_path(base_path, input_path)
        self.input_ls_path = resolve_data_path(base_path, input_ls_path)
        self.sg_path = resolve_data_path(base_path, sg_path)
        self.resolution = resolution
        self.max_count = max_count
        self.save_hyperparameters()

    def dataset(self, targets=False):
        return SGNetDataset(
            self.input_path,
            self.input_ls_path,
            self.sg_path if targets else None,
            self.max_count,
        )

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.fit_data(self.dataset(targets=True))
        if stage == "test":
            self.test_data = self.dataset(targets=True)
            self.data_num = len(self.test_data)
        if stage == "predict":
            self.predict_data = self.dataset()
            self.data_num = len(self.predict_data)


def canonicalize_asg(asg):
    angle, lamb, mu, weight = torch.split(asg, [1, 1, 1, 3], dim=-1)
    swap_axes = lamb < mu
    angle = torch.where(swap_axes, angle + torch.pi / 2, angle)
    angle = torch.remainder(angle, torch.pi)
    lamb, mu = torch.maximum(lamb, mu), torch.minimum(lamb, mu)
    return torch.cat((angle, lamb, mu, weight), dim=-1)


class ASGNetDataset(Dataset):
    def __init__(
        self,
        input_path,
        ldr_path=None,
        asg_path=None,
        resolution=(512, 256),
        max_count=None,
    ):
        modalities = {"ldr": ldr_path, "asg": asg_path} if ldr_path is not None else {}
        self.records = limit(
            matched_paths(input_path, modalities, IMAGE_SUFFIXES), max_count
        )
        self.resolution = resolution

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        rgb = color_tensor(record["input"], PERS_RES)
        if "ldr" not in record:
            return rgb, record["name"]
        asg = torch.from_numpy(np.load(record["asg"]).astype(np.float32))
        asg = canonicalize_asg(asg)
        ldr = color_tensor(record["ldr"], self.resolution)
        return rgb, ldr, asg, record["name"]


class ASGNetDataModule(BaseDataModule):
    def __init__(
        self,
        base_path,
        input_path,
        ldr_path=None,
        asg_path=None,
        resolution=(512, 256),
        batch_size=1,
        max_count=None,
        num_workers=None,
    ):
        super().__init__(batch_size, num_workers)
        self.input_path = resolve_data_path(base_path, input_path)
        self.ldr_path = resolve_data_path(base_path, ldr_path)
        self.asg_path = resolve_data_path(base_path, asg_path)
        self.resolution = resolution
        self.max_count = max_count
        self.save_hyperparameters()

    def dataset(self, targets=False):
        return ASGNetDataset(
            self.input_path,
            self.ldr_path if targets else None,
            self.asg_path if targets else None,
            self.resolution,
            self.max_count,
        )

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.fit_data(self.dataset(targets=True))
        if stage == "test":
            self.test_data = self.dataset(targets=True)
            self.data_num = len(self.test_data)
        if stage == "predict":
            self.predict_data = self.dataset()
            self.data_num = len(self.predict_data)


def center_perspective_crop(panorama):
    size = panorama.shape[0]
    axis = np.linspace(-1, 1, size, dtype=np.float32)
    x, y = np.meshgrid(axis, axis[::-1])
    norm = np.sqrt(x * x + y * y + 1)
    u, v = world_to_image(x / norm, y / norm, -1 / norm)
    map_x = (u * panorama.shape[1] - 0.5).astype(np.float32)
    map_y = (v * panorama.shape[0] - 0.5).astype(np.float32)
    return cv.remap(panorama, map_x, map_y, cv.INTER_LINEAR, borderMode=cv.BORDER_WRAP)


def hdr_exposure(hdr, name):
    crop = center_perspective_crop(hdr)
    luminance = crop[..., :3] @ LUM_WEIGHT_BGR
    valid = luminance[luminance > 0]
    if not valid.size:
        raise ValueError(f"HDR center crop has no positive luminance: {name}")
    median = max(float(np.median(valid)), 1e-8)
    return np.float32(0.45**2.4 / median)


def ldr_from_hdr(hdr, name):
    exposure = hdr_exposure(hdr, name)
    mapped = np.minimum(hdr, 1 / exposure) * exposure
    mapped = linear_to_srgb(np.clip(mapped, 0, 1))
    ldr = np.round(np.clip(mapped, 0.5 / 255, 1) * 255).astype(np.uint8)
    return ldr, exposure


class HDRNetDataset(Dataset):
    def __init__(
        self,
        hdr_path,
        sg_path,
        ls_path,
        resolution=(512, 256),
        augment=False,
        max_hdr=1e4,
    ):
        self.records = matched_paths(
            hdr_path,
            {"sg": sg_path, "mask": ls_path},
            HDR_SUFFIXES,
        )
        self.resolution = resolution
        self.augment = augment
        self.max_hdr = max_hdr

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        hdr = read_image(record["input"])
        hdr = np.maximum(
            np.nan_to_num(hdr, nan=0.0, posinf=0.0, neginf=0.0),
            0.0,
        ).astype(np.float32)
        if hdr.ndim != 3 or hdr.shape[2] < 3:
            raise ValueError(
                f"HDRNet HDR must have at least 3 channels: {record['input']}"
            )
        hdr = hdr[..., :3]
        ldr, exposure = ldr_from_hdr(hdr, record["name"])
        ldr = cv.resize(ldr, self.resolution, interpolation=cv.INTER_AREA)
        ldr = torch.from_numpy(ldr.astype(np.float32) / 255).permute(2, 0, 1) * 2 - 1
        sg = load_sg_luminance(record["sg"])
        mask = mask_tensor(record["mask"], self.resolution, signed=True)
        hdr = np.minimum(hdr, self.max_hdr / exposure) * exposure
        hdr = np.clip(hdr, 1e-8, self.max_hdr)
        hdr = cv.resize(hdr, self.resolution, interpolation=cv.INTER_AREA)
        hdr = torch.from_numpy(np.log(hdr).astype(np.float32)).permute(2, 0, 1)
        do_flip = self.augment and random.random() < 0.5
        if do_flip:
            ldr, mask, hdr = [
                torch.flip(value, dims=[-1]) for value in (ldr, mask, hdr)
            ]
        return ldr, sg, mask, hdr, exposure, do_flip, record["name"]


class HDRNetPredictDataset(Dataset):
    def __init__(self, ldr_path, sg_path, ls_path, resolution=(512, 256)):
        self.records = matched_paths(
            ldr_path,
            {"sg": sg_path, "mask": ls_path},
            IMAGE_SUFFIXES,
        )
        self.resolution = resolution

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        return (
            color_tensor(record["input"], self.resolution, signed=True),
            load_sg_luminance(record["sg"]),
            mask_tensor(record["mask"], self.resolution, signed=True),
            record["name"],
        )


def source_pano_name(name):
    prefix, separator, suffix = name.rpartition("_")
    if separator and len(suffix) == 2 and suffix.isdigit():
        return f"{prefix}_00"
    return name


def read_name_list(path):
    path = Path(path).expanduser()
    names = {line.strip() for line in path.read_text(encoding="utf-8").splitlines()}
    names.discard("")
    if not names:
        raise ValueError(f"HDR training name list is empty: {path}")
    return names


def split_hdr_sources(records):
    groups = {}
    for index, record in enumerate(records):
        name = source_pano_name(record["name"])
        groups.setdefault(name, []).append(index)

    sources = sorted(groups)
    if len(sources) < 2:
        raise ValueError("HDRNet requires at least two source panoramas")
    order = torch.randperm(
        len(sources),
        generator=torch.Generator().manual_seed(42),
    ).tolist()
    split = int(len(order) * 0.95)
    train_sources = {sources[index] for index in order[:split]}
    val_sources = {sources[index] for index in order[split:]}
    train_indices = [index for name in sorted(train_sources) for index in groups[name]]
    val_indices = [index for name in sorted(val_sources) for index in groups[name]]
    return groups, train_indices, val_indices


def listed_train_indices(records, groups, train_indices, path, label):
    allowed = read_name_list(path)
    unknown = sorted(allowed - set(groups))
    if unknown:
        preview = ", ".join(unknown[:20])
        raise ValueError(f"HDR {label} training names do not match data: {preview}")
    selected = [
        index
        for index in train_indices
        if source_pano_name(records[index]["name"]) in allowed
    ]
    if not selected:
        raise ValueError(f"HDR {label} training list selected no training samples")
    return selected


class HDRNetDataModule(BaseDataModule):
    def __init__(
        self,
        base_path,
        sg_path,
        ls_path,
        hdr_path,
        resolution=(512, 256),
        batch_size=1,
        num_workers=None,
        max_hdr=1e4,
        train_list_path=None,
        medium_train_list_path=None,
        hard_train_list_path=None,
        low_epochs=10,
        medium_epochs=5,
        hard_epochs=5,
    ):
        super().__init__(batch_size, num_workers)
        self.sg_path = resolve_data_path(base_path, sg_path)
        self.ls_path = resolve_data_path(base_path, ls_path)
        self.hdr_path = resolve_data_path(base_path, hdr_path)
        self.resolution = resolution
        self.max_hdr = max_hdr
        self.train_list_path = train_list_path
        self.medium_train_list_path = medium_train_list_path
        self.hard_train_list_path = hard_train_list_path
        self.low_epochs = low_epochs
        self.medium_epochs = medium_epochs
        self.hard_epochs = hard_epochs
        self.train_stages = None
        self.save_hyperparameters()

    def dataset(self, augment=False):
        return HDRNetDataset(
            self.hdr_path,
            self.sg_path,
            self.ls_path,
            self.resolution,
            augment,
            self.max_hdr,
        )

    def setup(self, stage=None):
        if stage in (None, "fit"):
            dataset = self.dataset()
            groups, train_indices, val_indices = split_hdr_sources(dataset.records)
            augmented = self.dataset(augment=True)
            if self.train_list_path is not None:
                train_indices = listed_train_indices(
                    dataset.records,
                    groups,
                    train_indices,
                    self.train_list_path,
                    "override",
                )
                self.train_stages = None
                self.train_data = Subset(augmented, train_indices)
            elif self.medium_train_list_path is not None:
                medium_indices = listed_train_indices(
                    dataset.records,
                    groups,
                    train_indices,
                    self.medium_train_list_path,
                    "medium",
                )
                hard_indices = listed_train_indices(
                    dataset.records,
                    groups,
                    train_indices,
                    self.hard_train_list_path,
                    "hard",
                )
                self.train_stages = {
                    "low": Subset(augmented, train_indices),
                    "medium": Subset(augmented, medium_indices),
                    "hard": Subset(augmented, hard_indices),
                }
                self.train_data = self.train_stages["low"]
            else:
                self.train_stages = None
                self.train_data = Subset(augmented, train_indices)
            self.val_data = Subset(dataset, val_indices)
            self.data_num = [len(self.train_data), len(self.val_data)]
        if stage == "test":
            self.test_data = self.dataset()
            self.data_num = len(self.test_data)

    def train_stage(self, epoch):
        if epoch < self.low_epochs:
            return "low"
        if epoch < self.low_epochs + self.medium_epochs:
            return "medium"
        return "hard"

    def train_dataloader(self):
        if self.train_stages is None:
            return super().train_dataloader()
        trainer = getattr(self, "_trainer", None)
        epoch = trainer.current_epoch if trainer is not None else 0
        stage = self.train_stage(epoch)
        self.train_data = self.train_stages[stage]
        return self.loader(self.train_data, training=True)


def smoothstep(value, lower, upper):
    value = np.clip((value - lower) / (upper - lower), 0, 1)
    return value * value * (3 - 2 * value)


def id_ldr_from_hdr(hdr, exposure_ev=0.0):
    luminance = np.sum(hdr[..., :3] * LUM_WEIGHT_BGR, axis=-1)
    positive = luminance[luminance > 0]
    if not positive.size:
        return np.zeros_like(hdr, dtype=np.float32)
    exposure = 0.45**2.4 / np.median(positive)
    exposure *= 2**exposure_ev
    return np.clip(np.maximum(hdr * exposure, 0) ** (1 / 2.4), 0, 1)


def id_lum_from_hdr(hdr):
    luminance = np.sum(hdr[..., :3] * LUM_WEIGHT_BGR, axis=-1)
    valid = luminance > 0
    if not valid.any():
        return np.zeros((*luminance.shape, 1), dtype=np.float32)

    floor = float(luminance[valid].min())
    log_luminance = np.log(np.where(valid, luminance, floor)).astype(np.float32)
    saliency = cv.GaussianBlur(log_luminance, (0, 0), 1)
    lower, upper = np.percentile(saliency[valid], (85, 99.5))
    tolerance = np.finfo(np.float32).eps * max(abs(lower), abs(upper), 1)
    if upper - lower <= tolerance:
        target = valid.astype(np.float32) * 0.05
    else:
        target = smoothstep(saliency, lower, upper).astype(np.float32)
        target[~valid] = 0
    return target[..., None]


class IDNetDataset(Dataset):
    def __init__(
        self,
        path,
        resolution=512,
        max_count=None,
        targets=False,
        augment=False,
    ):
        suffixes = HDR_SUFFIXES if targets else IMAGE_SUFFIXES
        self.records = limit(matched_paths(path, {}, suffixes), max_count)
        self.resolution = resolution
        self.targets = targets
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        if not self.targets:
            return color_tensor(record["input"], self.resolution), record["name"]

        hdr = read_image(record["input"])
        hdr = np.maximum(
            np.nan_to_num(hdr, nan=0.0, posinf=0.0, neginf=0.0), 0.0
        ).astype(np.float32)
        if hdr.ndim != 3 or hdr.shape[2] < 3:
            raise ValueError(
                f"IDNet HDR must have at least 3 channels: {record['input']}"
            )
        hdr = hdr[..., :3]
        hdr = resize_image(hdr, self.resolution)
        exposure_ev = 0.0
        if self.augment:
            exposure_ev = random.uniform(-1, 1)
        ldr = id_ldr_from_hdr(hdr, exposure_ev)
        lum = id_lum_from_hdr(hdr)
        if self.augment and random.random() < 0.5:
            ldr = np.flip(ldr, axis=1).copy()
            lum = np.flip(lum, axis=1).copy()
        ldr = torch.from_numpy(ldr).permute(2, 0, 1)
        lum = torch.from_numpy(lum).permute(2, 0, 1)
        return ldr, lum, record["name"]


class IDNetDataModule(BaseDataModule):
    def __init__(
        self,
        hdr_path=None,
        input_path=None,
        resolution=512,
        batch_size=1,
        num_workers=None,
        max_count=None,
    ):
        super().__init__(batch_size, num_workers)
        self.hdr_path = Path(hdr_path).expanduser() if hdr_path is not None else None
        self.input_path = (
            Path(input_path).expanduser() if input_path is not None else None
        )
        self.resolution = resolution
        self.max_count = max_count
        self.save_hyperparameters()

    def dataset(self, targets=False, augment=False):
        path = self.hdr_path if targets else self.input_path
        return IDNetDataset(
            path,
            self.resolution,
            self.max_count,
            targets,
            augment,
        )

    def setup(self, stage=None):
        if stage in (None, "fit"):
            dataset = self.dataset(targets=True)
            indices = torch.randperm(
                len(dataset), generator=torch.Generator().manual_seed(42)
            ).tolist()
            split = int(len(indices) * 0.95)
            self.train_data = Subset(
                self.dataset(targets=True, augment=True), indices[:split]
            )
            self.val_data = Subset(dataset, indices[split:])
            self.data_num = [len(self.train_data), len(self.val_data)]
        if stage == "test":
            self.test_data = self.dataset(targets=True)
            self.data_num = len(self.test_data)
        if stage == "predict":
            self.predict_data = self.dataset()
            self.data_num = len(self.predict_data)
