import os
from pathlib import Path

import numpy as np

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import lightning as L
from torch.utils.data import random_split, Dataset, DataLoader

import torch
from torchvision.transforms import GaussianBlur

import cv2 as cv

exr_save_params = [cv.IMWRITE_EXR_TYPE, cv.IMWRITE_EXR_TYPE_HALF, cv.IMWRITE_EXR_COMPRESSION,
                   cv.IMWRITE_EXR_COMPRESSION_PIZ]

PERS_RES = (256, 256)
CLIP_RES = (224, 224)
PANO_RES = (512, 256)


class BaseDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.predict_data, self.test_data, self.val_data, self.train_data = None, None, None, None
        self.batch_size = 16

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, drop_last=True, shuffle=True, num_workers=4,
                          persistent_workers=True)  #

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=4, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=4, persistent_workers=True)

    def predict_dataloader(self):
        return DataLoader(self.predict_data, batch_size=self.batch_size, num_workers=4, persistent_workers=True)


class SGNetDataset(Dataset):
    def __init__(self, rgb_path, input_ls_path=None, sg_path=None, env_path=None, resolution=(256, 256), max_count=None):
        input_list = [p.as_posix() for p in Path(rgb_path).glob('*.*')]
        input_list.sort()
        self.input_list = input_list if max_count is None else input_list[:max_count]
        self.rgb_path = rgb_path
        self.input_ls_path = input_ls_path
        self.sg_path = sg_path
        self.env_path = env_path
        self.res = resolution
        self.lum_weight = np.asarray([0.2126, 0.7152, 0.0722])[None, None, ...]

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        img_name = self.input_list[idx].split('/')[-1][:-4]

        ldr_img_path = [p.as_posix() for p in Path(self.rgb_path).glob(f'*{img_name}.*')][0]
        ldr_img = cv.imread(ldr_img_path)
        ldr_img = cv.resize(ldr_img, (256, 256), interpolation=cv.INTER_AREA) / 255.
        ldr_img = torch.from_numpy(ldr_img).float().permute(2, 0, 1) * 2 - 1

        # inference
        if self.input_ls_path is None:
            lum_img = np.sum(ldr_img * self.lum_weight, axis=-1, keepdims=True)
            mask_img = torch.from_numpy(lum_img).float().permute(2, 0, 1)
        else:
            mask_path = [p.as_posix() for p in Path(self.input_ls_path).glob(f'*{img_name}.*')][0]
            mask_img = cv.imread(mask_path)
            mask_img = cv.resize(mask_img, (256, 256), interpolation=cv.INTER_AREA) / 255.
            if mask_img.shape[-1] == 3:
                mask_img = mask_img[..., 0]
            mask_img = torch.from_numpy(mask_img).float()[..., None].permute(2, 0, 1)

        # testing
        if self.env_path is None:
            return ldr_img, mask_img, img_name

        sg_gt = np.load(f'{self.sg_path}/{img_name}.npy')
        sg_gt = torch.from_numpy(sg_gt)

        env_gt = cv.imread(f'{self.env_path}/{img_name}.exr', cv.IMREAD_UNCHANGED)
        env_gt = cv.resize(env_gt, self.res, interpolation=cv.INTER_AREA)
        env_gt = np.nan_to_num(env_gt, nan=0.0)
        lum = np.sum(env_gt * self.lum_weight, axis=-1, keepdims=True).repeat(3, -1)
        env_gt_clip = env_gt.copy()
        env_gt_clip[lum < 1] = 0.
        env_gt = torch.from_numpy(env_gt)
        env_gt_clip = torch.from_numpy(env_gt_clip)

        return ldr_img, mask_img, env_gt, env_gt_clip, sg_gt, img_name


class SGNetDataModule(BaseDataModule):
    def __init__(self, base_path, input_path, input_ls_path=None, hdr_path=None, sg_path=None, resolution=256,
                 batch_size=1, id_net_ckpt_path=None, max_count=None):  # , model_type, frames=24
        super().__init__()
        self.predict_data, self.test_data, self.val_data, self.train_data = None, None, None, None
        self.max_count = max_count
        self.data_num = None
        self.input_path = base_path + '/' + input_path
        if input_ls_path is not None:
            self.input_ls_path = base_path + '/' + input_ls_path
        if hdr_path is not None:
            self.hdr_path = base_path + '/' + hdr_path
            self.sg_path = base_path + '/' + sg_path
        self.id_net_ckpt_path = id_net_ckpt_path
        self.resolution = resolution
        self.batch_size = batch_size
        self.save_hyperparameters()

    def setup(self, stage: str):
        if stage == "fit" or stage is None:
            mtds = SGNetDataset(self.input_path, self.input_ls_path, self.sg_path, self.hdr_path, self.resolution)
            self.train_data, self.val_data = random_split(mtds, [0.95, 0.05], torch.Generator().manual_seed(42))
            self.data_num = [len(self.train_data), len(self.val_data)]

        if stage == "test":
            mtds = SGNetDataset(self.input_path, self.input_ls_path, self.sg_path, self.hdr_path, self.resolution)
            self.test_data = mtds
            self.data_num = len(self.test_data)

        if stage == "predict":
            mtds = SGNetDataset(self.input_path, self.input_ls_path, resolution=self.resolution, max_count=self.max_count)
            self.predict_data = mtds
            self.data_num = len(self.predict_data)


class ASGNetDataset(Dataset):
    def __init__(self, input_path, ldr_path=None, asg_path=None, resolution=(512, 256), max_count=None):
        input_list = [p.as_posix() for p in Path(input_path).glob('*.*')]
        input_list.sort()
        self.input_list = input_list if max_count is None else input_list[:max_count]
        self.ldr_path = ldr_path
        self.asg_path = asg_path
        self.res = resolution
        self.blur = GaussianBlur(5, sigma=5)

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        img_name = self.input_list[idx].split('/')[-1][:-4]
        ldr_img = cv.imread(self.input_list[idx])
        ldr_img = cv.resize(ldr_img, (256, 256), interpolation=cv.INTER_AREA) / 255.
        ldr_img = torch.from_numpy(ldr_img).float().permute(2, 0, 1)

        if self.ldr_path is None or self.asg_path is None:
            # return {'input_img': ldr_img, 'img_name': img_name, }
            return ldr_img, img_name

        asg_gt = np.load(f'{self.asg_path}/{img_name}.npy')
        asg_gt = torch.from_numpy(asg_gt)

        ldr_gt = cv.imread(f'{self.ldr_path}/{img_name}.jpg')
        ldr_gt = cv.resize(ldr_gt, self.res, interpolation=cv.INTER_AREA) / 255.
        ldr_gt = torch.from_numpy(ldr_gt).float()
        ldr_gt = self.blur(ldr_gt.permute(2, 0, 1))

        return ldr_img, ldr_gt, asg_gt, img_name


class ASGNetDataModule(BaseDataModule):
    def __init__(self, base_path, input_path, ldr_path=None, asg_path=None, resolution=(512, 256),
                 batch_size=1, max_count=None):  # , model_type, frames=24
        super().__init__()
        self.data_num = None
        self.max_count = max_count
        self.predict_data, self.test_data, self.val_data, self.train_data = None, None, None, None
        self.input_path = base_path + '/' + input_path
        if ldr_path is not None:
            self.ldr_path = base_path + '/' + ldr_path
            self.asg_path = base_path + '/' + asg_path
        self.resolution = resolution
        self.batch_size = batch_size
        self.save_hyperparameters()

    def setup(self, stage: str):
        if stage == "fit" or stage is None:
            mtds = ASGNetDataset(self.input_path, self.ldr_path, self.asg_path, self.resolution)
            self.train_data, self.val_data = random_split(mtds, [0.95, 0.05], torch.Generator().manual_seed(42))
            self.data_num = [len(self.train_data), len(self.val_data)]

        if stage == "test":
            mtds = ASGNetDataset(self.input_path, resolution=self.resolution)
            self.test_data = mtds
            self.data_num = len(self.test_data)

        if stage == "predict":
            mtds = ASGNetDataset(self.input_path, resolution=self.resolution, max_count=self.max_count)
            self.predict_data = mtds
            self.data_num = len(self.predict_data)


class HDRNetDataset(Dataset):
    def __init__(self, ldr_path, sg_path=None, ls_path=None, hdr_path=None, resolution=(512, 256), sg_scale=1.0):
        input_list = [p.as_posix() for p in Path(ldr_path).glob('*.*')]
        input_list.sort()
        self.input_list = input_list
        self.ldr_path = ldr_path
        self.sg_path = sg_path
        self.ls_path = ls_path
        self.hdr_path = hdr_path
        self.res = resolution
        self.sg_scale = sg_scale
        self.lum_weight = np.asarray([0.0722, 0.7152, 0.2126])[None, None, ...]  # rgb

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        img_name = self.input_list[idx].split('/')[-1][:-4]
        ldr_img_path = [p.as_posix() for p in Path(self.ldr_path).glob(f'*{img_name}.*')][0]
        ldr_img = cv.imread(ldr_img_path)
        ldr_img = cv.resize(ldr_img, self.res, interpolation=cv.INTER_AREA) / 255.
        ldr_img = torch.from_numpy(ldr_img).float().permute(2, 0, 1)

        if self.ls_path is None: return ldr_img, img_name

        mask_img = cv.imread(f'{self.ls_path}/{img_name}.png')[..., 0]
        mask_img = cv.resize(mask_img, self.res, interpolation=cv.INTER_AREA) / 255.
        mask_img = torch.from_numpy(mask_img).float()[None]

        sg_img = cv.imread(f'{self.sg_path}/{img_name}.exr', cv.IMREAD_UNCHANGED)
        sg_img = cv.resize(sg_img, self.res, interpolation=cv.INTER_AREA)
        sg_img = np.nan_to_num(sg_img, nan=0.0)
        sg_lum = np.sum(sg_img * self.lum_weight, axis=-1, keepdims=True).repeat(3, -1)
        sg_img[sg_lum < 1] = 0.
        sg_img = np.log1p(sg_img * self.sg_scale)
        sg_img = torch.from_numpy(sg_img).float().permute(2, 0, 1)

        if self.hdr_path is None: return ldr_img, sg_img, mask_img, img_name

        hdr_gt = cv.imread(f'{self.hdr_path}/{img_name}.exr', cv.IMREAD_UNCHANGED)
        hdr_gt = cv.resize(hdr_gt, self.res, interpolation=cv.INTER_AREA)
        hdr_gt = np.nan_to_num(hdr_gt, nan=0.0)
        hdr_gt = np.log1p(hdr_gt)
        hdr_gt = torch.from_numpy(hdr_gt).float().permute(2, 0, 1)

        return ldr_img, sg_img, mask_img, hdr_gt, img_name  # hdr_gt_mask


class HDRNetDataModule(BaseDataModule):
    def __init__(self, base_path, input_path='hdr', sg_path=None, ls_path=None, hdr_path=None, resolution=(512, 256), sg_scale=1.0, batch_size=1):
        super().__init__()
        self.predict_data, self.test_data, self.val_data, self.train_data = None, None, None, None
        self.data_num = None
        self.input_path = base_path + '/' + input_path
        self.sg_path = base_path + '/' + sg_path if sg_path is not None else None
        self.ls_path = base_path + '/' + ls_path if ls_path is not None else None
        self.hdr_path = base_path + '/' + hdr_path if hdr_path is not None else None
        self.resolution = resolution
        self.sg_scale = sg_scale
        self.batch_size = batch_size
        self.save_hyperparameters()

    def setup(self, stage: str):
        if stage == "fit" or stage is None:
            mtds = HDRNetDataset(self.input_path, self.sg_path, self.ls_path, self.hdr_path, self.resolution)
            self.train_data, self.val_data = random_split(mtds, [0.95, 0.05], torch.Generator().manual_seed(42))
            self.data_num = [len(self.train_data), len(self.val_data)]

        if stage == "test":
            mtds = HDRNetDataset(self.input_path, self.sg_path, self.ls_path, self.hdr_path, self.resolution)
            self.test_data = mtds
            self.data_num = len(self.test_data)

        if stage == "predict":
            mtds = HDRNetDataset(self.input_path, self.sg_path, self.ls_path, resolution=self.resolution, sg_scale=self.sg_scale)
            self.predict_data = mtds
            self.data_num = len(self.predict_data)


class IDNetDataset(Dataset):
    def __init__(self, ldr_path, ls_path=None, resolution=(512, 256)):
        input_list = [p.as_posix() for p in Path(ldr_path).glob('*.*')]
        input_list.sort()
        self.input_list = input_list
        self.ldr_path = ldr_path
        self.ls_path = ls_path
        self.res = resolution
        self.lum_weight = np.asarray([0.0722, 0.7152, 0.2126])[None, None, ...]  # rgb

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        img_name = self.input_list[idx].split('/')[-1][:-4]  # .split('_')[0]
        ldr_img_path = [p.as_posix() for p in Path(self.ldr_path).glob(f'*{img_name}.*')][0]
        ldr_img = cv.imread(ldr_img_path) / 255.
        ldr_img = cv.resize(ldr_img, self.res, interpolation=cv.INTER_AREA)
        lum_img = np.sum(ldr_img * self.lum_weight, axis=-1, keepdims=True)
        ldr_img = torch.from_numpy(ldr_img).float().permute(2, 0, 1)

        if self.ls_path is None:
            return ldr_img, img_name

        lum_img = torch.from_numpy(lum_img).float().permute(2, 0, 1)
        mask_img = cv.imread(f'{self.ls_path}/{img_name}.png')[..., 0] / 255.
        mask_img = cv.resize(mask_img, self.res, interpolation=cv.INTER_AREA)
        mask_img = torch.from_numpy(mask_img).float()[None]

        return ldr_img, mask_img, lum_img, img_name


class IDNetDataModule(BaseDataModule):
    def __init__(self, base_path, input_path, ls_path=None, resolution=(512, 256), batch_size=1):
        super().__init__()
        self.predict_data, self.test_data, self.val_data, self.train_data = None, None, None, None
        self.data_num = None
        self.input_path = base_path + '/' + input_path
        if ls_path: self.ls_path = base_path + '/' + ls_path
        self.resolution = resolution
        self.batch_size = batch_size
        self.save_hyperparameters()

    def setup(self, stage: str):
        if stage == "fit" or stage is None:
            mtds = IDNetDataset(self.input_path, self.ls_path, self.resolution)
            self.train_data, self.val_data = random_split(mtds, [0.95, 0.05], torch.Generator().manual_seed(42))
            self.data_num = [len(self.train_data), len(self.val_data)]

        if stage == "test":
            mtds = IDNetDataset(self.input_path, self.ls_path, self.resolution)
            self.test_data = mtds
            self.data_num = len(self.test_data)

        if stage == "predict":
            mtds = IDNetDataset(self.input_path, resolution=self.resolution)
            self.predict_data = mtds
            self.data_num = len(self.predict_data)


class PipeDataset(Dataset):
    def __init__(self, input_path, input_pano_path, resolution=(512, 256), task='stage3', is_full_pipe=False):
        self.input_list = sorted(Path(input_path).glob('*.*'))
        self.input_pano_path = input_pano_path
        self.res = resolution
        self.task = task
        self.is_full_pipe = is_full_pipe

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        path = self.input_list[idx]
        img_name = path.stem
        ldr_img = cv.imread(path.as_posix())
        ldr_img = cv.resize(ldr_img, PERS_RES, interpolation=cv.INTER_AREA) / 255.
        ldr_img = torch.from_numpy(ldr_img).float().permute(2, 0, 1)

        if self.is_full_pipe:
            return ldr_img, img_name

        if self.task in ('stage3', 'full'):
            pano_path = [p.as_posix() for p in Path(self.input_pano_path).glob(f'*{img_name}*')][0]
            pano_img = cv.imread(pano_path, cv.IMREAD_UNCHANGED)
            pano_img = cv.resize(pano_img, self.res, interpolation=cv.INTER_AREA) / 255.
            pano_img = np.nan_to_num(pano_img, nan=0.0)
            pano_img = torch.from_numpy(pano_img).float().permute(2, 0, 1)
            return ldr_img, pano_img, img_name
        else:
            return ldr_img, [], img_name


class PipeDataModule(BaseDataModule):
    def __init__(self, batch_size=1, **kwargs):
        super().__init__()
        self.data_params = kwargs
        self.batch_size = batch_size
        self.save_hyperparameters()

    def setup(self, stage: str):
        mtds = PipeDataset(**self.data_params)
        self.predict_data = mtds


if __name__ == '__main__':
    pass
