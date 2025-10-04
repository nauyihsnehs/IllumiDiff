import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import pathlib
import random

import torch
from PIL import Image
from torchvision.transforms import GaussianBlur

import cv2 as cv
import numpy as np

from torch.utils.data import Dataset

lum_weight = np.asarray([0.2126, 0.7152, 0.0722])[None, None, ...]  # rgb


# def norm(tensor):
    # return (tensor - tensor.min()) / (tensor.max() - tensor.min())


class ControlLDMASGSGDataset(Dataset):
    def __init__(self, size, img_root, pano_root, asg_root, sg_root, start_idx=0):
        super().__init__()
        self.W = size
        self.H = size // 2
        self.img_path = img_root
        self.pano_path = pano_root
        self.sg_path = sg_root
        self.sg_pre_path = sg_root + '_pre'
        self.asg_path = asg_root
        self.asg_pre_path = asg_root + '_pre'
        asg_list = [p.as_posix() for p in pathlib.Path(self.asg_pre_path).glob('*.*')]
        asg_list.sort()
        self.asg_list = asg_list[start_idx:]

        self.blur = GaussianBlur(5, sigma=5)
        mask_path = 'pano_gen/outpainting-mask.png'
        mask = np.array(Image.open(mask_path).convert("L").resize((self.W, self.H), resample=0)) / 255.0
        self.mask = 1 - torch.from_numpy(mask).float()[..., None]

    def __len__(self):
        return len(self.asg_list)

    def __getitem__(self, i):
        random_lum = random.random() / 0.7 + 0.3 if random.randint(0, 100) < 50 else 1.0
        img_name = self.asg_list[i].split('/')[-1][:-4]

        img_path = list(pathlib.Path(self.img_path).glob(f'{img_name}.*'))[0]
        img_image = np.array(Image.open(img_path).convert("RGB").resize((224, 224))) / 255.0
        img = torch.from_numpy(img_image).permute(2, 0, 1).float()
        img = img * random_lum

        pano_path = list(pathlib.Path(self.pano_path).glob(f'{img_name}.*'))[0]
        pano_image = np.array(Image.open(pano_path).convert("RGB").resize((self.W, self.H))) / 255.0
        pano_image = torch.from_numpy(pano_image).float()
        pano = pano_image * random_lum  # random adjust the brightness

        sg_path = self.sg_path if random.randint(0, 100) < 50 else self.sg_pre_path
        sg_image = cv.imread(f'{sg_path}/{img_name}.exr', cv.IMREAD_UNCHANGED)
        sg_image = cv.resize(sg_image, (self.W, self.H))
        sg_image = cv.cvtColor(sg_image, cv.COLOR_BGR2RGB)
        sg_image = np.sum(sg_image * lum_weight, axis=-1, keepdims=True)
        sg = torch.from_numpy(sg_image).float()
        sg = sg * random_lum

        asg_path = self.asg_path if random.randint(0, 100) < 50 else self.asg_pre_path
        asg_image = np.array(Image.open(f'{asg_path}/{img_name}.jpg').convert("RGB").resize((self.W, self.H))) / 255.0
        asg = torch.from_numpy(asg_image).float()
        asg = asg * self.mask + pano_image * (1 - self.mask)
        asg = asg * random_lum  # random adjust the brightness
        asg = self.blur(asg.permute(2, 0, 1)).permute(1, 2, 0) if random.randint(0, 100) < 50 else asg
        asg = torch.concat((asg, self.mask), dim=-1)

        img = img * 2 - 1
        pano = pano * 2 - 1
        asg = asg * 2 - 1
        sg = sg - 1

        batch = {"img": img, "pano": pano, "hint": asg, "hdr": sg, "img_name": img_name}

        return batch
