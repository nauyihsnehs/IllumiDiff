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


class PureASGTrainLDMHDR(Dataset):
    def __init__(self, size, img_root, pano_root, asg_root, sg_root):
        super().__init__()
        self.W = size
        self.H = int(size / 2)
        self.img_path = img_root
        self.pano_path = pano_root
        self.sg_path = sg_root
        self.asg_path = asg_root
        self.asg_pre_path = asg_root + '-pre'
        self.asg_list = [p.as_posix() for p in pathlib.Path(self.asg_pre_path).glob('*.jpg')]
        self.asg_list.sort()

        self.blur = GaussianBlur(11, sigma=5)

    def __len__(self):
        return len(self.asg_list)

    def __getitem__(self, i):
        def norm(tensor):
            return (tensor - tensor.min()) / (tensor.max() - tensor.min())

        random_int = random.randint(0, 100)
        random_lum = random.random() / 0.7 + 0.3
        img_name = self.asg_list[i].split('/')[-1][:-4]

        img_image = np.array(Image.open(f'{self.img_path}/{img_name}.png').convert("RGB").resize((224, 224))) / 255.0
        img = torch.from_numpy(img_image).permute(2, 0, 1).float()
        img = norm(img)

        pano_image = np.array(
            Image.open(f'{self.pano_path}/{img_name}.png').convert("RGB").resize((self.W, self.H))) / 255.0
        pano = torch.from_numpy(pano_image).float()
        pano = pano ** random_lum
        pano = norm(pano)

        sg_image = cv.imread(f'{self.sg_path}/{img_name}.exr', cv.IMREAD_UNCHANGED)
        sg_image = cv.resize(sg_image, (self.W, self.H))
        sg_image = cv.cvtColor(sg_image, cv.COLOR_BGR2RGB)
        sg_image = np.sum(sg_image * lum_weight, axis=-1, keepdims=True)  # .repeat(3, -1)
        if sg_image.max() < 1:
            ninety_pixel = np.percentile(sg_image, 98)
            sg_image[sg_image < ninety_pixel] = 0
        sg_image[sg_image < 1] = 0
        sg_image[sg_image > 0] = 1
        sg = torch.from_numpy(sg_image).float()

        asg_path = self.asg_path if random_int < 10 else self.asg_pre_path
        asg_image = np.array(Image.open(f'{asg_path}/{img_name}.jpg').convert("RGB").resize((self.W, self.H))) / 255.0
        asg = torch.from_numpy(asg_image).float()
        asg = asg ** random_lum
        asg = self.blur(asg.permute(2, 0, 1)).permute(1, 2, 0) if random_int < 10 else asg
        asg = norm(asg)

        img = img * 2 - 1
        pano = pano * 2 - 1
        asg = asg * 2 - 1
        sg = sg * 2 - 1

        batch = {"img": img, "pano": pano, "hint": asg, "hdr": sg}

        return batch
