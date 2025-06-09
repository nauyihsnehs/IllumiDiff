import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import math
import pathlib
import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

TINY = 1e-8
exr_save_params = [cv.IMWRITE_EXR_TYPE, cv.IMWRITE_EXR_TYPE_HALF, cv.IMWRITE_EXR_COMPRESSION, cv.IMWRITE_EXR_COMPRESSION_PIZ]


class ASGDataset(Dataset):
    def __init__(self, img_path, width, gpu_num, gpu_id):
        self.img_list = [p.as_posix() for p in pathlib.Path(img_path).glob('*.*')]
        self.img_list.sort()
        self.w = width
        self.h = int(width / 2)
        avg_num = len(self.img_list) / gpu_num
        start_num = int(np.floor(gpu_id * avg_num))
        end_num = np.ceil((gpu_id + 1) * avg_num)
        end_num = int(end_num) if end_num <= len(self.img_list) else len(self.img_list)
        self.img_list = self.img_list[start_num:end_num]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = cv.imread(self.img_list[idx]) / 255.
        # img = cv.imread(self.img_list[idx], cv.IMREAD_UNCHANGED)
        img = cv.resize(img, (self.w, self.h), interpolation=cv.INTER_AREA)
        img = cv.GaussianBlur(img, (11, 11), 5)
        img_name = self.img_list[idx].split('/')[-1][:-4]

        return {'img': img, 'img_name': img_name}


class ASGEnvOptim:
    def __init__(self, niter, envWidth, envHeight, ASGNum, bs=4, lr=0.2, ch=3, gpuId=0):
        self.ASGNum = ASGNum
        self.niter = niter
        self.ch = ch
        self.envHeight = envHeight
        self.envWidth = envWidth
        self.batch_size = bs
        self.loss = None

        fib_xyz = np.load(f'./fib_lobes_{ASGNum}.npy')
        fib_x = torch.from_numpy(fib_xyz[:, 0:1] + TINY).cuda()
        fib_y = torch.from_numpy(fib_xyz[:, 1:2] + TINY).cuda()
        fib_z = torch.from_numpy(fib_xyz[:, 2:] + TINY).cuda()
        self.fib_y = fib_y[None, ...].expand(self.batch_size, -1, -1)
        self.fib_z = fib_z[None, ...].expand(self.batch_size, -1, -1)
        self.fib_x = fib_x[None, ...].expand(self.batch_size, -1, -1)

        # Azimuth/Elevation to xyz
        Az = ((np.arange(envWidth)) / envWidth) * 2 * math.pi
        El = ((np.arange(envHeight)) / envHeight) * math.pi
        Az, El = np.meshgrid(Az, El)
        Az = Az[:, :, np.newaxis]
        El = El[:, :, np.newaxis]
        lx = np.sin(El) * np.cos(Az)
        ly = np.sin(El) * np.sin(Az)
        lz = np.cos(El)
        self.ls = np.concatenate((lx, ly, lz), axis=2)[np.newaxis, np.newaxis, np.newaxis, :].astype(np.float32)
        self.ls = torch.from_numpy(self.ls)
        self.ls = self.ls.expand([self.batch_size, self.ASGNum, 1, self.envHeight, self.envWidth, 3]).cuda(gpuId)

        weight = torch.randn(self.batch_size, ASGNum, ch) / 7.5
        lamb = torch.randn(self.batch_size, ASGNum, 1) * 20
        mu = torch.randn(self.batch_size, ASGNum, 1) * 20
        angle = torch.randn(self.batch_size, ASGNum, 1)

        self.param = torch.cat([angle, lamb, mu, weight], dim=-1).cuda()
        self.param.requires_grad = True

        self.optEnv = torch.optim.Adam([self.param, ], lr=lr)

        self.bi_angle = torch.ones(self.batch_size, ASGNum, 1).cuda() * math.pi / 2.

        self.zeros = torch.zeros(self.batch_size, self.ASGNum, self.ch, self.envHeight, self.envWidth).cuda()

    def render_asg(self, angle, lamb, mu, weight):
        nx = self.fib_x
        ny = self.fib_y
        nz = self.fib_z
        denom = torch.sqrt(nx * nx + nz * nz)
        tx = -1 * nx * ny / denom
        tz = -1 * ny * nz / denom
        ty = denom

        normal = torch.cat([nx, ny, nz], dim=-1)
        normal_tan = torch.cat([tx, ty, tz], dim=-1)
        tan = torch.cos(angle) * normal_tan + torch.sin(angle) * torch.cross(normal, normal_tan, dim=-1)
        bi_tan = torch.cos(self.bi_angle) * tan + torch.sin(self.bi_angle) * torch.cross(normal, tan, dim=-1)

        normal = normal[..., None, None, None, :]
        normal = normal.expand([-1, -1, 1, self.envHeight, self.envWidth, -1])
        tan = tan[..., None, None, None, :]
        tan = tan.expand([-1, -1, 1, self.envHeight, self.envWidth, -1])
        bi_tan = bi_tan[..., None, None, None, :]
        bi_tan = bi_tan.expand([-1, -1, 1, self.envHeight, self.envWidth, -1])
        lamb = lamb[..., None, None]
        lamb = lamb.expand([-1, -1, -1, self.envHeight, self.envWidth])
        mu = mu[..., None, None]
        mu = mu.expand([-1, -1, -1, self.envHeight, self.envWidth])
        weight = weight[..., None, None]
        weight = weight.expand([-1, -1, -1, self.envHeight, self.envWidth])

        e_power = lamb * (torch.sum(self.ls * tan, dim=-1) * torch.sum(self.ls * tan, dim=-1)) + mu * (torch.sum(self.ls * bi_tan, dim=-1) * torch.sum(self.ls * bi_tan, dim=-1))
        e_item = torch.exp(-1 * e_power)
        e_item = e_item.expand([self.batch_size, self.ASGNum, self.ch, self.envHeight, self.envWidth])
        smooth = torch.maximum(self.zeros, torch.sum(self.ls * normal, dim=-1))
        smooth = smooth.expand([self.batch_size, self.ASGNum, self.ch, self.envHeight, self.envWidth])

        envmaps = smooth * weight * e_item
        envmap = torch.sum(envmaps, dim=1)

        return envmap

    def de_parameterize(self):
        angle, lamb, mu, weight = torch.split(self.param.view(self.batch_size, self.ASGNum, 6), [1, 1, 1, 3], dim=-1)
        lamb = torch.abs(lamb)
        mu = torch.abs(mu)
        weight = torch.abs(weight)
        angle = angle * 2
        return angle, lamb, mu, weight

    def optimize(self, envmap):
        save_npy = None
        rec_map = None

        for i in range(self.niter):
            self.optEnv.zero_grad()
            an, la, mu, w = self.de_parameterize()
            panoImage = self.render_asg(an, la, mu, w)
            loss = ((panoImage - envmap) ** 2).view(panoImage.shape[0], -1).mean(1).sum(0)
            self.loss = loss
            loss.backward()
            self.optEnv.step()
            if torch.isnan(torch.sum(self.param)):
                return None
            if (i + 1) == self.niter:
                print(self.loss.item())
                rec_map = panoImage.permute(0, 2, 3, 1)
                rec_map = rec_map.cpu().detach().numpy().astype(np.float32)
                rec_map = np.clip(rec_map, 0, 1)
                #     cv.imwrite(f'{save_path}/{name}_iter{str(i).zfill(3)}.jpg', rec_map * 255).reshape(self.ASGNum, 1)
                save_npy = np.concatenate((an.cpu().detach().numpy(),
                                           la.cpu().detach().numpy(),
                                           mu.cpu().detach().numpy(),
                                           w.cpu().detach().numpy()), axis=-1)
                print(f'an: {an.mean().item()}, la: {la.mean().item()}, mu: {mu.mean().item()}, w: {w.mean().item()}')
        return rec_map, save_npy


# 256 b8 38 4.7
# 256 b16 75 4.7

def main(gpu_num, gpu_id):
    asg_dataset = ASGDataset(file_path, W, gpu_num, gpu_id)
    asg_dataloader = torch.utils.data.DataLoader(asg_dataset, batch_size=batch_size, shuffle=False)
    asg_view = ASGEnvOptim(envWidth=512, envHeight=256, ASGNum=asg_num, niter=500, lr=1e-1, bs=batch_size)
    for i, data in enumerate(tqdm(asg_dataloader)):
        save_names = data['img_name']
        pano_ori = data['img'].cuda()
        pano = pano_ori.permute(0, 3, 1, 2)
        asg = ASGEnvOptim(envWidth=W, envHeight=int(W / 2), ASGNum=asg_num, niter=500, lr=1e-1, bs=pano.shape[0])
        rec_maps, save_npys = asg.optimize(envmap=pano)
        an, la, mu, w = torch.split(torch.from_numpy(save_npys).cuda(), [1, 1, 1, 3], dim=-1)
        asg_imgs = asg_view.render_asg(an, la, mu, w).permute(0, 2, 3, 1).cpu().detach().numpy().astype(np.float32)
        asg_imgs = np.clip(asg_imgs, 0, 1) * 255
        for asg_img, save_npy, save_name in zip(asg_imgs, save_npys, save_names):
            np.save(f'{save_path}/{save_name}.npy', save_npy)
            cv.imwrite(f'{save_path}/{save_name}.jpg', asg_img)


if __name__ == '__main__':
    import sys

    file_path = ''
    save_path = ''
    W = 256
    batch_size = 2
    asg_num = 128

    main(int(sys.argv[1]), int(sys.argv[2]))
    # main(1, 0)
