import os
import pathlib

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2 as cv
import numpy as np
import torch

exr_save_params = [cv.IMWRITE_EXR_TYPE, cv.IMWRITE_EXR_TYPE_HALF, cv.IMWRITE_EXR_COMPRESSION,
                   cv.IMWRITE_EXR_COMPRESSION_PIZ]
TINY = 1e-8

file_path = '/mnt/data2/ssy/illumidiff_dataset/pano_hdr_128'
save_path = 'results/sg_fitting'
TINY_NUMBER = 1e-8
SGNum = 12


class SGEnvOptim:
    def __init__(self, niter, envWidth, envHeight, SGNum, lr=0.2, ch=3):
        self.SGNum = SGNum
        self.niter = niter
        self.ch = ch
        self.envHeight = envHeight
        self.envWidth = envWidth
        self.min_loss = 1e8
        self.min_params = None

        phi, theta = torch.meshgrid([torch.linspace(0., np.pi, self.envHeight), torch.linspace(0., 2 * np.pi, self.envWidth)], indexing='ij')
        view_dirs = torch.stack([torch.cos(theta) * torch.sin(phi), torch.cos(phi), torch.sin(theta) * torch.sin(phi)], dim=-1)

        self.ls = view_dirs[None, :].float()
        self.ls = self.ls.expand([self.SGNum, -1, -1, -1]).cuda()

        p = torch.randn(SGNum, 3)
        p = p / p.norm(dim=-1, keepdim=True)
        weight = torch.randn(SGNum, ch) * 1
        lamb = torch.randn(SGNum, 1) * 20

        self.param = torch.cat([p, lamb, weight], dim=1)
        self.param = self.param.view(SGNum * 7).cuda()
        self.param.requires_grad = True

        self.optEnv = torch.optim.NAdam([self.param, ], lr=lr)

    def render_sg(self, position, lamb, weight):
        la = lamb[:, None, None, ...].repeat(1, self.envHeight, self.envWidth, 1)
        lobes = position[:, None, None, ...].repeat(1, self.envHeight, self.envWidth, 1)

        w = weight[:, None, None, ...].repeat(1, self.envHeight, self.envWidth, 1)

        rgb = w * torch.exp(la * (torch.sum(self.ls * lobes, dim=-1, keepdim=True) - 1.))
        rgb = torch.sum(rgb, dim=0)
        return rgb

    def de_parameterize(self):
        p, lamb, weight = torch.split(self.param.view(self.SGNum, 7), [3, 1, 3], dim=1)
        p = p / p.norm(dim=-1, keepdim=True)
        lamb = torch.abs(lamb)
        weight = torch.abs(weight)
        return p, lamb, weight

    def optimize(self, envmap, name=None):
        save_npy = None
        rec_map = None
        envmap = torch.from_numpy(envmap).cuda().float()
        l2 = torch.nn.MSELoss(reduction='mean')

        for i in range(self.niter):
            self.optEnv.zero_grad()
            p, la, w = self.de_parameterize()
            panoImage = self.render_sg(p, la, w)
            loss = l2(panoImage, envmap)
            loss.backward()
            self.optEnv.step()
            if torch.isnan(torch.sum(self.param)):
                return None
            if (i + 1) > self.niter - 100:
                if loss.item() < self.min_loss:
                    self.min_loss = loss.item()
                    self.min_params = [p.clone(), la.clone(), w.clone()]
            if (i + 1) == self.niter:
                panoImage = self.render_sg(*self.min_params)
                rec_map = panoImage.cpu().detach().numpy().astype(np.float32)
                save_npy = torch.cat(self.min_params, dim=-1).cpu().detach().numpy()
                print(name, self.min_loss, rec_map.max())

        return rec_map, save_npy, self.min_loss


def main(gpu_num, gpu_id):
    ldr_list = [p.as_posix() for p in pathlib.Path(file_path).glob('*.exr')]
    ldr_list.sort()
    avg_num = len(ldr_list) / gpu_num
    start_num = int(np.floor(gpu_id * avg_num))
    end_num = np.ceil((gpu_id + 1) * avg_num)
    end_num = int(end_num) if end_num <= len(ldr_list) else len(ldr_list)
    lum_weight = np.asarray([0.0722, 0.7152, 0.2126])[None, None]
    print(start_num, end_num)
    for idx, ldr_dir in enumerate(ldr_list[start_num:end_num]):
        save_name = ldr_dir.split('/')[-1][:-4]
        pano_ori = cv.imread(ldr_dir, cv.IMREAD_UNCHANGED)
        lum = (np.sum(pano_ori * lum_weight, -1, keepdims=True)).repeat(3, axis=-1)
        lum99 = np.percentile(lum, 99)
        lum99 = min(lum99, 10)
        pano_ori[lum < lum99] = 0
        if pathlib.Path(f'{save_path}/{save_name}.exr').is_file():
            continue
        sg = SGEnvOptim(envWidth=128, envHeight=64, SGNum=SGNum, niter=10000, lr=1e-1)
        try:
            rec_map, save_npy, loss = sg.optimize(envmap=pano_ori, name=save_name)
        except:
            print(f'Error: {save_name}')
            continue
        cv.imwrite(f'{save_path}/{save_name}.exr', rec_map, exr_save_params)
        rec_map = np.clip(rec_map ** (1 / 2.2), 0, 1) * 255
        cv.imwrite(f'{save_path}/{save_name}.jpg', rec_map)
        np.save(f'{save_path}/{save_name}.npy', save_npy)


if __name__ == '__main__':
    import sys

    main(int(sys.argv[1]), int(sys.argv[2]))
    # main(1, 0)
