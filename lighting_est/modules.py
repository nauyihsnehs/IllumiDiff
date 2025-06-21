import os, sys

root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root)

import numpy as np
from scipy.optimize import linear_sum_assignment

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import lightning as L
import torch
from torch import optim, nn
import cv2 as cv

from models import HDRNet, SGNet, IDNet, ASGNet
from torch.nn import functional as F
from torchvision import models, transforms

torch.set_float32_matmul_precision('medium')
TINY_NUMBER = 1e-8
exr_save_params = [cv.IMWRITE_EXR_TYPE, cv.IMWRITE_EXR_TYPE_HALF, cv.IMWRITE_EXR_COMPRESSION,
                   cv.IMWRITE_EXR_COMPRESSION_PIZ]


class VGGLoss(nn.Module):
    models = {'vgg16': models.vgg16, 'vgg19': models.vgg19}

    def __init__(self, model='vgg19', layer=8, shift=0, reduction='mean'):
        super().__init__()
        self.shift = shift
        self.reduction = reduction
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.model = self.models[model](weights='DEFAULT').features[:layer + 1]
        self.model.eval()
        self.model.requires_grad_(False)

    def get_features(self, x):
        return self.model(self.normalize(x))

    def train(self, mode=True):
        self.training = mode

    def forward(self, x, target, target_is_features=False):
        if x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2)
        if target.shape[1] != 3:
            target = target.permute(0, 3, 1, 2)
        if target_is_features:
            input_feats = self.get_features(x)
            target_feats = target
        else:
            sep = x.shape[0]
            batch = torch.cat([x, target])
            feats = self.get_features(batch)
            input_feats, target_feats = feats[:sep], feats[sep:]
        return F.mse_loss(input_feats, target_feats, reduction=self.reduction)


class BaseModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.data_num = None

    def get_data_num(self, data_num):
        self.data_num = data_num

    def training_step(self, batch, batch_idx):
        return self(batch, batch_idx, 'train', self.data_num[0])

    def validation_step(self, batch, batch_idx):
        return self(batch, batch_idx, 'val', self.data_num[1])

    def predict_step(self, batch, batch_idx):
        self.inference(*batch)
        return []

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        return [optimizer], [scheduler]


class IDNetModule(BaseModule):
    def __init__(self, img_log_dir=None, learning_rate=1e-4):
        super().__init__()
        self.img_log_dir = img_log_dir
        self.model = IDNet(3, 1)

        self.learning_rate = learning_rate
        self.save_hyperparameters()
        self.example_input_array = [[torch.Tensor(1, 3, 256, 512),
                                     torch.Tensor(1, 1, 256, 512),
                                     torch.Tensor(1, 1, 256, 512),
                                     'name'], 0, 'inference', 100]

        self.bce_loss = nn.BCEWithLogitsLoss()

    def log_image(self, epoch, idx_max, mask_pre, mask_gt, ldr_img, img_name, stage, batch_idx):
        save_idx = np.random.randint(0, idx_max)
        mask_pre_save = mask_pre[save_idx].detach().cpu().permute(1, 2, 0)
        mask_pre_save = torch.sigmoid(mask_pre_save).numpy()
        mask_pre_save[mask_pre_save > 0.5] = 1
        mask_pre_save[mask_pre_save <= 0.5] = 0
        mask_pre_save = np.clip(mask_pre_save, 0, 1) * 255
        mask_gt_save = mask_gt[save_idx].detach().cpu().permute(1, 2, 0).numpy() * 255
        ldr_img_save = ldr_img[save_idx].detach().cpu().permute(1, 2, 0).numpy() * 255
        grid = np.concatenate((ldr_img_save, mask_pre_save, mask_gt_save), axis=1)
        cv.imwrite(f'{self.img_log_dir}/{stage}/e{str(epoch).zfill(3)}_b{batch_idx}_{img_name[save_idx]}_mask_grid.jpg', grid.astype(np.uint8))

    def inference(self, ldr_imgs, img_names=None, is_save=True):
        masks = self.model(ldr_imgs)
        masks = torch.sigmoid(masks)
        masks[masks > 0.5] = 1
        masks[masks <= 0.5] = 0
        masks = torch.clamp(masks, 0, 1)
        if is_save:
            mask_saves = masks.detach().cpu().permute(0, 2, 3, 1).numpy() * 255
            for mask_save, img_name in zip(mask_saves, img_names):
                cv.imwrite(f'{self.img_log_dir}/{img_name}.png', mask_save.astype(np.uint8))
                print(f'id_net inference: {img_name}')
        return masks

    def forward(self, batch, batch_idx, stage, data_num):
        ldr_img, mask_gt, lum_img, img_name = batch
        bs = ldr_img.size(0)
        sbs = 2 if data_num // bs < 2 else data_num // bs
        mask_pre = self.model(ldr_img)
        if (batch_idx + 1) % sbs == 0:
            self.log_image(self.current_epoch, bs, mask_pre, mask_gt, lum_img, img_name, stage, batch_idx)
        if stage not in ['train', 'val', 'test']:
            return mask_pre
        bce_loss = self.bce_loss(mask_pre, mask_gt)
        log_info = {'bce': bce_loss}
        self.log_dict(log_info, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, batch_size=bs)
        return bce_loss


class HDRNetModule(BaseModule):
    def __init__(self, img_log_dir=None, learning_rate=1e-4):
        super().__init__()
        self.img_log_dir = img_log_dir
        self.model = HDRNet(4, 3)

        self.learning_rate = learning_rate
        self.save_hyperparameters()
        # self.example_input_array = torch.Tensor(1, 6, 240, 320)
        self.example_input_array = [
            [torch.Tensor(1, 3, 256, 512),
             torch.Tensor(1, 3, 256, 512),
             torch.Tensor(1, 1, 256, 512),
             torch.Tensor(1, 3, 256, 512),
             'name'], 0, 'inference', 100
        ]

        self.vgg_loss = VGGLoss()
        self.mse_loss = nn.MSELoss()

    def log_image(self, epoch, idx_max, hdr_pre, hdr_gt, sg_img, mask_img, img_name, stage, batch_idx):
        save_idx = np.random.randint(0, idx_max)
        hdr_save = torch.expm1(hdr_pre[save_idx]).detach().cpu().permute(1, 2, 0).numpy()
        hdr_gt_save = torch.expm1(hdr_gt[save_idx]).detach().cpu().permute(1, 2, 0).numpy()
        sg_img = torch.expm1(sg_img[save_idx]).detach().cpu().permute(1, 2, 0).numpy()
        mask_img = mask_img[save_idx].detach().cpu().permute(1, 2, 0).numpy().repeat(3, axis=-1)
        grid0 = np.concatenate((hdr_save, hdr_gt_save), axis=0)
        grid1 = np.concatenate((sg_img, mask_img), axis=0)
        grid = np.concatenate((grid0, grid1), axis=1)
        cv.imwrite(f'{self.img_log_dir}/{stage}/e{str(epoch).zfill(3)}_b_{batch_idx}_{img_name[save_idx]}_hdr_grid.exr',
                   grid, exr_save_params)

    def scale_invariant_l1(self, pred, target):
        B = pred.size(0)
        pred_flat = pred.view(B, -1)
        target_flat = target.view(B, -1)

        ratio = target_flat / (pred_flat.abs().clamp(min=1e-6))
        alpha = ratio.median(dim=1).values.view(B, 1, 1, 1)

        loss = (alpha * pred - target).abs().mean()
        return loss

    def get_loss(self, hdr_pre, hdr_gt, mask_img=None):
        ldr_loss = self.scale_invariant_l1(hdr_pre, hdr_gt)
        lum_loss = self.mse_loss(hdr_pre * mask_img, hdr_gt * mask_img)
        vgg_loss = self.vgg_loss(hdr_pre, hdr_gt)
        total_loss = 10 * lum_loss + 0.1 * ldr_loss + 0.01 * vgg_loss
        log_info = {'tl': total_loss, 'lum': lum_loss, 'ldr': ldr_loss, 'vgg': vgg_loss}
        return total_loss, log_info

    def inference(self, rgb_imgs, sg_imgs, mask_imgs, img_names, is_save=True):
        hdr_pres = self.model(rgb_imgs, sg_imgs, mask_imgs)
        hdr_pres = torch.expm1(hdr_pres)
        if is_save:
            hdr_saves = hdr_pres.detach().cpu().permute(0, 2, 3, 1).numpy()
            for hdr_save, img_name in zip(hdr_saves, img_names):
                cv.imwrite(f'{self.img_log_dir}/{img_name}.exr', hdr_save, exr_save_params)
                print(f'hdr_net inference: {img_name}, max: {hdr_save.max()}, min: {hdr_save.min()}, range: {hdr_save.max() / hdr_save.min()}')
        return hdr_pres

    def forward(self, batch, batch_idx, stage, data_num):
        rgb_img, sg_img, mask_img, hdr_gt, img_name = batch
        bs = rgb_img.size(0)
        sbs = max(data_num / bs // 5, 2)
        hdr_pre = self.model(rgb_img, sg_img, mask_img)
        if (batch_idx + 1) % sbs == 0:
            self.log_image(self.current_epoch, bs, hdr_pre, hdr_gt, sg_img, mask_img, img_name, stage, batch_idx)
        if stage not in ['train', 'val', 'test']: return hdr_pre
        total_loss, log_info = self.get_loss(hdr_pre, hdr_gt, mask_img)
        self.log_dict(log_info, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, batch_size=bs)
        return total_loss


class SGNetModule(BaseModule):
    def __init__(self, img_log_dir=None, resolution=(512, 256), learning_rate=1e-4):
        super().__init__()
        self.img_log_dir = img_log_dir
        self.sg_param_num = 7
        self.sg_num = 12
        self.model = SGNet(self.sg_num)
        self.mse_loss = nn.MSELoss()
        lum_weight = torch.tensor([0.0722, 0.7152, 0.2126])[None, None, None]

        self.width, self.height = resolution
        phi, theta = torch.meshgrid([torch.linspace(0., torch.pi, 128), torch.linspace(0, 2 * torch.pi, 256)], indexing='ij')
        view_dirs = torch.stack([torch.cos(theta) * torch.sin(phi), torch.cos(phi), torch.sin(theta) * torch.sin(phi)], dim=-1)  # [H, W, 3]
        ls = view_dirs[None, None, :].float()
        self.register_buffer('ls', ls)
        self.register_buffer('lum_weight', lum_weight)

        self.learning_rate = learning_rate
        self.save_hyperparameters()
        self.example_input_array = [
            [torch.Tensor(1, 3, 256, 256),
             torch.Tensor(1, 1, 256, 256),
             torch.Tensor(1, 3, 256, 512),
             torch.Tensor(1, 1, 256, 512),
             torch.Tensor(1, 12, 7),
             'name'], 0, 'inference', 100
        ]  # rgb_img, mask_img, env_gt, sg_gt, img_name

    def log_image(self, epoch, idx_max, env_pre, env_gt, img_name, stage, batch_idx):
        save_idx = np.random.randint(0, idx_max)
        # env_save = torch.expm1(env_pre[save_idx]).detach().cpu().numpy()  # .permute(1, 2, 0)
        env_save = env_pre[save_idx].detach().cpu().numpy()  # .permute(1, 2, 0)
        # env_gt_save = env_gt[save_idx].detach().cpu().numpy()
        env_gt_save = env_gt[save_idx].detach().cpu().numpy()
        # if env_sg is not None:
        #     env_sg_save = env_sg[save_idx].detach().cpu().numpy()
        #     grid = np.concatenate((env_save, env_gt_save, env_sg_save), axis=0)
        # else:
        grid = np.concatenate((env_save, env_gt_save), axis=0)
        grid_ldr = np.clip(grid ** (1 / 2.2), 0, 1) * 255
        cv.imwrite(f'{self.img_log_dir}/{stage}/e{str(epoch).zfill(3)}_b{batch_idx}_{img_name[save_idx]}_grid.exr',
                   grid, exr_save_params)
        cv.imwrite(f'{self.img_log_dir}/{stage}/e{str(epoch).zfill(3)}_b{batch_idx}_{img_name[save_idx]}_grid.jpg',
                   grid_ldr.astype(np.uint8))

    def get_sg_pair_loss(self, bs, tp_pre, tp_gt, la_pre=None, w_pre=None, la_gt=None, w_gt=None):
        index_tensor = torch.zeros((bs, self.sg_num), dtype=torch.int64).cuda()

        for batch in range(bs):
            cost_matrix = torch.cdist(tp_pre[batch], tp_gt[batch], p=2).cpu().detach().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            index_tensor[batch] = torch.tensor(col_ind).cuda()

        tp_paired = torch.stack([tp_gt[i, index_tensor[i]] for i in range(bs)])
        tp_loss = self.mse_loss(tp_pre, tp_paired)

        if la_pre is None or w_pre is None or la_gt is None or w_gt is None:
            return tp_loss

        la_paired = torch.stack([la_gt[i, index_tensor[i]] for i in range(bs)])
        w_paired = torch.stack([w_gt[i, index_tensor[i]] for i in range(bs)])

        la_loss = self.mse_loss(la_pre, 1 / (la_paired + 1e-8))
        w_loss = self.mse_loss(torch.log1p(w_pre), torch.log1p(w_paired))
        return tp_loss + la_loss + w_loss

    def get_lum_loss(self, env_pre, env_gt):
        lum_pre = torch.sum(env_pre * self.lum_weight, dim=-1)
        lum_gt = torch.sum(env_gt * self.lum_weight, dim=-1)
        return self.mse_loss(lum_pre, lum_gt)

    def sg2env(self, params, ls=None, is_gt=False):
        ls = self.ls if ls is None else ls
        p = params[0]
        la = 1 / (params[1] + 1e-8)
        w = params[2]

        p = p[..., None, None, :]
        la = la[..., None, None, :]
        w = w[..., None, None, :]

        rgb = w * torch.exp(la * (torch.sum(ls * p, dim=-1, keepdim=True) - 1.))
        rgb = torch.sum(rgb, dim=1)

        rgb = torch.clamp(rgb, TINY_NUMBER)
        if is_gt:
            max_rgb = rgb.max(dim=-1, keepdim=True)[0]
            mask = max_rgb < 1
            rgb[mask.expand_as(rgb)] = 0
        return rgb.float()

    def inference(self, rgb_imgs, mask_imgs, img_names, is_clip=False, is_save=True, is_tonemap=True):
        phi, theta = torch.meshgrid([torch.linspace(0., torch.pi, self.height), torch.linspace(0, 2 * torch.pi, self.width)], indexing='ij')
        view_dirs = torch.stack([torch.cos(theta) * torch.sin(phi), torch.cos(phi), torch.sin(theta) * torch.sin(phi)], dim=-1)  # [H, W, 3]
        ls = view_dirs[None, None, :].float().to(self.device)

        tp_pre, la_pre, w_pre = self.model(rgb_imgs, mask_imgs)
        env_pres = self.sg2env([tp_pre, la_pre, w_pre], ls=ls)
        if is_save:
            env_saves = env_pres.detach().cpu().numpy()
            if is_clip:
                env_lums = np.sum(env_saves * self.lum_weight.cpu().numpy(), axis=-1, keepdims=True).repeat(3, -1)
                env_clips = env_saves.copy()
                env_clips[env_lums < 1] = 0.
                env_saves = env_clips
            for env_save, img_name in zip(env_saves, img_names):
                cv.imwrite(f'{self.img_log_dir}/{img_name}.exr', env_save, exr_save_params)
                if is_tonemap:
                    env_save_ldr = np.clip(env_save ** (1 / 2.2), 0, 1) * 255
                    cv.imwrite(f'{self.img_log_dir}/{img_name}.jpg', env_save_ldr.astype(np.uint8))
                print(f'sg_net inference: {img_name}')
        return env_pres.permute(0, 3, 1, 2)

    def forward(self, batch, batch_idx, stage, data_num):
        rgb_img, mask_img, env_gt, env_gt_clip, sg_gt, img_name = batch
        bs = rgb_img.shape[0]
        sbs = max(data_num / bs // 5, 2)
        tp_gt, la_gt, w_gt = torch.split(sg_gt, [3, 1, 3], dim=-1)
        tp_pre, la_pre, w_pre = self.model(rgb_img, mask_img)
        env_pre = self.sg2env([tp_pre, la_pre, w_pre])
        # env_sg = self.sg2env([tp_gt, la_gt, w_gt], is_gt=True)
        if (batch_idx + 1) % sbs == 0:
            self.log_image(self.current_epoch, bs, env_pre, env_gt_clip, img_name, stage, batch_idx)
        if stage not in ['train', 'val', 'test']: return env_pre
        pano_loss = self.mse_loss(torch.log1p(env_pre), torch.log1p(env_gt))
        sg_loss = self.get_sg_pair_loss(bs, tp_pre, tp_gt, la_pre=la_pre, w_pre=w_pre, la_gt=la_gt, w_gt=w_gt)
        total_loss = pano_loss + sg_loss
        log_info = {'tl': total_loss, 'pano': pano_loss, 'sg': sg_loss}
        self.log_dict(log_info, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, batch_size=bs)
        return total_loss
        # return sg_loss


class ASGViewer:
    def __init__(self, res=(512, 256)):
        super().__init__()
        self.asg_param_num = 6
        self.asg_num = 128
        self.width, self.height = res
        self.lobe_path = './lighting_est'
        Az = ((np.arange(self.width)) / self.width) * 2 * np.pi
        El = ((np.arange(self.height)) / self.height) * np.pi
        Az, El = np.meshgrid(Az, El)
        Az = Az[:, :, np.newaxis]
        El = El[:, :, np.newaxis]
        lx = np.sin(El) * np.cos(Az)
        ly = np.sin(El) * np.sin(Az)
        lz = np.cos(El)
        ls = np.concatenate((lx, ly, lz), axis=2)[np.newaxis, np.newaxis, np.newaxis, :].astype(np.float32)
        ls = torch.from_numpy(ls)
        self.ls = ls.expand([1, self.asg_num, 1, self.height, self.width, 3])

        self.bi_angle = torch.ones((self.asg_num, 1))
        zeros = torch.zeros(self.asg_num, 3, self.height, self.width)

        fib_xyz = np.load(f'{self.lobe_path}/fib_lobes_{self.asg_num}.npy')
        nx = torch.from_numpy(fib_xyz[:, 0:1] + TINY_NUMBER).expand([1, self.asg_num, 1])
        ny = torch.from_numpy(fib_xyz[:, 1:2] + TINY_NUMBER).expand([1, self.asg_num, 1])
        nz = torch.from_numpy(fib_xyz[:, 2:] + TINY_NUMBER).expand([1, self.asg_num, 1])

        denom = torch.sqrt(nx * nx + nz * nz)
        tx = -1 * nx * ny / denom
        tz = -1 * ny * nz / denom
        ty = denom

        self.normal_ori = torch.cat([nx, ny, nz], dim=-1)
        self.normal_tan = torch.cat([tx, ty, tz], dim=-1)
        self.normal_cross = torch.cross(self.normal_ori, self.normal_tan, dim=-1)

        normal = self.normal_ori[:, :, None, None, None, :]
        normal = normal.expand([1, self.asg_num, 1, self.height, self.width, 3])
        smooth = torch.maximum(zeros, torch.sum(self.ls * normal, dim=-1))
        self.smooth = smooth.expand([1, self.asg_num, 3, self.height, self.width])

        self.normal_tan = self.normal_tan.cuda()
        self.normal_cross = self.normal_cross.cuda()
        self.normal_ori = self.normal_ori.cuda()
        self.bi_angle = self.bi_angle.cuda()
        self.ls = self.ls.cuda()
        self.smooth = self.smooth.cuda()

    def __call__(self, params):
        batch_size = params[0].shape[0]
        angle = params[0]
        weight = params[-1]
        lamb = 1 / (params[1] + 1e-8)
        mu = 1 / (params[2] + 1e-8)

        cos_an = torch.cos(angle)
        sin_an = torch.sin(angle)

        tan = cos_an * self.normal_tan + sin_an * self.normal_cross
        bi_tan = torch.cos(self.bi_angle) * tan + torch.sin(self.bi_angle) * torch.cross(self.normal_ori, tan, dim=-1)

        tan = tan[:, :, None, None, None, :]
        tan = tan.expand(batch_size, self.asg_num, 1, self.height, self.width, 3)

        bi_tan = bi_tan[:, :, None, None, None, :]
        bi_tan = bi_tan.expand(batch_size, self.asg_num, 1, self.height, self.width, 3)

        lamb = lamb[:, :, None, None, :]
        lamb = lamb.expand(batch_size, self.asg_num, 1, self.height, self.width)
        mu = mu[:, :, None, None, :]
        mu = mu.expand(batch_size, self.asg_num, 1, self.height, self.width)

        weight = weight[:, :, :, None, None]
        weight = weight.expand(batch_size, self.asg_num, 3, self.height, self.width)

        e_power1 = lamb * (torch.sum(self.ls * tan, dim=-1) ** 2)
        e_power2 = mu * (torch.sum(self.ls * bi_tan, dim=-1) ** 2)
        e_power = e_power1 + e_power2

        e_item = torch.exp(-e_power)
        e_item = e_item.expand(batch_size, self.asg_num, 3, self.height, self.width)

        envmaps = self.smooth * weight * e_item
        envmap = torch.sum(envmaps, dim=1)  # .permute(0, 2, 3, 1)

        envmap = torch.clamp_min(envmap, TINY_NUMBER).float()
        return envmap


class ASGNetModule(BaseModule):
    def __init__(self, img_log_dir=None, resolution=(256, 128), learning_rate=1e-4):
        super().__init__()
        self.lobe_path = './lighting_est'
        self.img_log_dir = img_log_dir
        self.asg_param_num = 6
        self.asg_num = 128
        self.model = ASGNet(self.asg_num)
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.vgg_loss = VGGLoss()

        self.asg_viewer = ASGViewer(resolution)
        self.width, self.height = 256, 128  # fixed for training
        Az = ((np.arange(self.width)) / self.width) * 2 * np.pi
        El = ((np.arange(self.height)) / self.height) * np.pi
        Az, El = np.meshgrid(Az, El)
        Az = Az[:, :, np.newaxis]
        El = El[:, :, np.newaxis]
        lx = np.sin(El) * np.cos(Az)
        ly = np.sin(El) * np.sin(Az)
        lz = np.cos(El)
        ls = np.concatenate((lx, ly, lz), axis=2)[np.newaxis, np.newaxis, np.newaxis, :].astype(np.float32)
        ls = torch.from_numpy(ls)
        ls = ls.expand([1, self.asg_num, 1, self.height, self.width, 3])  # .cuda()

        bi_angle = torch.ones((self.asg_num, 1))  # .cuda() * torch.pi / 2.
        zeros = torch.zeros(self.asg_num, 3, self.height, self.width)  # .cuda()

        fib_xyz = np.load(f'{self.lobe_path}/fib_lobes_{self.asg_num}.npy')
        nx = torch.from_numpy(fib_xyz[:, 0:1] + TINY_NUMBER).expand([1, self.asg_num, 1])  # .cuda()
        ny = torch.from_numpy(fib_xyz[:, 1:2] + TINY_NUMBER).expand([1, self.asg_num, 1])  # .cuda()
        nz = torch.from_numpy(fib_xyz[:, 2:] + TINY_NUMBER).expand([1, self.asg_num, 1])  # .cuda()

        denom = torch.sqrt(nx * nx + nz * nz)
        tx = -1 * nx * ny / denom
        tz = -1 * ny * nz / denom
        ty = denom

        normal_ori = torch.cat([nx, ny, nz], dim=-1)
        normal_tan = torch.cat([tx, ty, tz], dim=-1)
        normal_cross = torch.cross(normal_ori, normal_tan, dim=-1)

        normal = normal_ori[:, :, None, None, None, :]
        normal = normal.expand([1, self.asg_num, 1, self.height, self.width, 3])
        smooth = torch.maximum(zeros, torch.sum(ls * normal, dim=-1))
        smooth = smooth.expand([1, self.asg_num, 3, self.height, self.width])

        self.learning_rate = learning_rate
        self.save_hyperparameters()
        self.example_input_array = [[torch.Tensor(1, 3, 256, 256),
                                     torch.Tensor(1, 3, 256, 512),
                                     torch.Tensor(1, 128, 6), 'name'], 0, 'inference', 100]

        self.register_buffer('normal_tan', normal_tan)
        self.register_buffer('normal_cross', normal_cross)
        self.register_buffer('normal_ori', normal_ori)
        self.register_buffer('bi_angle', bi_angle)
        self.register_buffer('ls', ls.clone())
        self.register_buffer('smooth', smooth)

    def log_image(self, epoch, idx_max, env_pre, env_gt, img_name, stage, batch_idx):
        save_idx = np.random.randint(0, idx_max)
        env_save = env_pre[save_idx].permute(1, 2, 0).detach().cpu().numpy()  # .permute(1, 2, 0)
        env_gt_save = env_gt[save_idx].permute(1, 2, 0).detach().cpu().numpy()
        grid = np.concatenate((env_save, env_gt_save), axis=0)
        grid = np.clip(grid, 0, 1) * 255
        cv.imwrite(f'{self.img_log_dir}/{stage}/e{str(epoch).zfill(3)}_b{batch_idx}_{img_name[save_idx]}_grid.jpg',
                   grid.astype(np.uint8))

    def get_loss(self, an_pre, la_pre, mu_pre, w_pre, an_gt, la_gt, mu_gt, w_gt):
        la_gt = 1 / torch.clamp_min(la_gt, 1)
        mu_gt = 1 / torch.clamp_min(mu_gt, 1)
        an_loss = self.mse_loss(torch.sin(an_pre), torch.sin(an_gt))
        la_loss = self.mse_loss(la_pre, la_gt)
        mu_loss = self.mse_loss(mu_pre, mu_gt)
        w_loss = self.mse_loss(w_pre, w_gt)
        return 0.01 * an_loss + la_loss + mu_loss + w_loss

    def asg2env(self, params):
        batch_size = params[0].shape[0]
        angle = params[0]
        weight = params[-1]
        lamb = 1 / (params[1] + 1e-8)
        mu = 1 / (params[2] + 1e-8)

        cos_an = torch.cos(angle)
        sin_an = torch.sin(angle)

        tan = cos_an * self.normal_tan + sin_an * self.normal_cross
        bi_tan = torch.cos(self.bi_angle) * tan + torch.sin(self.bi_angle) * torch.cross(self.normal_ori, tan, dim=-1)

        tan = tan[:, :, None, None, None, :]
        tan = tan.expand(batch_size, self.asg_num, 1, self.height, self.width, 3)

        bi_tan = bi_tan[:, :, None, None, None, :]
        bi_tan = bi_tan.expand(batch_size, self.asg_num, 1, self.height, self.width, 3)

        lamb = lamb[:, :, None, None, :]
        lamb = lamb.expand(batch_size, self.asg_num, 1, self.height, self.width)
        mu = mu[:, :, None, None, :]
        mu = mu.expand(batch_size, self.asg_num, 1, self.height, self.width)

        weight = weight[:, :, :, None, None]
        weight = weight.expand(batch_size, self.asg_num, 3, self.height, self.width)

        e_power1 = lamb * (torch.sum(self.ls * tan, dim=-1) ** 2)
        e_power2 = mu * (torch.sum(self.ls * bi_tan, dim=-1) ** 2)
        e_power = e_power1 + e_power2

        e_item = torch.exp(-e_power)
        e_item = e_item.expand(batch_size, self.asg_num, 3, self.height, self.width)

        envmaps = self.smooth * weight * e_item
        envmap = torch.sum(envmaps, dim=1)  # .permute(0, 2, 3, 1)

        envmap = torch.clamp_min(envmap, TINY_NUMBER).float()
        return envmap

    def inference(self, rgb_imgs, img_names, is_save=True):
        an_pre, la_pre, mu_pre, w_pre = self.model(rgb_imgs)
        env_pres = self.asg_viewer([an_pre, la_pre, mu_pre, w_pre])
        env_pres = torch.clamp(env_pres, 0, 1)
        if is_save:
            env_saves = env_pres.permute(0, 2, 3, 1).detach().cpu().numpy() * 255
            for env_save, img_name in zip(env_saves, img_names):
                cv.imwrite(f'{self.img_log_dir}/{img_name}.jpg', env_save.astype(np.uint8))
                print(f'asg_net inference: {img_name}')
        return env_pres

    def forward(self, batch, batch_idx, stage, data_num):
        rgb_img, env_gt, asg_gt, img_name = batch
        bs = rgb_img.shape[0]
        sbs = max(data_num / bs // 5, 2)
        an_gt, la_gt, mu_gt, w_gt = torch.split(asg_gt, [1, 1, 1, 3], dim=-1)
        an_pre, la_pre, mu_pre, w_pre = self.model(rgb_img)
        env_pre = self.asg2env([an_pre, la_pre, mu_pre, w_pre])
        if (batch_idx + 1) % sbs == 0:
            self.log_image(self.current_epoch, bs, env_pre, env_gt, img_name, stage, batch_idx)
        if stage not in ['train', 'val', 'test']:
            return env_pre
        pano_loss = self.mse_loss(env_pre, env_gt)
        asg_loss = self.get_loss(an_pre, la_pre, mu_pre, w_pre, an_gt, la_gt, mu_gt, w_gt)
        vgg_loss = self.vgg_loss(env_pre, env_gt)
        total_loss = pano_loss + asg_loss + 0.01 * vgg_loss
        log_info = {'tl': total_loss, 'pano': pano_loss, 'asg': asg_loss, 'vgg': vgg_loss}
        self.log_dict(log_info, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, batch_size=bs)
        return total_loss
