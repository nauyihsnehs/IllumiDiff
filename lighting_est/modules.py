import math
from pathlib import Path

import lightning as L
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn, optim
from torch.nn import functional as F
from torchvision import models, transforms

from utils import (
    EXR_SAVE_PARAMS,
    render_sg_luminance,
    write_image,
)
from lighting_est.models import ASGNet, HDRNet, IDNet, SGNet

torch.set_float32_matmul_precision("medium")
TINY_NUMBER = 1e-8
exr_save_params = EXR_SAVE_PARAMS


def stretch_for_hdrnet(panoramas, resolution):
    width, height = resolution
    if panoramas.shape[-2:] == (height, width):
        return panoramas
    return F.interpolate(
        panoramas,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )


def strip_frozen_loss(checkpoint):
    state_dict = checkpoint.get("state_dict", {})
    for key in [key for key in state_dict if key.startswith("vgg_loss.")]:
        state_dict.pop(key)


def restore_frozen_loss(checkpoint, module):
    if module.vgg_loss is None:
        strip_frozen_loss(checkpoint)
        return
    state_dict = checkpoint.get("state_dict", {})
    for key, value in module.state_dict().items():
        if key.startswith("vgg_loss.") and key not in state_dict:
            state_dict[key] = value


def name_at(img_name, index=0):
    if isinstance(img_name, (list, tuple)):
        return img_name[index]
    return img_name


def log_gradient_loss(prediction, target):
    prediction_x = torch.roll(prediction, shifts=-1, dims=-1) - prediction
    target_x = torch.roll(target, shifts=-1, dims=-1) - target
    prediction_y = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_y = target[..., 1:, :] - target[..., :-1, :]
    horizontal = F.l1_loss(prediction_x, target_x)
    vertical = F.l1_loss(prediction_y, target_y)
    return (horizontal + vertical) * 0.5


def to_hwc(value, index=0):
    if isinstance(value, torch.Tensor):
        if value.ndim == 4:
            value = value[index]
        value = value.detach().cpu()
        if value.ndim == 3 and value.shape[0] in (1, 3, 4):
            value = value.permute(1, 2, 0)
        return value.numpy()
    value = np.asarray(value)
    if value.ndim == 4:
        value = value[index]
    return value


def to_rgb(image):
    if image.ndim == 2:
        image = image[..., None]
    if image.shape[-1] == 1:
        image = image.repeat(3, axis=-1)
    return image


def to_uint8(image):
    return (np.clip(image, 0, 1) * 255).astype(np.uint8)


def linearize_ldr(image, index=0):
    image = (to_hwc(image, index) + 1) * 0.5
    return np.clip(image, 0, 1) ** 2.2


def prefixed(log_info, stage):
    return {f"{stage}/{key}": value for key, value in log_info.items()}


def match_nchw(value, reference):
    if (
        value.ndim == 4
        and value.shape[1] != reference.shape[1]
        and value.shape[-1] == reference.shape[1]
    ):
        return value.permute(0, 3, 1, 2).contiguous()
    return value


class VGGLoss(nn.Module):
    models = {"vgg16": models.vgg16, "vgg19": models.vgg19}

    def __init__(self, model="vgg19", layers=(8,), distance="mse"):
        super().__init__()
        self.layers = tuple(layers)
        self.loss = F.l1_loss if distance == "l1" else F.mse_loss
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.model = self.models[model](weights="DEFAULT").features[
            : max(self.layers) + 1
        ]
        self.model.eval()
        self.model.requires_grad_(False)

    def prepare(self, x):
        if x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2)
        return self.normalize(x[:, [2, 1, 0]].clamp(0, 1))

    def train(self, mode=True):
        super().train(False)
        return self

    def forward(self, x, target):
        x = self.prepare(x)
        with torch.no_grad():
            target = self.prepare(target)
        losses = []
        for index, layer in enumerate(self.model):
            x = layer(x)
            with torch.no_grad():
                target = layer(target)
            if index in self.layers:
                losses.append(self.loss(x, target))
        return torch.stack(losses).mean()


class BaseModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.lr_decay_steps = 50

    def training_step(self, batch, batch_idx):
        return self(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        loss = self(batch, batch_idx, "val")
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            batch_size=batch[0].shape[0],
        )
        return loss

    def predict_step(self, batch, batch_idx):
        self.inference(*batch)
        return []

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=max(int(self.lr_decay_steps), 1), gamma=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def on_save_checkpoint(self, checkpoint):
        strip_frozen_loss(checkpoint)

    def perceptual_loss(self, **kwargs):
        if getattr(self, "vgg_loss", None) is None:
            self.vgg_loss = VGGLoss(**kwargs).to(self.device)
        return self.vgg_loss


class IDNetModule(BaseModule):
    def __init__(self, img_log_dir=None, learning_rate=1e-4, lr_decay_steps=50):
        super().__init__()
        self.img_log_dir = img_log_dir
        self.model = IDNet(3, 1)

        self.learning_rate = learning_rate
        self.lr_decay_steps = lr_decay_steps
        self.save_hyperparameters()
        self.example_input_array = [
            [
                torch.Tensor(1, 3, 256, 512),
                torch.Tensor(1, 1, 256, 512),
                "name",
            ],
            0,
            "inference",
        ]

    def lum_record(self, ldr_img, lum_pre, lum_gt, img_name, index=0):
        lum_pre_save = to_hwc(lum_pre, index)
        ldr_save = to_rgb(to_hwc(ldr_img, index))
        views = [ldr_save, to_rgb(lum_pre_save)]
        if lum_gt is not None:
            views.append(to_rgb(to_hwc(lum_gt, index)))
        grid = np.concatenate(views, axis=1)
        return {
            "name": name_at(img_name, index),
            "image": to_uint8(grid),
            "ext": ".jpg",
        }

    @torch.no_grad()
    def log_images(self, batch, split="train"):
        ldr_img, lum_gt, img_name = batch
        lum_pre = self.model(ldr_img)
        return [self.lum_record(ldr_img, lum_pre, lum_gt, img_name)]

    @torch.no_grad()
    def log_wild_images(self, batch):
        ldr_img, img_name = batch
        ldr_img = ldr_img.to(self.device).float()
        lum_pre = self.model(ldr_img)
        return [self.lum_record(ldr_img, lum_pre, None, img_name)]

    @torch.no_grad()
    def inference(self, ldr_imgs, img_names=None, is_save=True, output_dir=None):
        lum = self.model(ldr_imgs)
        if is_save and img_names is not None:
            root = output_dir or self.img_log_dir
            lum_saves = lum.detach().cpu().permute(0, 2, 3, 1).numpy() * 255
            for lum_save, img_name in zip(lum_saves, img_names):
                write_image(
                    Path(root, f"{img_name}.png"),
                    lum_save.astype(np.uint8),
                )
                print(f"id_net inference: {img_name}")
        return lum

    def get_loss(self, lum_pre, lum_gt):
        loss = F.smooth_l1_loss(lum_pre, lum_gt)
        return loss, {"tl": loss, "smooth_l1": loss}

    def forward(self, batch, batch_idx, stage):
        ldr_img, lum_gt, img_name = batch
        bs = ldr_img.size(0)
        lum_pre = self.model(ldr_img)
        if stage not in ["train", "val", "test"]:
            return lum_pre
        total_loss, log_info = self.get_loss(lum_pre, lum_gt)
        self.log_dict(
            prefixed(log_info, stage),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            prog_bar=True,
            batch_size=bs,
        )
        return total_loss


class HDRNetModule(BaseModule):
    def __init__(
        self,
        img_log_dir=None,
        learning_rate=1e-5,
        sg_scale=1.0,
        exposure_loss_weight=0.1,
        perceptual_loss_weight=0.0,
        gradient_loss_weight=0.05,
        highlight_weight=2.0,
        highlight_loss_weight=0.1,
        max_hdr=1e4,
        **legacy_hparams,
    ):
        super().__init__()
        self.img_log_dir = img_log_dir
        self.model = HDRNet(4, 3, sg_channels=1)

        self.learning_rate = learning_rate
        self.sg_scale = sg_scale
        self.exposure_loss_weight = exposure_loss_weight
        self.perceptual_loss_weight = perceptual_loss_weight
        self.gradient_loss_weight = gradient_loss_weight
        self.highlight_weight = highlight_weight
        self.highlight_loss_weight = highlight_loss_weight
        self.max_hdr = max_hdr
        self.save_hyperparameters(ignore=["legacy_hparams"])
        self.example_input_array = [
            [
                torch.Tensor(1, 3, 256, 512),
                torch.Tensor(1, 12, 5),
                torch.Tensor(1, 1, 256, 512),
                torch.Tensor(1, 3, 256, 512),
                torch.ones(1),
                torch.zeros(1, dtype=torch.bool),
                "name",
            ],
            0,
            "inference",
        ]

        self.vgg_loss = None

    def on_load_checkpoint(self, checkpoint):
        restore_frozen_loss(checkpoint, self)

    def on_train_start(self):
        for param_group in self.trainer.optimizers[0].param_groups:
            param_group["lr"] = self.learning_rate

    def configure_optimizers(self):
        return optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.5, 0.999),
        )

    def sg_image(self, params, reference, scale=None, flips=None):
        scale = self.sg_scale if scale is None else scale
        image = render_sg_luminance(
            params,
            reference.shape[-1],
            reference.shape[-2],
            flips=flips,
        )
        scale = torch.as_tensor(scale, device=image.device, dtype=image.dtype)
        if scale.ndim:
            scale = scale.reshape(-1, 1, 1, 1)
        image = torch.log1p(image * scale).sub_(1)
        upper = math.log1p(self.max_hdr) - 1
        image = torch.nan_to_num(image, nan=-1, posinf=upper, neginf=-1)
        image = image.clamp(-1, upper)
        return image.permute(0, 3, 1, 2).contiguous()

    def linear_hdr(self, log_hdr):
        lower = math.log(TINY_NUMBER)
        upper = math.log(self.max_hdr)
        return torch.exp(log_hdr.clamp(lower, upper))

    def hdr_record(self, ldr_img, hdr_pre, hdr_gt, sg_img, lum_img, img_name, index=0):
        ldr_save = linearize_ldr(ldr_img, index)
        hdr_save = to_hwc(self.linear_hdr(hdr_pre), index)
        sg_linear = np.expm1(to_hwc(sg_img, index) + 1)
        sg_save = to_rgb(sg_linear)
        lum_save = (to_rgb(to_hwc(lum_img, index)) + 1) * 0.5
        if hdr_gt is None:
            grid = np.concatenate((ldr_save, hdr_save, sg_save, lum_save), axis=1)
        else:
            hdr_gt_save = to_hwc(self.linear_hdr(hdr_gt), index)
            blank = np.zeros_like(ldr_save)
            top = np.concatenate((ldr_save, hdr_save, hdr_gt_save), axis=1)
            bottom = np.concatenate((sg_save, lum_save, blank), axis=1)
            grid = np.concatenate((top, bottom), axis=0)
        return {
            "name": name_at(img_name, index),
            "image": grid.astype(np.float32, copy=False),
            "ext": ".exr",
            "params": exr_save_params,
        }

    @torch.no_grad()
    def log_images(self, batch, split="train"):
        rgb_img, sg_params, lum_img, hdr_gt, _exposure, sg_flip, img_name = batch
        sg_img = self.sg_image(
            sg_params,
            rgb_img,
            scale=1.0,
            flips=sg_flip,
        )
        hdr_pre = self.model(rgb_img, sg_img, lum_img)
        hdr_gt = match_nchw(hdr_gt, hdr_pre)
        return [
            self.hdr_record(
                rgb_img,
                hdr_pre,
                hdr_gt,
                sg_img,
                lum_img,
                img_name,
            )
        ]

    @torch.no_grad()
    def log_wild_images(self, batch):
        rgb_img, sg_params, lum_img, img_name = batch
        rgb_img = rgb_img.to(self.device).float()
        sg_params = sg_params.to(self.device).float()
        lum_img = lum_img.to(self.device).float()
        sg_img = self.sg_image(sg_params, rgb_img)
        hdr_pre = self.model(rgb_img, sg_img, lum_img)
        return [
            self.hdr_record(
                rgb_img,
                hdr_pre,
                None,
                sg_img,
                lum_img,
                img_name,
                index,
            )
            for index in range(rgb_img.shape[0])
        ]

    def get_loss(self, hdr_pre, hdr_gt, lum_img):
        error = hdr_pre - hdr_gt
        mask = ((lum_img + 1) * 0.5).clamp(0, 1)
        weights = (1 + self.highlight_weight * mask).expand_as(error)
        dimensions = tuple(range(1, error.ndim))
        weight_sum = weights.sum(dim=dimensions).clamp_min(TINY_NUMBER)
        mean_error = (weights * error).sum(dim=dimensions) / weight_sum
        mean_square = (weights * error.square()).sum(dim=dimensions) / weight_sum

        si_loss = (mean_square - mean_error.square()).clamp_min(0).mean()
        exposure_loss = mean_error.square().mean()
        highlight_mask = mask.expand_as(error)
        highlight_sum = highlight_mask.sum(dim=dimensions).clamp_min(TINY_NUMBER)
        highlight_loss = (
            (highlight_mask * error.square()).sum(dim=dimensions) / highlight_sum
        ).mean()
        gradient_loss = log_gradient_loss(hdr_pre, hdr_gt)
        vgg_loss = hdr_pre.new_zeros(())
        if self.perceptual_loss_weight > 0:
            display_pre = torch.sigmoid(hdr_pre)
            display_gt = torch.sigmoid(hdr_gt)
            vgg_loss = self.perceptual_loss(
                model="vgg16",
                layers=(3, 8, 15),
                distance="l1",
            )(display_pre, display_gt)
        total_loss = (
            si_loss
            + self.exposure_loss_weight * exposure_loss
            + self.perceptual_loss_weight * vgg_loss
            + self.gradient_loss_weight * gradient_loss
            + self.highlight_loss_weight * highlight_loss
        )
        log_info = {
            "tl": total_loss,
            "si": si_loss,
            "exposure": exposure_loss,
            "vgg": vgg_loss,
            "gradient": gradient_loss,
            "highlight": highlight_loss,
        }
        return total_loss, log_info

    @torch.no_grad()
    def inference(
        self, rgb_imgs, sg_params, lum_imgs, img_names, is_save=True, sg_scale=None
    ):
        sg_imgs = self.sg_image(sg_params, rgb_imgs, sg_scale)
        hdr_logs = self.model(rgb_imgs, sg_imgs, lum_imgs)
        hdr_pres = self.linear_hdr(hdr_logs)
        if is_save:
            hdr_saves = hdr_pres.detach().cpu().permute(0, 2, 3, 1).numpy()
            for hdr_save, img_name in zip(hdr_saves, img_names):
                write_image(
                    Path(self.img_log_dir, f"{img_name}.exr"), hdr_save, exr_save_params
                )
                minimum = max(float(hdr_save.min()), TINY_NUMBER)
                print(
                    f"hdr_net inference: {img_name}, max: {hdr_save.max()}, min: {hdr_save.min()}, range: {hdr_save.max() / minimum}"
                )
        return hdr_pres

    def forward(self, batch, batch_idx, stage):
        rgb_img, sg_params, lum_img, hdr_gt, _exposure, sg_flip, img_name = batch
        bs = rgb_img.size(0)
        sg_img = self.sg_image(
            sg_params,
            rgb_img,
            scale=1.0,
            flips=sg_flip,
        )
        hdr_pre = self.model(rgb_img, sg_img, lum_img)
        if stage not in ["train", "val", "test"]:
            return hdr_pre
        hdr_gt = match_nchw(hdr_gt, hdr_pre)
        total_loss, log_info = self.get_loss(hdr_pre, hdr_gt, lum_img)
        self.log_dict(
            prefixed(log_info, stage),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            prog_bar=True,
            batch_size=bs,
        )
        return total_loss


class SGNetModule(BaseModule):
    def __init__(
        self,
        img_log_dir=None,
        resolution=(512, 256),
        learning_rate=1e-4,
        lr_decay_steps=50,
        backbone_weights="DEFAULT",
        pano_loss_weight=0.0,
    ):
        super().__init__()
        self.img_log_dir = img_log_dir
        self.sg_num = 12
        self.model = SGNet(self.sg_num, backbone_weights=backbone_weights)
        self.mse_loss = nn.MSELoss()

        self.width, self.height = resolution

        self.learning_rate = learning_rate
        self.lr_decay_steps = lr_decay_steps
        self.pano_loss_weight = pano_loss_weight
        self.save_hyperparameters()
        self.example_input_array = [
            [
                torch.Tensor(1, 3, 256, 256),
                torch.Tensor(1, 1, 256, 256),
                torch.Tensor(1, 12, 5),
                "name",
            ],
            0,
            "inference",
        ]

    def on_load_checkpoint(self, checkpoint):
        strip_frozen_loss(checkpoint)
        state_dict = checkpoint.get("state_dict")
        if state_dict is not None:
            state_dict.pop("ls", None)
            state_dict.pop("area_weight", None)
            state_dict.pop("lum_weight", None)

    def env_records(self, ldr_img, env_pre, env_gt, img_name, index=0):
        ldr_save = linearize_ldr(ldr_img, index)
        env_save = to_rgb(to_hwc(env_pre, index))
        views = [ldr_save, env_save]
        if env_gt is not None:
            views.append(to_rgb(to_hwc(env_gt, index)))
        grid = np.concatenate(views, axis=1)
        return [
            {
                "name": name_at(img_name, index),
                "image": grid.astype(np.float32, copy=False),
                "ext": ".exr",
                "params": exr_save_params,
            }
        ]

    def get_sg_pair_loss(
        self, tp_pre, tp_gt, la_pre=None, w_pre=None, la_gt=None, w_gt=None
    ):
        cost = torch.cdist(tp_pre.detach(), tp_gt.detach(), p=2)
        if (
            la_pre is not None
            and w_pre is not None
            and la_gt is not None
            and w_gt is not None
        ):
            cost += 0.05 * torch.cdist(
                torch.log1p(la_pre.detach()), torch.log1p(la_gt), p=1
            )
            cost += 0.05 * torch.cdist(
                torch.log1p(w_pre.detach()), torch.log1p(w_gt), p=1
            )
        assignments = []
        for cost_matrix in cost.detach().cpu().numpy():
            rows, columns = linear_sum_assignment(cost_matrix)
            assignment = np.empty(self.sg_num, dtype=np.int64)
            assignment[rows] = columns
            assignments.append(assignment)
        index_tensor = torch.as_tensor(np.stack(assignments), device=tp_pre.device)

        gather_direction = index_tensor[..., None].expand(-1, -1, tp_gt.shape[-1])
        tp_paired = torch.gather(tp_gt, 1, gather_direction)
        tp_loss = (1 - torch.sum(tp_pre * tp_paired, dim=-1).clamp(-1, 1)).mean()

        if la_pre is None or w_pre is None or la_gt is None or w_gt is None:
            return tp_loss

        la_paired = torch.gather(la_gt, 1, index_tensor[..., None])
        w_paired = torch.gather(
            w_gt, 1, index_tensor[..., None].expand(-1, -1, w_gt.shape[-1])
        )

        la_loss = self.mse_loss(torch.log1p(la_pre), torch.log1p(la_paired))
        w_loss = self.mse_loss(torch.log1p(w_pre), torch.log1p(w_paired))
        return tp_loss + 0.1 * la_loss + 0.1 * w_loss

    def pack_params(self, direction, lamb, weight):
        return torch.cat((direction, lamb, weight), dim=-1)

    def sg2env(self, params):
        return render_sg_luminance(params, self.width, self.height)

    @torch.no_grad()
    def inference(
        self,
        rgb_imgs,
        lum_imgs,
        img_names,
        is_save=True,
        is_tonemap=True,
    ):
        tp_pre, la_pre, w_pre = self.model(rgb_imgs, lum_imgs)
        params = self.pack_params(tp_pre, la_pre, w_pre)
        if is_save:
            env_saves = self.sg2env(params).detach().cpu().numpy()
            param_saves = params.detach().cpu().numpy().astype(np.float32)
            for param_save, env_save, img_name in zip(
                param_saves, env_saves, img_names
            ):
                np.save(Path(self.img_log_dir, f"{img_name}.npy"), param_save)
                if is_tonemap:
                    env_save_ldr = np.clip(env_save[..., 0] ** (1 / 2.2), 0, 1) * 255
                    write_image(
                        Path(self.img_log_dir, f"{img_name}.jpg"),
                        env_save_ldr.astype(np.uint8),
                    )
                print(f"sg_net inference: {img_name}")
        return params

    @torch.no_grad()
    def log_images(self, batch, split="train"):
        rgb_img, lum_img, sg_gt, img_name = batch
        tp_pre, la_pre, w_pre = self.model(rgb_img, lum_img)
        env_pre = self.sg2env(self.pack_params(tp_pre, la_pre, w_pre))
        env_gt = self.sg2env(sg_gt)
        return self.env_records(rgb_img, env_pre, env_gt, img_name)

    @torch.no_grad()
    def log_wild_images(self, batch):
        rgb_img, lum_img, img_name = batch
        rgb_img = rgb_img.to(self.device).float()
        lum_img = lum_img.to(self.device).float()
        tp_pre, la_pre, w_pre = self.model(rgb_img, lum_img)
        env_pre = self.sg2env(self.pack_params(tp_pre, la_pre, w_pre))
        return self.env_records(rgb_img, env_pre, None, img_name)

    def forward(self, batch, batch_idx, stage):
        rgb_img, lum_img, sg_gt, img_name = batch
        bs = rgb_img.shape[0]
        tp_gt, la_gt, w_gt = torch.split(sg_gt, [3, 1, 1], dim=-1)
        tp_pre, la_pre, w_pre = self.model(rgb_img, lum_img)
        if stage not in ["train", "val", "test"]:
            return self.pack_params(tp_pre, la_pre, w_pre)
        sg_loss = self.get_sg_pair_loss(
            tp_pre, tp_gt, la_pre=la_pre, w_pre=w_pre, la_gt=la_gt, w_gt=w_gt
        )
        total_loss = sg_loss
        log_info = {"sg": sg_loss}
        if self.pano_loss_weight > 0:
            env_pre = self.sg2env(self.pack_params(tp_pre, la_pre, w_pre))
            with torch.no_grad():
                env_gt = self.sg2env(sg_gt)
            pano_loss = self.mse_loss(torch.log1p(env_pre), torch.log1p(env_gt))
            total_loss = total_loss + self.pano_loss_weight * pano_loss
            log_info["pano"] = pano_loss
        log_info["tl"] = total_loss
        self.log_dict(
            prefixed(log_info, stage),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            prog_bar=True,
            batch_size=bs,
        )
        return total_loss


class ASGViewer(nn.Module):
    def __init__(self, res=(512, 256)):
        super().__init__()
        self.asg_num = 128
        self.width, self.height = res

        az = torch.arange(self.width).float() / self.width * 2 * torch.pi
        el = torch.arange(self.height).float() / self.height * torch.pi
        el, az = torch.meshgrid(el, az, indexing="ij")
        ls = torch.stack(
            [
                torch.sin(el) * torch.cos(az),
                torch.cos(el),
                torch.sin(el) * torch.sin(az),
            ],
            dim=-1,
        ).view(-1, 3)

        fib_xyz = np.load(
            Path(__file__).with_name(f"fib_lobes_{self.asg_num}.npy")
        ).astype(np.float32)
        fib = torch.from_numpy(fib_xyz)
        fib = F.normalize(fib + TINY_NUMBER, dim=-1)
        normal_ori = fib[None]
        normal_tan, normal_cross = self.get_tangent_frame(normal_ori)
        self.register_buffer("normal_ori", normal_ori, persistent=False)
        self.register_buffer("normal_tan", normal_tan, persistent=False)
        self.register_buffer("normal_cross", normal_cross, persistent=False)
        self.register_buffer("ls_t", ls.t(), persistent=False)
        self.register_buffer(
            "smooth",
            torch.clamp_min(torch.matmul(normal_ori, ls.t()), 0.0),
            persistent=False,
        )

    def get_tangent_frame(self, normal):
        nx, ny, nz = torch.split(normal, [1, 1, 1], dim=-1)
        denom = torch.sqrt(nx * nx + nz * nz + TINY_NUMBER)
        normal_tan = torch.cat([-nx * ny / denom, denom, -ny * nz / denom], dim=-1)
        normal_tan = F.normalize(normal_tan, dim=-1)
        normal_cross = torch.cross(normal, normal_tan, dim=-1)
        return normal_tan, normal_cross

    def forward(self, params):
        batch_size = params[0].shape[0]
        angle = params[0]
        weight = params[-1]
        lamb = torch.clamp_min(params[1], TINY_NUMBER)
        mu = torch.clamp_min(params[2], TINY_NUMBER)

        tan = torch.cos(angle) * self.normal_tan + torch.sin(angle) * self.normal_cross
        bi_tan = torch.cross(self.normal_ori, tan, dim=-1)

        tan_dot = torch.matmul(tan, self.ls_t)
        bi_dot = torch.matmul(bi_tan, self.ls_t)
        e_power = lamb * tan_dot.square() + mu * bi_dot.square()
        basis = self.smooth * torch.exp(-e_power)
        envmap = torch.einsum("bnp,bnc->bcp", basis, weight)
        envmap = envmap.reshape(batch_size, 3, self.height, self.width)
        return torch.clamp_min(envmap, TINY_NUMBER).float()


class ASGNetModule(BaseModule):
    def __init__(
        self,
        img_log_dir=None,
        resolution=(256, 128),
        learning_rate=1e-4,
        lr_decay_steps=50,
        backbone_weights="DEFAULT",
        asg_loss_weight=0.1,
        vgg_loss_weight=0.01,
        **legacy_hparams,
    ):
        super().__init__()
        self.img_log_dir = img_log_dir
        self.asg_num = 128
        self.model = ASGNet(self.asg_num, backbone_weights=backbone_weights)
        self.smooth_l1_loss = nn.SmoothL1Loss()
        self.mae_loss = nn.L1Loss()
        self.vgg_loss = None

        self.asg_viewer = ASGViewer(resolution)

        self.learning_rate = learning_rate
        self.lr_decay_steps = lr_decay_steps
        self.asg_loss_weight = asg_loss_weight
        self.vgg_loss_weight = vgg_loss_weight
        self.save_hyperparameters(ignore=["legacy_hparams"])
        self.example_input_array = [
            [
                torch.Tensor(1, 3, 256, 256),
                torch.Tensor(1, 3, 256, 512),
                torch.Tensor(1, 128, 6),
                "name",
            ],
            0,
            "inference",
        ]

    def on_load_checkpoint(self, checkpoint):
        restore_frozen_loss(checkpoint, self)
        state_dict = checkpoint.get("state_dict")
        if state_dict is None:
            return
        for key in [
            "normal_tan",
            "normal_cross",
            "normal_ori",
            "bi_angle",
            "ls",
            "smooth",
            "area_weight_nchw",
            "lum_weight_nchw",
        ]:
            state_dict.pop(key, None)

    def asg_record(self, env_pre, env_gt, img_name, index=0):
        env_save = np.clip(to_hwc(env_pre, index), 0, 1)
        grid = (
            env_save
            if env_gt is None
            else np.concatenate((env_save, to_hwc(env_gt, index)), axis=0)
        )
        return {
            "name": name_at(img_name, index),
            "image": to_uint8(grid),
            "ext": ".jpg",
        }

    def get_loss(self, an_pre, la_pre, mu_pre, w_pre, an_gt, la_gt, mu_gt, w_gt):
        la_target = torch.clamp_min(la_gt, TINY_NUMBER)
        mu_target = torch.clamp_min(mu_gt, TINY_NUMBER)
        w_target = torch.clamp_min(w_gt, TINY_NUMBER)
        an_loss = self.smooth_l1_loss(
            torch.sin(2 * an_pre), torch.sin(2 * an_gt)
        ) + self.smooth_l1_loss(torch.cos(2 * an_pre), torch.cos(2 * an_gt))
        la_loss = self.smooth_l1_loss(torch.log1p(la_pre), torch.log1p(la_target))
        mu_loss = self.smooth_l1_loss(torch.log1p(mu_pre), torch.log1p(mu_target))
        w_loss = self.smooth_l1_loss(torch.log1p(w_pre), torch.log1p(w_target))
        return 0.1 * an_loss + la_loss + mu_loss + w_loss

    def asg2env(self, params):
        return self.asg_viewer(params)

    @torch.no_grad()
    def inference(self, rgb_imgs, img_names, is_save=True):
        an_pre, la_pre, mu_pre, w_pre = self.model(rgb_imgs)
        env_pres = self.asg_viewer([an_pre, la_pre, mu_pre, w_pre])
        env_pres = torch.clamp(env_pres, 0, 1)
        if is_save:
            env_saves = env_pres.permute(0, 2, 3, 1).detach().cpu().numpy() * 255
            for env_save, img_name in zip(env_saves, img_names):
                write_image(
                    Path(self.img_log_dir, f"{img_name}.jpg"), env_save.astype(np.uint8)
                )
                print(f"asg_net inference: {img_name}")
        return env_pres

    @torch.no_grad()
    def log_images(self, batch, split="train"):
        rgb_img, env_gt, asg_gt, img_name = batch
        an_pre, la_pre, mu_pre, w_pre = self.model(rgb_img)
        env_pre = self.asg2env([an_pre, la_pre, mu_pre, w_pre])
        return [self.asg_record(env_pre, env_gt, img_name)]

    @torch.no_grad()
    def log_wild_images(self, batch):
        rgb_img, img_name = batch
        rgb_img = rgb_img.to(self.device).float()
        an_pre, la_pre, mu_pre, w_pre = self.model(rgb_img)
        env_pre = self.asg2env([an_pre, la_pre, mu_pre, w_pre])
        return [self.asg_record(env_pre, None, img_name)]

    def forward(self, batch, batch_idx, stage):
        rgb_img, env_gt, asg_gt, img_name = batch
        bs = rgb_img.shape[0]
        an_gt, la_gt, mu_gt, w_gt = torch.split(asg_gt, [1, 1, 1, 3], dim=-1)
        an_pre, la_pre, mu_pre, w_pre = self.model(rgb_img)
        env_pre = self.asg2env([an_pre, la_pre, mu_pre, w_pre])
        if stage not in ["train", "val", "test"]:
            return env_pre
        env_pre_ldr = torch.clamp(env_pre, 0, 1)
        env_gt = torch.clamp(env_gt, 0, 1)
        pano_loss = self.mae_loss(env_pre_ldr, env_gt)
        asg_loss = self.get_loss(
            an_pre, la_pre, mu_pre, w_pre, an_gt, la_gt, mu_gt, w_gt
        )
        vgg_loss = env_pre.new_zeros(())
        if self.vgg_loss_weight > 0:
            vgg_loss = self.perceptual_loss()(env_pre_ldr, env_gt)
        total_loss = (
            pano_loss
            + self.asg_loss_weight * asg_loss
            + self.vgg_loss_weight * vgg_loss
        )
        log_info = {
            "tl": total_loss,
            "pano": pano_loss,
            "asg": asg_loss,
            "vgg": vgg_loss,
        }
        self.log_dict(
            prefixed(log_info, stage),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            prog_bar=True,
            batch_size=bs,
        )
        return total_loss
