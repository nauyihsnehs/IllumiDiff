import os

from PIL import Image

from pano_gen.pano_tools import pers2pano

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import click
import numpy as np
import cv2 as cv
import torch
import torch.nn.functional as F
from lightning import seed_everything

from lighting_est.dataset import PipeDataModule
from lighting_est.modules import (
    IDNetModule,
    SGNetModule,
    ASGNetModule,
    HDRNetModule,
)
from pano_gen.cldm.ddim_hacked import DDIMSampler
from pano_gen.cldm.model import create_model, load_state_dict


def run_sampler(
        model, input_pano, input_asg, input_sg, input_img,
        img_name, output_path, input_mask=None,
        batch=4, strength=1.0, ddim_steps=50, eta=0.3,
        is_outpainting=True
):
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)

    if is_outpainting:
        mask = torch.stack([input_mask for _ in range(batch)], dim=0)
        mask = F.interpolate(mask, scale_factor=1 / 8, mode='bilinear', align_corners=False)
    else:
        mask = None

    C, H, W = input_pano.shape
    x0 = input_pano.clone()
    x0 = torch.stack([x0 for _ in range(batch)], dim=0)

    control = torch.stack([input_asg for _ in range(batch)], dim=0)
    control_hdr = torch.stack([input_sg for _ in range(batch)], dim=0)
    control_txt = torch.stack([input_img for _ in range(batch)], dim=0)
    cond = {"c_concat": [control],
            "c_hdr": [control_hdr],
            "c_crossattn": [model.get_learned_conditioning(control_txt)]}

    shape = (4, H // 8, W // 8)
    ddim_sampler.make_schedule(ddim_steps, ddim_eta=eta, verbose=True)
    x0 = model.get_first_stage_encoding(model.encode_first_stage(x0))
    model.control_scales = ([strength] * 13)

    samples, _ = ddim_sampler.sample(S=ddim_steps, batch_size=batch, shape=shape, conditioning=cond,
                                     eta=eta, x0=x0, mask=mask)
    x_samples = model.decode_first_stage(samples)

    score, best_score, best_sample = 0, 1e4, None
    input_img_ori = input_img * 0.5 + 0.5
    for id, x_sample in enumerate(x_samples):
        x_sample = (x_sample + 1) / 2
        x_sample = x_sample.flip(0)
        score = abs(x_sample.mean() - input_img_ori.mean())
        if score < best_score:
            best_score = score
            best_sample = x_sample
        ldr = torch.clamp(x_sample * 255, 0, 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        cv.imwrite(f'{output_path}/{img_name}_{str(id).zfill(2)}.jpg', ldr)
        print(f'pano_ldm inference: {img_name}, id: {str(id).zfill(2)}, score: {score.item()}')

    return best_sample


@click.command()
@click.option('--seed', type=int, default=3407)  # Set to -1 for random seed
@click.option('--task', type=click.Choice(['stage1', 'stage2', 'stage3', 'full']), default='full')
@click.option('--id_ckpt', type=str, default='./ckpts/idnet-pers-e41-s10626-r256.ckpt')
@click.option('--id2_ckpt', type=str, default='./ckpts/idnet-pano-e41-s10626-r512.ckpt')
@click.option('--sg_ckpt', type=str, default='./ckpts/sg-epoch=49-step=56900.ckpt')
@click.option('--asg_ckpt', type=str, default='./ckpts/asg-epoch=99-step=155000.ckpt')
@click.option('--hdr_ckpt', type=str, default='./ckpts/hdr-epoch=49-step=28450.ckpt')
@click.option('--input_path', type=str, default='inputs')
@click.option('--output_path', type=str, default='pano_hdr')
@click.option('--batch_size', type=int, default=16)
@click.option('--pano_res', type=(int, int), default=(512, 256))
@click.option('--sg_scale', type=float, default=1.0)
@click.option('--ldm_config', type=str, default='./pano_gen/configs/cldm_v15_clip_asg_sg.yaml')
@click.option('--ldm_ckpt', type=str, default='ckpts/control-epoch=4-step=47950.ckpt')
@click.option('--mask_path', type=str, default='pano_gen/outpainting-mask.png')
@click.option('--ldm_batch', type=int, default=4)  # Batch size for LDM inference, and select only the best one
@click.option('--ddim_steps', type=int, default=50)  # 20 for fast inference, 50 for better quality
@click.option('--eta', type=float, default=0.3)  # 0 ~ 1, higher for more diverse results
@click.option('--is_outpainting', type=bool, default=True)  # True for keeping the input image area
def pipeline(task, id_ckpt, id2_ckpt, sg_ckpt, asg_ckpt, hdr_ckpt,
             input_path, output_path, batch_size, pano_res, sg_scale,
             ldm_config, ldm_ckpt, mask_path, ldm_batch, ddim_steps, eta, seed, is_outpainting):
    if seed >= 0: seed_everything(seed)

    def get_path(stage):
        save_path = output_path + f'/{stage}'
        os.makedirs(save_path, exist_ok=True)
        return save_path

    id_net_pers = IDNetModule.load_from_checkpoint(id_ckpt, img_log_dir=get_path('id_net_pers')).cuda()
    sg_net = SGNetModule.load_from_checkpoint(sg_ckpt, img_log_dir=get_path('sg_net'), resolution=pano_res).cuda()
    asg_net = ASGNetModule.load_from_checkpoint(asg_ckpt, img_log_dir=get_path('asg_net'), resolution=pano_res).cuda()
    id_net_pano = IDNetModule.load_from_checkpoint(id2_ckpt, img_log_dir=get_path('id_net_pano')).cuda() if task in ('stage3', 'full') else None
    hdr_net = HDRNetModule.load_from_checkpoint(hdr_ckpt, img_log_dir=get_path('hdr_net')).cuda() if task in ('stage3', 'full') else None
    lum_weight = torch.tensor([0.0722, 0.7152, 0.2126])[None, :, None, None].cuda()  # BGR

    if task in ('stage2', 'stage3', 'full'):
        pano_ldm = create_model(ldm_config).cpu()
        pano_ldm.load_state_dict(load_state_dict(ldm_ckpt))
        pano_ldm.eval()
        in_mask = np.array(Image.open(mask_path).convert("L").resize(pano_res, resample=0)) / 255.0
        in_mask = torch.from_numpy(in_mask).float().cuda()[..., None].permute(2, 0, 1)
    else:
        pano_ldm = None

    id_net_pers.eval()
    sg_net.eval()
    asg_net.eval()
    id_net_pano.eval() if id_net_pano else None
    hdr_net.eval() if hdr_net else None

    dataset = PipeDataModule(batch_size=batch_size, input_path=input_path,
                             resolution=pano_res, task=task, is_full_pipe=True)
    dataset.setup('predict')
    dataloader = dataset.predict_dataloader()

    def norm(tensor):
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        return tensor * 2 - 1

    with torch.no_grad():
        for input_imgs, img_names in dataloader:
            input_imgs = input_imgs.cuda()

            asg_panos = asg_net.inference(input_imgs, img_names) if asg_net else None

            lum_masks_pers = id_net_pers.inference(input_imgs, img_names) if id_net_pers else None

            sg_panos = sg_net.inference(input_imgs * 2 - 1, lum_masks_pers, img_names) if sg_net else None

            if task in ('stage2', 'stage3', 'full'):  # TODO batch inference
                ldr_panos = []
                for input_img, asg_pano, sg_pano, img_name in zip(input_imgs, asg_panos, sg_panos, img_names):
                    input_img = input_img.permute(1, 2, 0).cpu().numpy()[..., ::-1]  # BGR to RGB
                    in_pano = pers2pano(input_img, pano_res, vfov=90, rotation_ear=[0, 0, 0])
                    in_pano = norm(torch.from_numpy(in_pano).float().cuda()).permute(2, 0, 1)
                    in_asg = torch.concat((norm(asg_pano.flip(0)), 1 - in_mask), dim=0)  # asg_pano is BGR
                    in_sg = torch.sum(sg_pano * lum_weight[0], dim=0, keepdim=True) - 1
                    input_img = cv.resize(input_img, (224, 224), interpolation=cv.INTER_AREA)
                    input_img = norm(torch.from_numpy(input_img).float().cuda()).permute(2, 0, 1)
                    ldr_panos.append(run_sampler(model=pano_ldm, input_pano=in_pano, input_asg=in_asg, input_sg=in_sg,
                                                 input_img=input_img, img_name=img_name, output_path=get_path('pano_ldm'), input_mask=in_mask,
                                                 batch=ldm_batch, strength=1.0, ddim_steps=ddim_steps, eta=eta, is_outpainting=is_outpainting))
                ldr_panos = torch.stack(ldr_panos)

            if task in ('stage3', 'full'):
                lum_masks_pano = id_net_pano.inference(ldr_panos, img_names) if id_net_pano else None
                sg_lum = torch.sum(sg_panos * lum_weight, dim=1, keepdim=True).repeat(1, 3, 1, 1)
                sg_panos[sg_lum < 1] = 0.
                sg_panos = torch.log1p(sg_panos * sg_scale)

                hdr_panos = hdr_net.inference(ldr_panos, sg_panos, lum_masks_pano, img_names) if hdr_net else None


if __name__ == '__main__':
    pipeline()
