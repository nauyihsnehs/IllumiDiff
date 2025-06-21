import os

import click
import torch

from lighting_est.dataset import PipeDataModule
from lighting_est.modules import (
    IDNetModule,
    SGNetModule,
    ASGNetModule,
    HDRNetModule,
)


@click.command()
@click.option('--task', type=click.Choice(['stage1', 'stage3', 'full']), default='full')  # full includes stage1 and stage3
@click.option('--id_ckpt', type=str, default='./ckpts/idnet-pers-e41-s10626-r256.ckpt')
@click.option('--id2_ckpt', type=str, default='./ckpts/idnet-pano-e41-s10626-r512.ckpt')
@click.option('--sg_ckpt', type=str, default='./ckpts/sg-epoch=49-step=56900.ckpt')
@click.option('--asg_ckpt', type=str, default='./ckpts/asg-epoch=99-step=155000.ckpt')
@click.option('--hdr_ckpt', type=str, default='./ckpts/hdr-epoch=49-step=28450.ckpt')
@click.option('--input_path', type=str, default='inputs')
@click.option('--input_pano_path', type=str, default='pano_ldr')
@click.option('--output_path', type=str, default='pano_hdr')
@click.option('--batch_size', type=int, default=16)
@click.option('--pano_res', type=(int, int), default=(512, 256))
@click.option('--sg_scale', type=float, default=1.0)
def pipeline(task, id_ckpt, id2_ckpt, sg_ckpt, asg_ckpt, hdr_ckpt,
             input_path, input_pano_path, output_path, batch_size, pano_res, sg_scale):
    def get_path(stage):
        save_path = output_path + f'/{stage}'
        os.makedirs(save_path, exist_ok=True)
        return save_path

    id_net_pers = IDNetModule.load_from_checkpoint(id_ckpt, img_log_dir=get_path('id_net_pers')).cuda() if task in ('stage1', 'stage3', 'full') else None
    id_net_pano = IDNetModule.load_from_checkpoint(id2_ckpt, img_log_dir=get_path('id_net_pano')).cuda() if task in ('stage3', 'full') else None
    sg_net = SGNetModule.load_from_checkpoint(sg_ckpt, img_log_dir=get_path('sg_net'), resolution=pano_res).cuda() if task in ('stage1', 'stage3', 'full') else None
    asg_net = ASGNetModule.load_from_checkpoint(asg_ckpt, img_log_dir=get_path('asg_net'), resolution=pano_res).cuda() if task in ('stage1', 'full') else None
    hdr_net = HDRNetModule.load_from_checkpoint(hdr_ckpt, img_log_dir=get_path('hdr_net')).cuda() if task in ('stage3', 'full') else None
    lum_weight = torch.tensor([0.0722, 0.7152, 0.2126])[None, :, None, None].cuda()  # b, c, h, w

    id_net_pers.eval() if id_net_pers else None
    id_net_pano.eval() if id_net_pano else None
    sg_net.eval() if sg_net else None
    asg_net.eval() if asg_net else None
    hdr_net.eval() if hdr_net else None

    dataset = PipeDataModule(batch_size=batch_size, input_path=input_path, input_pano_path=input_pano_path, resolution=pano_res, task=task)
    dataset.setup('predict')
    dataloader = dataset.predict_dataloader()

    with torch.no_grad():
        for input_imgs, input_panos, img_names in dataloader:
            input_imgs = input_imgs.cuda()
            input_panos = input_panos.cuda() if len(input_panos) > 0 else None

            asg_panos = asg_net.inference(input_imgs, img_names) if asg_net else None

            lum_masks_pers = id_net_pers.inference(input_imgs, img_names) if id_net_pers else None

            sg_panos = sg_net.inference(input_imgs * 2 - 1, lum_masks_pers, img_names) if sg_net else None

            if task in ('stage3', 'full'):
                lum_masks_pano = id_net_pano.inference(input_panos, img_names) if id_net_pano else None

                sg_lum = torch.sum(sg_panos * lum_weight, dim=1, keepdim=True).repeat(1, 3, 1, 1)
                sg_panos[sg_lum < 1] = 0.
                sg_panos = torch.log1p(sg_panos * sg_scale)

                hdr_panos = hdr_net.inference(input_panos, sg_panos, lum_masks_pano, img_names) if hdr_net else None


if __name__ == '__main__':
    pipeline()
