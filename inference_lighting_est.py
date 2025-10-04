import os

import click
import lightning as L
from lightning.pytorch.callbacks import TQDMProgressBar

from lighting_est.dataset import HDRNetDataModule, SGNetDataModule, ASGNetDataModule, IDNetDataModule
from lighting_est.modules import HDRNetModule, SGNetModule, ASGNetModule, IDNetModule


@click.command()
@click.option('--task', type=str, default='id_net')  # hdr_net, sg_net, asg_net, id_net
@click.option('--gpus', default=1)
@click.option('--img_log_dir', type=str, default='./results')
@click.option('--ckpt_path', type=str, default=None)
@click.option('--batch_size', type=int, default=8)
@click.option('--lr', type=float, default=1e-3)
@click.option('--res', type=(int, int), default=(256, 128))
@click.option('--max_count', type=int, default=None)
@click.option('--base_path', type=str, default='./illumidiff_dataset')
@click.option('--input_path', type=str, default='input_256')
@click.option('--input_ls_path', type=str, default='input_ls_256')  # sg_net only
# @click.option('--input_asg_path', type=str, default='input_asg_256')  # asg_net only
@click.option('--input_sg_path', type=str, default='sg_hdr')  # hdr_net only
@click.option('--input_pano_path', type=str, default='pano_ldr_512')  # hdr_net only
@click.option('--input_pano_ls_path', type=str, default='pano_ls_512')  # hdr_net only
@click.option('--sg_scale', type=float, default=1.0)  # hdr_net only
def predict(task, gpus, img_log_dir, ckpt_path, batch_size, lr, res, max_count,
            base_path, input_path, input_ls_path, #input_asg_path,
            input_sg_path, input_pano_path, input_pano_ls_path, sg_scale):
    img_log_dir = img_log_dir + f'/{task}_log'
    os.makedirs(img_log_dir, exist_ok=True)
    if task == 'hdr_net':
        model = HDRNetModule(img_log_dir=img_log_dir, learning_rate=lr)
        dataset = HDRNetDataModule(base_path, input_pano_path, input_sg_path, input_pano_ls_path, resolution=res, sg_scale=sg_scale, batch_size=batch_size)
    elif task == 'sg_net':
        model = SGNetModule(img_log_dir=img_log_dir, resolution=res, learning_rate=lr)
        dataset = SGNetDataModule(base_path, input_path, input_ls_path=input_ls_path, resolution=res, batch_size=batch_size, max_count=max_count)
    elif task == 'asg_net':
        model = ASGNetModule(img_log_dir=img_log_dir, resolution=res, learning_rate=lr)
        dataset = ASGNetDataModule(base_path, input_path, resolution=res, batch_size=batch_size, max_count=max_count)
    elif task == 'id_net':
        model = IDNetModule(img_log_dir=img_log_dir, learning_rate=lr)
        dataset = IDNetDataModule(base_path, input_path, resolution=res, batch_size=batch_size)
    else:
        raise ValueError(f'Invalid task: {task}')

    dataset.setup(stage='predict')

    prog_callback = TQDMProgressBar()
    trainer = L.Trainer(logger=False, devices=gpus, callbacks=[prog_callback])
    trainer.predict(model=model, datamodule=dataset, ckpt_path=ckpt_path)


if __name__ == '__main__':
    predict()
