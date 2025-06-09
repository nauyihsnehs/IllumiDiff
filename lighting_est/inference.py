import os

import click
import lightning as L
from lightning.pytorch.callbacks import TQDMProgressBar

from dataset import HDRNetDataModule, SGNetDataModule, ASGNetDataModule, IDNetDataModule
from modules import HDRNetModule, SGNetModule, ASGNetModule, IDNetModule


@click.command()
@click.option('--task', type=str, default='id_net')  # hdr_net, sg_net, asg_net, id_net
@click.option('--gpus', default='2,3,4,5,6,7')
@click.option('--img_log_dir', type=str, default='./results')
@click.option('--ckpt_path', type=str, default=None)
@click.option('--id_net_ckpt_path', type=str, default=None)
@click.option('--batch_size', type=int, default=8)
@click.option('--lr', type=float, default=1e-3)
@click.option('--h', type=int, default=256)
@click.option('--w', type=int, default=512)
@click.option('--base_path', type=str, default='')
@click.option('--input_path', type=str, default='input_256')
@click.option('--input_ls_path', type=str, default='input_ls_256')
@click.option('--input_asg_path', type=str, default='input_asg_256')
@click.option('--sg_pano_path', type=str, default='sg_hdr')
@click.option('--ldr_path', type=str, default='pano_ldr_512')
@click.option('--pano_ls_path', type=str, default='pano_ls_512')
def predict(task, gpus, img_log_dir, ckpt_path, base_path, input_path, input_ls_path, input_asg_path,
            sg_pano_path, id_net_ckpt_path, ldr_path, pano_ls_path, batch_size, lr, h, w):
    resolution = (w, h)
    img_log_dir = img_log_dir + f'/{task}_log'
    if not os.path.exists(img_log_dir):
        os.makedirs(img_log_dir)
    if not os.path.exists(img_log_dir + '/train'):
        os.makedirs(img_log_dir + '/train')
    if not os.path.exists(img_log_dir + '/val'):
        os.makedirs(img_log_dir + '/val')
    if not os.path.exists(img_log_dir + '/inference'):
        os.makedirs(img_log_dir + '/inference')
    if task == 'hdr_net':
        model = HDRNetModule(img_log_dir=img_log_dir, learning_rate=lr)
        dataset = HDRNetDataModule(base_path, ldr_path, sg_pano_path, pano_ls_path, resolution=resolution, batch_size=batch_size)
    elif task == 'sg_net':
        model = SGNetModule(img_log_dir=img_log_dir, resolution=resolution, learning_rate=lr)
        dataset = SGNetDataModule(base_path, input_path, input_ls_path=input_ls_path, resolution=resolution, batch_size=batch_size, id_net_ckpt_path=id_net_ckpt_path)
    elif task == 'asg_net':
        model = ASGNetModule(img_log_dir=img_log_dir, learning_rate=lr)
        dataset = ASGNetDataModule(base_path, input_asg_path, resolution=resolution, batch_size=batch_size)
    elif task == 'id_net':
        model = IDNetModule(img_log_dir=img_log_dir, learning_rate=lr)
        dataset = IDNetDataModule(base_path, input_path, resolution=resolution, batch_size=batch_size)
    else:
        raise ValueError(f'Invalid task: {task}')

    if isinstance(gpus, str):
        gpus_num = [int(gpu) for gpu in gpus.split(',')]
        if gpus_num[-1] == '':
            gpus_num.pop(-1)
        gpus_num = len(gpus_num)
    elif isinstance(gpus, int):
        gpus_num = gpus
    else:
        gpus_num = 1

    dataset.setup(stage='predict')
    if isinstance(dataset.data_num, list):
        data_num_per_gpu = [dn // gpus_num for dn in dataset.data_num]
    else:
        data_num_per_gpu = dataset.data_num // gpus_num
    model.get_data_num(data_num_per_gpu)

    prog_callback = TQDMProgressBar()

    trainer = L.Trainer(
        default_root_dir='logs',
        callbacks=[
            prog_callback,
        ]
    )

    trainer.predict(
        model=model, datamodule=dataset,
        ckpt_path=ckpt_path,
    )


if __name__ == '__main__':
    predict()
