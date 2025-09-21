import click
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from pytorch_lightning.callbacks import ModelCheckpoint
from pano_gen.cldm.logger import ImageLogger
from pano_gen.cldm.model import create_model, load_state_dict
from pano_gen.dataset import ControlLDMASGSGDataset

import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


@click.command()
@click.option('--gpus', default='1')
@click.option('--ckpt_path', type=str, default='ckpts/control_sd15_clip_asg_sg.ckpt')
@click.option('--data_root', type=str, default='/mnt/data1/ssy/ldr-inpainting/dataset')
@click.option('--bs', type=int, default=8)
@click.option('--lr', type=float, default=1e-4)
@click.option('--epochs', type=int, default=10)
@click.option('--logger_freq', type=int, default=300)
def main(gpus, ckpt_path, data_root, bs, lr, epochs, logger_freq):
    model = create_model('pano_gen/configs/cldm_v15_clip_asg_sg.yaml').cpu()

    model.load_state_dict(load_state_dict(ckpt_path, location='cpu'), strict=False)

    model.learning_rate = lr
    model.sd_locked = True
    model.only_mid_control = False

    dataset = ControlLDMASGSGDataset(size=512,
                                     img_root=f'{data_root}/input-rotate-flip-256',
                                     pano_root=f'{data_root}/pano-rotate-flip-512',
                                     asg_root=f'{data_root}/asg-rotate-flip-512',
                                     sg_root=f'{data_root}/sg-rotate-flip-512')

    dataloader = DataLoader(dataset, num_workers=8, batch_size=bs, shuffle=False, persistent_workers=True)
    logger = ImageLogger(batch_frequency=logger_freq)
    model_ckpt = ModelCheckpoint(every_n_epochs=1, save_top_k=-1, enable_version_counter=False)

    trainer = pl.Trainer(
        default_root_dir='logs/pano_gen',
        devices=gpus,
        precision='16-mixed',
        strategy='ddp_find_unused_parameters_true',
        max_epochs=epochs,
        callbacks=[model_ckpt, logger]
    )

    trainer.fit(model, dataloader)


if __name__ == '__main__':
    main()
